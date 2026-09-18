"""Feature self-test: exercises every capability through the real API and reports PASS/FAIL.

    python scripts/selftest.py              # everything except the answer model (fast)
    python scripts/selftest.py --with-llm   # also starts the answer engine and asks real questions

It uses a throwaway data folder, so your indexed documents are untouched.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return bool(ok)


def sample_files() -> list[Path]:
    wanted = [
        ROOT / "samples" / "business" / "vendor_invoices.xlsx",
        ROOT / "samples" / "business" / "vendor_invoices.csv",
        ROOT / "samples" / "business" / "service_agreement_v1.pdf",
        ROOT / "samples" / "business" / "service_agreement_v2.pdf",
        ROOT / "samples" / "manuals" / "fault_code_manual.pdf",
        ROOT / "samples" / "manuals" / "maintenance_procedures.pdf",
        ROOT / "samples" / "manuals" / "technical_data_scanned.pdf",
        ROOT / "samples" / "manuals" / "site_memo.md",
    ]
    return [p for p in wanted if p.exists()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true", help="also start the answer engine and ask questions")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway data folder")
    args = ap.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="ragly-selftest-"))
    os.environ["RAGLY_DATA"] = str(data_dir)
    print(f"using a throwaway data folder: {data_dir}\n")

    from fastapi.testclient import TestClient

    from ragly_backend.api import api, app_state

    files = sample_files()
    if not files:
        print("No sample files found. Run scripts/make_samples.py, make_manual_samples.py and "
              "make_business_samples.py first.")
        return 2

    with TestClient(api) as client:
        app = app_state()

        # ---------- system / hardware ----------
        sysinfo = client.get("/api/system").json()
        check("system endpoint", sysinfo.get("version") is not None)
        hw = client.get("/api/hardware").json()
        check("hardware detection", "device" in hw and "backends" in hw,
              hw["device"]["summary"])
        check("no NPU claimed without runtime support",
              hw["backends"]["npu_claimed"] is False or hw["device"]["npu_available"] is True)
        diag = client.get("/api/diagnostics").json()
        check("diagnostics page data", diag.get("cloud_api_calls") == 0 and "models" in diag)
        check("offline guard active", diag["offline_guard"]["installed"] is True)
        reg = client.get("/api/registry").json()["models"]
        check("model registry", len(reg) >= 5,
              ", ".join(f"{m['key']}{'' if m['installed'] else ' (missing)'}" for m in reg))

        # ---------- projects ----------
        projects = client.get("/api/projects").json()
        check("default project exists", len(projects["projects"]) >= 1 and projects["active"],
              projects["projects"][0]["name"])
        made = client.post("/api/projects", json={"name": "Self-test project",
                                                  "description": "created by selftest"}).json()
        check("project created and activated", client.get("/api/projects").json()["active"] == made["id"])

        # ---------- ingestion ----------
        payload = [("files", (p.name, p.read_bytes())) for p in files]
        r = client.post("/api/documents", files=payload)
        check("multi-format upload", r.status_code == 200 and
              all("error" not in x for x in r.json()["results"]),
              f"{len(files)} files")

        deadline = time.time() + 600
        docs = []
        while time.time() < deadline:
            docs = client.get("/api/documents").json()["documents"]
            if docs and all(d["status"] in ("ready", "error") for d in docs):
                break
            time.sleep(1)
        failed = [d for d in docs if d["status"] != "ready"]
        check("all documents indexed", not failed,
              "; ".join(f"{d['name']}: {d['error']}" for d in failed) or f"{len(docs)} documents")

        counts = client.get("/api/system").json()
        check("pages and chunks stored", counts["index"]["pages"] > 0 and counts["index"]["chunks"] > 0,
              f"{counts['index']['pages']} pages, {counts['index']['chunks']} chunks")
        ctypes = counts.get("content_types", {})
        check("page content types classified", len(ctypes) >= 2, str(ctypes))
        check("OCR ran on the scanned document",
              any(d["ocr_pages"] > 0 for d in docs),
              f"OCR backend: {diag['ocr'].get('backend')}")
        check("images extracted", counts["images"]["images"] > 0, str(counts["images"]))
        check("duplicate upload is detected",
              client.post("/api/documents", files=[payload[0]]).json()["results"][0]["created"] is False)

        # ---------- auto-tuning ----------
        pack = client.get("/api/pack").json()
        check("auto-tuned pack built", bool(pack.get("pack")), pack.get("summary", ""))
        check("suggested questions", isinstance(client.get("/api/suggestions").json()["questions"], list))

        # ---------- retrieval ----------
        s = client.get("/api/search", params={"q": "termination notice period", "top_k": 5}).json()
        top = s["hits"][0] if s["hits"] else {}
        check("hybrid search finds the contract", top.get("doc_name", "").startswith("service_agreement"),
              f"{top.get('doc_name')} p.{top.get('page')} score {top.get('vector_score')}")
        s2 = client.get("/api/search", params={"q": "E-47", "top_k": 5}).json()
        check("exact code lookup", any(h.get("exact_match") for h in s2["hits"]),
              f"{s2['timing'].get('exact_hits')} exact hits")
        s3 = client.get("/api/search", params={"q": "everything about ABC Ltd", "top_k": 5}).json()
        check("entity graph contributes to retrieval", s3["timing"].get("graph_hits", 0) > 0,
              f"{s3['timing'].get('graph_hits')} graph hits")

        # ---------- router ----------
        routes = {q: client.post("/api/route", json={"question": q}).json()
                  for q in ["How much did we spend in March?", "Find photos of a damaged laptop",
                            "What documents relate to ABC Ltd?", "Compare these two contracts",
                            "What does the agreement say about confidentiality?"]}
        check("router picks table route", routes["How much did we spend in March?"]["primary"] == "table")
        check("router picks image route", routes["Find photos of a damaged laptop"]["primary"] == "image")
        check("router picks entity route", routes["What documents relate to ABC Ltd?"]["primary"] == "entity")
        check("router picks compare route", routes["Compare these two contracts"]["primary"] == "compare")
        check("router defaults to semantic",
              routes["What does the agreement say about confidentiality?"]["primary"] == "semantic")

        # ---------- tables ----------
        tables = client.get("/api/tables").json()["tables"]
        check("tables extracted with structure", len(tables) > 0 and tables[0]["n_cols"] > 1,
              f"{len(tables)} tables, e.g. {tables[0]['title']} {tables[0]['n_rows']}×{tables[0]['n_cols']}")
        comp = client.post("/api/tables/compute",
                           json={"question": "What was the total expenditure in March?"}).json()
        check("deterministic arithmetic", comp.get("answered") is True,
              comp.get("table", {}).get("workings", comp.get("reason", "")))
        filt = client.post("/api/tables/compute",
                           json={"question": "List invoices from ABC above 50000"}).json()
        check("table filtering", filt.get("answered") is True,
              filt.get("table", {}).get("workings", filt.get("reason", "")))
        refuse = client.post("/api/tables/compute",
                             json={"question": "What is the termination notice period?"}).json()
        check("table path refuses non-numeric questions", refuse.get("answered") is False)

        # ---------- entity graph ----------
        nodes = client.get("/api/graph/nodes").json()
        check("entity graph built", nodes["counts"]["nodes"] > 0 and nodes["counts"]["edges"] > 0,
              f"{nodes['counts']['nodes']} entities, {nodes['counts']['edges']} relationships")
        summary = client.get("/api/graph/summary", params={"q": "ABC Ltd"}).json()
        check("entity summary resolves documents", len(summary.get("documents", [])) > 0,
              f"{len(summary.get('documents', []))} documents, {len(summary.get('invoices', []))} invoices")
        if summary.get("entity"):
            nb = client.get(f"/api/graph/node/{summary['entity']['id']}").json()
            check("graph neighbourhood expands", len(nb.get("edges", [])) > 0, f"{len(nb['edges'])} edges")

        # ---------- images ----------
        imgs = client.get("/api/images").json()
        check("image index reports its mode", imgs["capabilities"]["mode"] in ("clip", "ocr_phash"),
              f"{imgs['capabilities']['mode']} · {imgs['capabilities'].get('vision_model') or 'no vision model'}")
        if imgs["images"]:
            first = imgs["images"][0]
            check("image file served", client.get(f"/api/images/{first['id']}/file").status_code == 200)
            hits = client.post("/api/images/search", json={"query": "technical data sheet"}).json()["results"]
            check("text → image search", isinstance(hits, list) and len(hits) > 0,
                  f"{len(hits)} hits, top reason: {hits[0]['match_reason'] if hits else '—'}")
            blob = Path(first["path"]).read_bytes()
            r2 = client.post("/api/images/search-by-image",
                             files={"file": ("q.png", blob, "image/png")}).json()["results"]
            check("image → image search", len(r2) > 0 and r2[0]["image_id"] == first["id"],
                  r2[0]["match_reason"] if r2 else "")
            r3 = client.post("/api/images/search-by-image", params={"mode": "documents"},
                             files={"file": ("q.png", blob, "image/png")}).json()["results"]
            check("image → document search", len(r3) > 0, f"{len(r3)} documents")

        # ---------- comparison ----------
        v1 = next((d for d in docs if d["name"] == "service_agreement_v1.pdf"), None)
        v2 = next((d for d in docs if d["name"] == "service_agreement_v2.pdf"), None)
        if v1 and v2:
            cmp = client.post("/api/compare", json={"left_id": v1["id"], "right_id": v2["id"]}).json()
            changed = [c for c in cmp["changes"] if c["kind"] == "modified"]
            values = [f"{v['from']}→{v['to']}" for c in changed for v in c["value_changes"]]
            check("document comparison finds value changes", cmp["summary"]["value_changes"] >= 3,
                  ", ".join(values[:4]))

        # ---------- page rendering ----------
        pdf_doc = next((d for d in docs if d["name"].lower().endswith(".pdf")), docs[0])
        png = client.get(f"/api/documents/{pdf_doc['id']}/pages/1")
        check("page rendered for the evidence viewer",
              png.status_code == 200 and png.content[:4] == b"\x89PNG",
              f"{pdf_doc['name']} · {len(png.content) // 1024} KB")
        sheet = next((d for d in docs if d["name"].lower().endswith((".csv", ".xlsx", ".md"))), None)
        if sheet:
            txt = client.get(f"/api/documents/{sheet['id']}/pagetext", params={"page": 1}).json()
            check("page text fallback for files with no page image", len(txt["text"]) > 0,
                  f"{sheet['name']} · {len(txt['text'])} chars")

        # ---------- settings ----------
        before = client.get("/api/settings").json()
        client.post("/api/settings", json={"top_k": 7, "strict_grounding": True})
        after = client.get("/api/settings").json()
        check("settings persist", after["top_k"] == 7)
        client.post("/api/settings", json={"top_k": before["top_k"]})

        # ---------- verification (no LLM needed: a stub proves the guard) ----------
        class Stub:
            def __init__(self, reply):
                self.reply = reply

            def stream_chat(self, messages, max_tokens=None):
                for w in self.reply.split(" "):
                    yield {"token": w + " "}
                yield {"stats": {"model": "stub", "tokens_per_sec": 99.0}}

        res = app.answerer.ask(Stub("The notice period is 60 days [1]."), "What is the termination notice period?")
        check("grounded answer is accepted", res["refused"] is False and res["citations"],
              res["answer"][:70])
        bad = app.answerer.ask(Stub("The notice period is 175 days [1]."), "What is the termination notice period?")
        check("invented number is blocked", bad["refused"] is True and
              bad["grounding"]["flag"] == "unsupported_values", str(bad["grounding"]["unsupported_values"]))
        hedge = app.answerer.ask(Stub("The notice period is typically 60 days [1]."), "notice period?")
        check("hedging is blocked", hedge["refused"] is True and hedge["grounding"]["flag"] == "hedged_language")
        off = app.answerer.ask(Stub("Paris is the capital of France."), "What is the capital of France?")
        check("unrelated question is refused", off["refused"] is True,
              off["grounding"]["flag"])
        tbl = app.answerer.ask(Stub("should not be used"), "What was the total expenditure in March?")
        check("numeric question answered by computation, not the model",
              tbl["grounding"]["flag"] == "deterministic_table", tbl["answer"][:80])

        # ---------- voice ----------
        voice = client.get("/api/voice").json()
        check("voice capability reported honestly", "stt" in voice and "tts" in voice,
              f"stt={voice['stt']} tts={voice['tts']} ({voice.get('stt_error') or 'available'})")

        # ---------- benchmarks ----------
        bench = client.post("/api/benchmark", params={"iterations": 2, "include_llm": False}).json()
        tasks = {b["task"] for b in bench["results"]}
        check("benchmarks measured on this machine", {"embed_text", "embed_query"} <= tasks,
              ", ".join(f"{b['task']} {b['latency_ms']}ms on {b['backend']}" for b in bench["results"]))
        check("benchmark history stored", len(client.get("/api/benchmark").json()["runs"]) > 0)

        # ---------- answer engine (optional) ----------
        if args.with_llm:
            try:
                app.engine.start()
                llm = app.engine.require_client()
                out = app.answerer.ask(llm, "What is the termination notice period?")
                check("real model answers with a citation",
                      out["refused"] is False and bool(out["citations"]), out["answer"][:90])
                out2 = app.answerer.ask(llm, "What is the warranty period for the transformer?")
                check("real model refuses what is not in the documents", out2["refused"] is True,
                      out2["answer"][:60])
            except Exception as exc:
                check("answer engine", False, str(exc))
        else:
            r = client.post("/api/ask", json={"question": "What is the termination notice period?"})
            check("ask endpoint responds (200 with the engine up, 409 without)",
                  r.status_code in (200, 409),
                  "engine ready" if r.status_code == 200 else "engine not started")

        # ---------- project isolation ----------
        other = client.post("/api/projects", json={"name": "Empty project"}).json()
        empty = client.get("/api/system").json()
        check("a new project starts empty", empty["index"]["documents"] == 0 and empty["tables"] == 0
              and empty["graph"]["nodes"] == 0, str(empty["index"]))
        check("search is scoped to the project",
              len(client.get("/api/search", params={"q": "termination notice"}).json()["hits"]) == 0)
        client.post(f"/api/projects/{made['id']}/activate")
        back = client.get("/api/system").json()
        check("switching back restores the documents", back["index"]["documents"] == len(docs) - 0,
              f"{back['index']['documents']} documents")
        check("empty project deleted", client.delete(f"/api/projects/{other['id']}").status_code == 200)

        # ---------- local passcode ----------
        st = client.get("/api/auth/status").json()
        check("auth status reported", "enabled" in st, f"enabled={st['enabled']}")
        if not st["enabled"]:
            reg = client.post("/api/auth/register", json={"passcode": "selftest-pass",
                                                          "display_name": "Self test"}).json()
            check("passcode set", bool(reg.get("token")))
            check("wrong passcode rejected",
                  client.post("/api/auth/login", json={"passcode": "nope"}).status_code == 401)
            good = client.post("/api/auth/login", json={"passcode": "selftest-pass"}).json()
            check("correct passcode accepted", bool(good.get("token")))
            check("session token validates",
                  client.post("/api/auth/check", json={"token": good["token"]}).json()["valid"] is True)
            check("forged token rejected",
                  client.post("/api/auth/check", json={"token": "1.2"}).json()["valid"] is False)
            client.post("/api/auth/disable", json={"passcode": "selftest-pass"})

        # ---------- deletion ----------
        last = docs[-1]
        check("document deleted", client.delete(f"/api/documents/{last['id']}").status_code == 200)
        check("deleted document is gone", client.get(f"/api/documents/{last['id']}").status_code == 404)

    failed = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 68)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    if not args.keep:
        shutil.rmtree(data_dir, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
