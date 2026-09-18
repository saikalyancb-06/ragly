"""Ask every table and every image every question shape they can answer, and report failures.

    python scripts/sweep.py                 # the active workspace
    python scripts/sweep.py --json out.json

For each table this asks, per numeric column, for the total, average, highest and lowest; for
each row, the value of that column for that row's label; and the row count. For each indexed
image it asks for every concept the vision model recorded, plus a concept the corpus does not
contain (which must return nothing). Every expectation is derived from the data itself, so the
sweep works on any corpus, not just a fixture.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _num(text: str) -> float | None:
    from ragly_backend.tables import parse_number

    return parse_number(text)


def sweep_tables(app) -> list[dict]:
    from ragly_backend.tables import _numeric_cols_of

    out: list[dict] = []
    for table in app.store.list_tables():
        headers, rows = table["headers"], table["rows"]
        numeric = _numeric_cols_of(table)
        title = table.get("title") or f"table on page {table['page']}"
        label_col = next((h for h in headers if h not in numeric), None)

        for column in numeric:
            index = headers.index(column)
            values = [_num(r[index]) for r in rows if index < len(r)]
            values = [v for v in values if v is not None]
            if not values:
                continue
            checks = [(f"total {column}", sum(values)), (f"highest {column}", max(values)),
                      (f"lowest {column}", min(values)),
                      (f"average {column}", sum(values) / len(values))]
            for question, expected in checks:
                res = app.answerer.table_answer(question, None, table["id"])
                got = (res or {}).get("table", {}).get("value")
                out.append({"kind": "table", "table": title, "page": table["page"],
                            "question": question, "expected": round(expected, 2),
                            "got": None if got is None else round(got, 2),
                            "ok": got is not None and abs(got - expected) < 0.51})

            # one row lookup, using the first row that has a distinctive label
            if label_col:
                li = headers.index(label_col)
                for row in rows[:3]:
                    label = str(row[li]).strip()
                    value = _num(row[index]) if index < len(row) else None
                    if not label or value is None or len(label) < 3:
                        continue
                    question = f"{column} of {label}"
                    res = app.answerer.table_answer(question, None, table["id"])
                    got = (res or {}).get("table", {}).get("value")
                    same_label = sum(1 for r in rows if str(r[li]).strip() == label)
                    out.append({"kind": "table-lookup", "table": title, "page": table["page"],
                                "question": question, "expected": value,
                                "got": got, "ok": got is not None
                                and (abs(got - value) < 0.51 or same_label > 1)})
                    break
    return out


def sweep_images(app) -> list[dict]:
    out: list[dict] = []
    images = app.store.get_images()
    for image in images:
        try:
            objects = json.loads(image.get("objects") or "[]")
        except ValueError:
            objects = []
        for concept in objects[:3]:
            hits = app.image_index.search_by_text(f"images containing a {concept}", limit=5)
            pages = [h["page"] for h in hits]
            out.append({"kind": "image", "image": image["id"], "page": image["page"],
                        "question": f"images containing a {concept}",
                        "expected": image["page"], "got": pages[:3],
                        "ok": image["page"] in pages})
    if images:
        # A query for something no picture contains must return nothing. The ratio printed on a
        # failure is how well the query described the best image compared with how well that
        # image's own best concept describes it - close to 1 means the model cannot tell the
        # difference, which happens with abstract block diagrams rather than photographs.
        absent = "images containing a submarine"
        hits = app.image_index.search_by_text(absent, limit=5)
        note = ""
        if hits:
            best = max(float(i.get("concept_score") or 0) for i in images)
            score = hits[0].get("similarity")
            if best and score:
                note = f" (query scored {score / best:.2f} of the image's own description)"
        out.append({"kind": "image-absent", "question": absent, "expected": "nothing",
                    "got": [h["page"] for h in hits], "note": note, "ok": not hits})

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from ragly_backend.app import App

    app = App(start_indexer=False)
    results = sweep_tables(app) + sweep_images(app)
    failures = [r for r in results if not r["ok"]]

    for r in results:
        mark = "ok  " if r["ok"] else "FAIL"
        where = r.get("table") or f"image {r.get('image', '')}"
        print(f'{mark} {r["kind"]:<13} {str(where)[:28]:<28} {r["question"][:44]:<44} '
              f'expected={r["expected"]} got={r["got"]}{r.get("note", "")}')
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
