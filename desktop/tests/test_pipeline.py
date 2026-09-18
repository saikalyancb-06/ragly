"""End-to-end: index real sample files with the real embedder; the LLM is stubbed."""
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def stream_chat(self, messages, max_tokens=None):
        for w in self.reply.split(" "):
            yield {"token": w + " "}
        yield {"stats": {"model": "fake", "tokens_per_sec": 99.0}}


@pytest.fixture(scope="module")
def client():
    from ragly_backend.api import api

    with TestClient(api) as c:
        yield c


def wait_ready(c, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        docs = c.get("/api/documents").json()["documents"]
        if docs and all(d["status"] in ("ready", "error") for d in docs):
            return docs
        time.sleep(0.5)
    raise AssertionError("indexing timed out")


def test_upload_index_search_and_ask(client):
    files = [("files", (p.name, p.read_bytes())) for p in sorted(SAMPLES.iterdir()) if p.is_file()]
    r = client.post("/api/documents", files=files)
    assert r.status_code == 200
    docs = wait_ready(client)
    assert all(d["status"] == "ready" for d in docs), docs
    scanned = next(d for d in docs if d["name"] == "scanned_inspection.pdf")
    assert scanned["ocr_pages"] == 1

    # duplicate upload is detected
    again = client.post("/api/documents", files=[files[0]]).json()["results"][0]
    assert again["created"] is False

    hits = client.get("/api/search", params={"q": "termination notice period"}).json()["hits"]
    assert hits[0]["doc_name"] == "service_agreement.pdf" and hits[0]["page"] == 2

    hits = client.get("/api/search", params={"q": "fire extinguishers expired"}).json()["hits"]
    assert hits[0]["doc_name"] == "scanned_inspection.pdf"

    # engine is not running in tests -> ask returns 409, not a crash
    assert client.post("/api/ask", json={"question": "notice?"}).status_code == 409

    from ragly_backend.api import app_state

    a = app_state()
    res = a.answerer.ask(FakeLLM("Notice is 60 days [1] [9]."), "What is the termination notice period?")
    assert res["refused"] is False
    assert res["cited_numbers"] == [1] and res["dropped_citations"] == [9]
    assert res["citations"][0]["page"] == 2

    res = a.answerer.ask(FakeLLM("should not be called"), "What is the capital of France?")
    assert res["refused"] is True and res["grounding"]["flag"] == "low_relevance"

    res = a.answerer.ask(FakeLLM("Not found in your documents."), "What is the CEO salary under the agreement?")
    assert res["refused"] is True and res["citations"] == []

    # strict grounding: an invented number is caught even though the citation is valid
    res = a.answerer.ask(FakeLLM("The notice period is 90 days [1]."), "What is the termination notice period?")
    assert res["refused"] is True
    assert res["grounding"]["flag"] == "unsupported_values"
    assert "90 days" in res["grounding"]["unsupported_values"]

    # hedged wording is rejected too
    res = a.answerer.ask(FakeLLM("The notice period is typically 60 days [1]."), "notice period?")
    assert res["refused"] is True and res["grounding"]["flag"] == "hedged_language"

    png = client.get(f"/api/documents/{hits[0]['doc_id']}/pages/1", params={"chunk_id": hits[0]["chunk_id"]})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"

    sysinfo = client.get("/api/system").json()
    assert sysinfo["offline_guard"]["installed"] is True
    assert sysinfo["index"]["documents"] == len(files)   # every uploaded sample was indexed

    # toggle refuses Snapdragon on a non-Snapdragon machine
    if not sysinfo["device"]["is_snapdragon"]:
        assert client.post("/api/engine", json={"mode": "snapdragon"}).status_code == 409

    doc_id = docs[0]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
