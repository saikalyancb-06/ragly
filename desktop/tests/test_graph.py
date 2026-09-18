"""Entity graph: rule-based extraction, relation inference, build + retrieval.

Everything here runs against a real Store on a tmp_path database - no mocks, no
network, no models. The extraction is rules only, so these tests pin the
behaviour of the rules, not the accuracy of a model.
"""
import numpy as np
import pytest

from ragly_backend.chunker import Chunk
from ragly_backend.graph import (
    GraphBuilder,
    GraphRetriever,
    extract_entities,
    infer_relations,
    is_entity_question,
)
from ragly_backend.store import Store

INVOICE_TEXT = (
    "Invoice No. INV-1001 dated 12 March 2026 for INR 2,50,000 was raised by "
    "Kredo Automation Pvt Ltd, Bengaluru and billed to ABC Ltd. "
    "GSTIN 29ABCDE1234F1Z5 and PAN ABCDE1234F are on record for PO 4501 under Project Falcon. "
    "Signed by Mr. Rajesh Kumar, Finance Head at ABC Ltd."
)

CONTRACT_TEXT = (
    "This Agreement is made on 1 March 2026 between Acme Health Pvt Ltd (Client) and "
    "Kredo Automation (Provider) for a term of 24 months with a monthly fee of INR 1,50,000. "
    "The Master Services Agreement was reviewed by Dr. S. Menon in Bengaluru."
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "graph.db")
    yield s
    s.conn.close()


def seed_doc(store: Store, name: str, pages: list[tuple[int, str]]):
    """Add a ready document with real chunk rows; return (doc_id, chunks, chunk_ids)."""
    doc_id = store.add_document(name, f"/files/{name}", name, sum(len(t) for _, t in pages), len(pages))
    chunks = [Chunk(page=page, ord=i, text=text, heading="") for i, (page, text) in enumerate(pages)]
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    store.replace_chunks(doc_id, chunks, vectors)
    store.mark_ready(doc_id, chunk_count=len(chunks))
    chunk_ids = [r[0] for r in store.conn.execute(
        "SELECT id FROM chunks WHERE doc_id=? ORDER BY ord", (doc_id,))]
    return doc_id, chunks, chunk_ids


def kinds_of(entities, kind):
    return [e for e in entities if e["kind"] == kind]


def norms_of(entities, kind):
    return {e["norm"] for e in entities if e["kind"] == kind}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_invoice_paragraph_extraction():
    ents = extract_entities(INVOICE_TEXT)

    assert norms_of(ents, "org") == {"kredo automation", "abc"}
    assert norms_of(ents, "person") == {"rajesh kumar"}
    assert norms_of(ents, "date") == {"12 march 2026"}
    assert norms_of(ents, "amount") == {"inr 2,50,000"}
    assert norms_of(ents, "invoice") == {"inv1001"}
    assert norms_of(ents, "project") == {"project falcon"}
    assert norms_of(ents, "place") == {"bengaluru"}
    # GSTIN, PAN and the purchase order all land as id-like codes
    assert norms_of(ents, "id") == {"29abcde1234f1z5", "abcde1234f", "po4501"}

    # spans point at the text they came from, and the list is in document order
    for ent in ents:
        start, end = ent["span"]
        assert INVOICE_TEXT[start:end].strip(" .,")
    assert [e["span"][0] for e in ents] == sorted(e["span"][0] for e in ents)


def test_contract_paragraph_extraction():
    ents = extract_entities(CONTRACT_TEXT)

    assert norms_of(ents, "org") == {"acme health", "kredo automation"}
    # "This Agreement" loses its determiner; the titled agreement is its own node
    assert norms_of(ents, "contract") == {"agreement", "master services agreement"}
    assert norms_of(ents, "date") == {"1 march 2026"}
    assert norms_of(ents, "duration") == {"24 months"}
    assert norms_of(ents, "amount") == {"inr 1,50,000"}
    assert norms_of(ents, "person") == {"s menon"}          # "Dr. S. Menon"


def test_company_suffix_variants_share_one_norm():
    for text in ("ABC Ltd raised the bill.", "ABC Limited raised the bill.",
                 "M/s ABC Pvt. Ltd. raised the bill."):
        assert norms_of(extract_entities(text), "org") == {"abc"}


def test_junk_and_duplicates_are_dropped():
    ents = extract_entities("ABC Ltd, ABC Limited and ABC Ltd again.")
    assert len(kinds_of(ents, "org")) == 1                   # de-duplicated per call
    assert extract_entities("") == []


def test_generic_company_words_are_not_entities():
    """"The Company" has no name of its own: as a node it would link every
    contract in the corpus to every other one, so it is rejected."""
    assert kinds_of(extract_entities("The Company shall indemnify the Client."), "org") == []
    assert kinds_of(extract_entities("The Bank and the Hospital agreed."), "org") == []
    # a company with a real name is still recognised, suffix and all
    assert norms_of(extract_entities("Apollo Hospital and ABC Ltd agreed."), "org") == {
        "apollo hospital", "abc"}


def test_extraction_is_deterministic_and_capped():
    assert extract_entities(INVOICE_TEXT) == extract_entities(INVOICE_TEXT)
    many = " ".join(f"Invoice No. INV-{i} for INR {i},000 was raised." for i in range(1, 300))
    assert len(extract_entities(many)) <= 200


# --------------------------------------------------------------------------- #
# relations
# --------------------------------------------------------------------------- #
def test_invoice_relations():
    ents = extract_entities(INVOICE_TEXT)
    rels = set(infer_relations(ents, INVOICE_TEXT))

    assert ("invoice:inv1001", "issued_by", "org:kredo automation") in rels
    assert ("invoice:inv1001", "has_amount", "amount:inr 2,50,000") in rels
    assert ("invoice:inv1001", "dated", "date:12 march 2026") in rels
    assert ("invoice:inv1001", "belongs_to", "project:project falcon") in rels
    assert ("org:kredo automation", "located_in", "place:bengaluru") in rels
    assert ("person:rajesh kumar", "works_for", "org:abc") in rels
    # the invoice is attributed to the issuer after "raised by", not to the customer
    assert ("invoice:inv1001", "issued_by", "org:abc") not in rels


def test_contract_relations():
    ents = extract_entities(CONTRACT_TEXT)
    rels = set(infer_relations(ents, CONTRACT_TEXT))

    assert ("contract:agreement", "between", "org:acme health") in rels
    assert ("contract:agreement", "between", "org:kredo automation") in rels
    assert ("contract:agreement", "signed_on", "date:1 march 2026") in rels
    assert ("contract:agreement", "has_term", "duration:24 months") in rels


def test_co_occurrence_fallback_rules():
    text = "Kredo Automation Pvt Ltd and Project Falcon appear together here."
    rels = infer_relations(extract_entities(text), text)
    assert ("org:kredo automation", "co_occurs_with", "project:project falcon") in rels

    # no amount-to-date or date-to-amount fallback edges anywhere
    for src, rel, dst in infer_relations(extract_entities(INVOICE_TEXT), INVOICE_TEXT):
        if rel == "co_occurs_with":
            assert not (src.split(":")[0] in ("amount", "date") and dst.split(":")[0] in ("amount", "date"))

    # a specific relation suppresses the fallback for that pair
    rels = set(infer_relations(extract_entities(CONTRACT_TEXT), CONTRACT_TEXT))
    assert ("contract:agreement", "co_occurs_with", "org:acme health") not in rels
    assert ("org:acme health", "co_occurs_with", "contract:agreement") not in rels


def test_relations_are_deterministic():
    ents = extract_entities(CONTRACT_TEXT)
    assert infer_relations(ents, CONTRACT_TEXT) == infer_relations(ents, CONTRACT_TEXT)
    assert infer_relations([], CONTRACT_TEXT) == []


# --------------------------------------------------------------------------- #
# GraphBuilder
# --------------------------------------------------------------------------- #
def test_index_chunks_writes_nodes_edges_and_mentions(store):
    doc_id, chunks, chunk_ids = seed_doc(store, "invoice.pdf", [(1, INVOICE_TEXT)])
    counts = GraphBuilder(store).index_chunks(doc_id, chunks, chunk_ids)

    assert counts["chunks"] == 1
    assert counts["nodes"] >= 8 and counts["edges"] > 0 and counts["mentions"] >= 8
    graph = store.graph_counts()
    assert graph["nodes"] == counts["nodes"] and graph["edges"] == counts["edges"]

    # every mention carries the chunk id it came from
    rows = store.conn.execute("SELECT DISTINCT chunk_id FROM node_mentions WHERE doc_id=?", (doc_id,)).fetchall()
    assert [r[0] for r in rows] == chunk_ids


def test_indexing_twice_does_not_double_edges(store):
    doc_id, chunks, chunk_ids = seed_doc(store, "invoice.pdf", [(1, INVOICE_TEXT)])
    builder = GraphBuilder(store)

    first = builder.index_chunks(doc_id, chunks, chunk_ids)
    after_first = store.graph_counts()
    second = builder.index_chunks(doc_id, chunks, chunk_ids)
    after_second = store.graph_counts()

    assert after_first["edges"] == after_second["edges"]
    assert after_first["mentions"] == after_second["mentions"]
    assert after_first["nodes"] == after_second["nodes"]
    assert first["edges"] == second["edges"] and first["mentions"] == second["mentions"]


def test_index_document_pages_and_link_image(store):
    doc_id, _chunks, _ids = seed_doc(store, "contract.pdf", [(1, CONTRACT_TEXT)])
    counts = GraphBuilder(store).index_document(doc_id, [(1, CONTRACT_TEXT), (2, INVOICE_TEXT)])
    assert counts["pages"] == 2 and counts["nodes"] > 0

    # page-level indexing has pages but no chunk ids
    chunk_cols = store.conn.execute(
        "SELECT COUNT(*) FROM node_mentions WHERE doc_id=? AND chunk_id IS NOT NULL", (doc_id,)).fetchone()[0]
    assert chunk_cols == 0

    image_id = store.add_image({"doc_id": doc_id, "page": 2, "path": "/files/fig.png", "sha256": "img-sha"})
    linked = GraphBuilder(store).link_image(image_id, doc_id, 2)
    assert linked > 0

    node = GraphRetriever(store).resolve("ABC Ltd")[0]
    assert image_id in GraphRetriever(store).expand(node["id"])["images"]


# --------------------------------------------------------------------------- #
# GraphRetriever
# --------------------------------------------------------------------------- #
@pytest.fixture()
def seeded(store):
    """Two documents that only share entities, never wording."""
    invoice_doc, inv_chunks, inv_ids = seed_doc(store, "invoice.pdf", [(1, INVOICE_TEXT)])
    contract_doc, con_chunks, con_ids = seed_doc(store, "contract.pdf", [(3, CONTRACT_TEXT)])
    builder = GraphBuilder(store)
    builder.index_chunks(invoice_doc, inv_chunks, inv_ids)
    builder.index_chunks(contract_doc, con_chunks, con_ids)
    return {
        "store": store,
        "retriever": GraphRetriever(store),
        "invoice_doc": invoice_doc, "invoice_chunks": inv_ids,
        "contract_doc": contract_doc, "contract_chunks": con_ids,
    }


def test_resolve_exact_normalised_and_partial(seeded):
    retriever = seeded["retriever"]

    best = retriever.resolve("show me everything related to ABC Ltd")[0]
    assert (best["kind"], best["norm"], best["match"]) == ("org", "abc", "exact")

    # the other spelling of the same company resolves to the same node
    assert retriever.resolve("ABC Limited")[0]["id"] == best["id"]

    # punctuation-insensitive code lookup
    invoice = retriever.resolve("which vendor raised invoice INV 1001?")
    assert any(n["kind"] == "invoice" and n["norm"] == "inv1001" for n in invoice)

    # partial name match still finds the company
    assert any(n["norm"] == "kredo automation" for n in retriever.resolve("documents about Kredo"))
    assert retriever.resolve("nothing like this exists here") == [] or all(
        n["match"] == "partial" for n in retriever.resolve("nothing like this exists here"))


def test_expand_returns_neighbourhood_documents_and_pages(seeded):
    retriever = seeded["retriever"]
    node = retriever.resolve("Kredo Automation")[0]

    hood = retriever.expand(node["id"], hops=1)
    assert hood["seed"]["norm"] == "kredo automation"
    names = {n["name"] for n in hood["nodes"]}
    assert "INV-1001" in names and "Bengaluru" in names

    docs = {d["doc_name"]: d for d in hood["documents"]}
    assert docs["invoice.pdf"]["pages"] == [1]
    assert docs["contract.pdf"]["pages"] == [3]
    assert hood["edges"] and all("rel" in e and "weight" in e for e in hood["edges"])
    assert len(hood["nodes"]) <= 200

    # two hops reaches further but stays capped
    two = retriever.expand(node["id"], hops=2)
    assert len(two["nodes"]) >= len(hood["nodes"]) <= 200
    assert retriever.expand(node["id"], hops=9)["hops"] == 2


def test_chunk_ids_for_returns_seeded_chunk_ids(seeded):
    retriever = seeded["retriever"]
    seeded_ids = set(seeded["invoice_chunks"]) | set(seeded["contract_chunks"])

    ids, why = retriever.chunk_ids_for("show me everything related to ABC Ltd")
    assert ids and set(ids) <= seeded_ids
    assert set(seeded["invoice_chunks"]) <= set(ids)
    assert why and all({"node", "rel", "why"} <= set(row) for row in why)
    assert any(row["rel"] == "mentions" for row in why)

    # a company named in both files links the two documents through the graph
    ids, why = retriever.chunk_ids_for("everything related to Kredo Automation")
    assert set(ids) == seeded_ids
    assert len(ids) == len(set(ids))                          # no duplicates

    # the limit is honoured, and an unknown entity returns nothing
    assert retriever.chunk_ids_for("everything related to Kredo Automation", limit=1)[0] == ids[:1]
    assert retriever.chunk_ids_for("everything related to Zzzz Qqqq Ltd") == ([], [])


def test_summary_collects_invoices_documents_and_images(seeded):
    retriever = seeded["retriever"]
    out = retriever.summary("show me everything related to ABC Ltd")

    assert out["entity"]["kind"] == "org" and out["entity"]["norm"] == "abc"
    assert "Kredo Automation Pvt Ltd" in out["related"]["orgs"]
    assert "Mr. Rajesh Kumar" in out["related"]["people"]
    assert "Project Falcon" in out["related"]["projects"]

    invoice = next(i for i in out["invoices"] if i["invoice"] == "INV-1001")
    assert invoice["amounts"] == ["INR 2,50,000"]
    assert invoice["dates"] == ["12 March 2026"]
    assert invoice["issued_by"] == "Kredo Automation Pvt Ltd"

    assert {d["doc_name"] for d in out["documents"]} == {"invoice.pdf"}
    assert out["images"] == []
    assert retriever.summary("Zzzz Qqqq Ltd")["entity"] is None


# --------------------------------------------------------------------------- #
# router helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "show me everything related to ABC Ltd",
    "Show me all invoices from Kredo Automation",
    "documents about Project Falcon",
    "who is Dr. S. Menon?",
    "which vendor raised this bill?",
    "everything related to Acme Health Pvt Ltd",
    "what do we have on ABC Limited",
    "ABC Ltd",
])
def test_is_entity_question_true(question):
    assert is_entity_question(question) is True


@pytest.mark.parametrize("question", [
    "What is the termination notice period?",
    "Summarise clause 4",
    "What is the torque for bolt M8?",
    "How do I reset the breaker after fault E-47?",
    "Which values are out of range?",
    "",
])
def test_is_entity_question_false(question):
    assert is_entity_question(question) is False
