"""Entity names written by an older build are repaired in place, without re-indexing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragly_backend.store import Store  # noqa: E402


def test_sentence_words_are_stripped_and_merged(tmp_path):
    store = Store(tmp_path / "g.db")
    good = store.upsert_node("org", "Axis Bank", "axis bank")
    bad = store.upsert_node("org", "No Axis Bank", "no axis bank")
    store.upsert_node("org", "Profile Northstar Edge Analytics Pvt. Ltd.", "profile northstar edge analytics")
    store.add_edge(bad, "mentions", good, doc_id=1, page=1)

    fixed = store.clean_graph_nodes()

    names = sorted(r[0] for r in store.conn.execute("SELECT name FROM graph_nodes WHERE kind='org'"))
    assert fixed >= 2
    assert names == ["Axis Bank", "Northstar Edge Analytics Pvt. Ltd."]
    assert not list(store.conn.execute("SELECT 1 FROM graph_edges WHERE src=dst")), "self-links removed"


def test_cleanup_is_safe_to_run_twice(tmp_path):
    store = Store(tmp_path / "g2.db")
    store.upsert_node("org", "No Axis Bank", "no axis bank")
    store.clean_graph_nodes()
    assert store.clean_graph_nodes() == 0
