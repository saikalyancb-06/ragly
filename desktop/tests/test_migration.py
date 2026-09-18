"""A database created by an older build must keep working after an upgrade."""
import sqlite3

from ragly_backend.store import Store

OLD_SCHEMA = """
CREATE TABLE documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE, size_bytes INTEGER NOT NULL DEFAULT 0, pages INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'queued', created_at REAL NOT NULL);
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER NOT NULL, page INTEGER NOT NULL,
  ord INTEGER NOT NULL, text TEXT NOT NULL);
INSERT INTO documents(name, path, sha256, pages, status, created_at)
VALUES ('legacy.pdf', '/tmp/legacy.pdf', 'sha-legacy', 9, 'ready', 0);
INSERT INTO chunks(doc_id, page, ord, text) VALUES (1, 1, 0, 'legacy chunk text');
"""


def test_old_database_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(OLD_SCHEMA)
    con.commit()
    con.close()

    store = Store(db)

    # the columns added after the first release exist now
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(documents)")}
    assert {"table_count", "image_count", "ocr_pages", "index_seconds", "error"} <= cols
    assert "heading" in {r[1] for r in store.conn.execute("PRAGMA table_info(chunks)")}

    # the old row survived and the new columns are writable (this is the crash the user hit)
    store.set_status(1, "ready", table_count=2, image_count=1, ocr_pages=3, index_seconds=1.2, error=None)
    doc = store.get_document(1)
    assert doc["name"] == "legacy.pdf" and doc["table_count"] == 2 and doc["image_count"] == 1

    # the multimodal tables were created alongside
    assert store.graph_counts() == {"nodes": 0, "edges": 0, "mentions": 0}
    assert store.image_counts() == {"images": 0, "with_vectors": 0}
    assert store.list_tables() == []
    assert store.chunk_ids(1) == [1]


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(OLD_SCHEMA)
    con.commit()
    con.close()
    Store(db)
    store = Store(db)          # opening twice must not fail or duplicate columns
    names = [r[1] for r in store.conn.execute("PRAGMA table_info(documents)")]
    assert len(names) == len(set(names))
