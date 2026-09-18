"""SQLite store: documents, chunks, FTS5 keyword index and embedding vectors (one local file)."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .config import DB_PATH, ensure_dirs

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  pages INTEGER NOT NULL DEFAULT 0,
  ocr_pages INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',   -- queued | indexing | ready | error
  error TEXT,
  index_seconds REAL,
  table_count INTEGER NOT NULL DEFAULT 0,
  image_count INTEGER NOT NULL DEFAULT 0,
  project_id INTEGER,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  ord INTEGER NOT NULL,
  heading TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  doc_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  value TEXT NOT NULL,
  norm TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(norm);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
END;
"""


class Store:   # noqa: E301  (MultimodalMixin is attached at the bottom)
    def __init__(self, path: Path | str = DB_PATH):
        ensure_dirs()
        self.path = str(path)
        self._lock = threading.RLock()
        # One SQLite connection per thread. The API answers requests on a thread pool while the
        # indexer works on its own thread; sharing a single connection between them interleaves
        # statements on the same cursor and SQLite rejects that with
        # "bad parameter or other API misuse". WAL lets those connections read while one writes.
        self._local = threading.local()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.executescript(SCHEMA_V2)
            self._migrate()
            # documents left mid-index by a crash go back to the queue
            self.conn.execute("UPDATE documents SET status='queued' WHERE status='indexing'")
            self.conn.commit()
        self.repair_counts()
        self._img_matrix = None
        self._img_ids = None
        self.active_project: int | None = None
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None
        self._matrix_docs: np.ndarray | None = None

    # ---------- migrations ----------
    # SQLite's CREATE TABLE IF NOT EXISTS never adds columns to a table that already
    # exists, so a database made by an older build keeps the old columns. Every column
    # added after the first release is listed here and applied with ALTER TABLE.
    MIGRATIONS: dict[str, dict[str, str]] = {
        "documents": {
            "table_count": "INTEGER NOT NULL DEFAULT 0",
            "image_count": "INTEGER NOT NULL DEFAULT 0",
            "ocr_pages": "INTEGER NOT NULL DEFAULT 0",
            "index_seconds": "REAL",
            "error": "TEXT",
            "project_id": "INTEGER",
        },
        "chunks": {"heading": "TEXT NOT NULL DEFAULT ''"},
        "images": {"caption_source": "TEXT NOT NULL DEFAULT ''",
                   "objects": "TEXT NOT NULL DEFAULT '[]'",
                   "concept_score": "REAL NOT NULL DEFAULT 0"},
        "tables": {"numeric_cols": "TEXT NOT NULL DEFAULT '[]'", "chunk_id": "INTEGER"},
    }

    def _rebuild_documents_if_globally_unique(self) -> bool:
        """Older builds had UNIQUE(sha256) on documents, which blocked the same file in two projects.

        SQLite cannot drop a column constraint, so the table is rebuilt once, preserving every row.
        """
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'").fetchone()
        if not row or "UNIQUE" not in (row[0] or ""):
            return False
        body = SCHEMA.split("CREATE TABLE IF NOT EXISTS documents (")[1].split(");")[0]
        create_new = "CREATE TABLE documents_new (" + body + ");"
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(documents)")]
        new_cols = [c.split()[0] for c in body.strip().splitlines() if c.strip() and not c.strip().startswith("--")]
        shared = ", ".join(c for c in cols if c in new_cols)
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute(create_new)
        self.conn.execute(f"INSERT INTO documents_new ({shared}) SELECT {shared} FROM documents")
        self.conn.execute("DROP TABLE documents")
        self.conn.execute("ALTER TABLE documents_new RENAME TO documents")
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=ON")
        return True

    def _migrate(self) -> None:
        """Bring an older database up to the current schema, without losing data."""
        applied = []
        for table, columns in self.MIGRATIONS.items():
            try:
                existing = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            except sqlite3.OperationalError:
                continue
            if not existing:          # table does not exist yet: SCHEMA created it correctly
                continue
            for name, decl in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    applied.append(f"{table}.{name}")
        if applied:
            self.conn.commit()
            logging.getLogger("ragly.store").info("database migrated: added %s", ", ".join(applied))
        try:
            rebuilt = self._rebuild_documents_if_globally_unique()
            # the per-project uniqueness index needs project_id, so it is created after the migration
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash_project ON documents(sha256, project_id)")
            self.conn.commit()
            if rebuilt:
                self.conn.executescript(SCHEMA)     # recreate indexes and triggers on the new table
                logging.getLogger("ragly.store").info(
                    "database migrated: documents are now unique per project, not globally")
        except sqlite3.Error as exc:
            logging.getLogger("ragly.store").warning("documents table rebuild skipped: %s", exc)

    # ---------- connection (one per thread) ----------
    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
        return conn

    # ---------- projects ----------
    def ensure_default_project(self) -> int:
        """Every document belongs to a project; older databases get one automatically."""
        row = self.conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()
        if row:
            pid = int(row[0])
        else:
            pid = self.create_project("My workspace", "Default project")
        with self._lock:
            self.conn.execute("UPDATE documents SET project_id=? WHERE project_id IS NULL", (pid,))
            self.conn.commit()
        return pid

    def create_project(self, name: str, description: str = "") -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO projects(name, description, created_at) VALUES (?,?,?)",
                (name.strip(), description.strip(), time.time()))
            self.conn.commit()
            return int(cur.lastrowid)

    def list_projects(self) -> list[dict]:
        rows = self.conn.execute("""
            SELECT p.*,
                   (SELECT COUNT(*) FROM documents d WHERE d.project_id = p.id) AS documents,
                   (SELECT COALESCE(SUM(d.chunk_count),0) FROM documents d WHERE d.project_id = p.id) AS chunks,
                   (SELECT COALESCE(SUM(d.image_count),0) FROM documents d WHERE d.project_id = p.id) AS images,
                   (SELECT COALESCE(SUM(d.table_count),0) FROM documents d WHERE d.project_id = p.id) AS tables
            FROM projects p ORDER BY p.created_at""")
        return [dict(r) for r in rows]

    def get_project(self, project_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return dict(row) if row else None

    def rename_project(self, project_id: int, name: str, description: str | None = None) -> None:
        with self._lock:
            if description is None:
                self.conn.execute("UPDATE projects SET name=? WHERE id=?", (name.strip(), project_id))
            else:
                self.conn.execute("UPDATE projects SET name=?, description=? WHERE id=?",
                                  (name.strip(), description.strip(), project_id))
            self.conn.commit()

    def delete_project(self, project_id: int) -> list[dict]:
        """Deletes the project and everything indexed inside it. Returns the removed documents."""
        docs = [d for d in self.list_documents(all_projects=True) if d["project_id"] == project_id]
        for d in docs:
            self.delete_document(d["id"])
        with self._lock:
            self.conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            self.conn.commit()
        return docs

    def project_doc_ids(self, project_id: int | None) -> list[int] | None:
        if project_id is None:
            return None
        return [r[0] for r in self.conn.execute("SELECT id FROM documents WHERE project_id=?", (project_id,))]

    def scope(self, project_id: int | None) -> None:
        """Limit every read to one project. None means 'everything'."""
        self.active_project = project_id
        self._invalidate()
        self._img_matrix = None

    def _scoped(self, doc_ids: list[int] | None) -> list[int] | None:
        """Intersect an explicit document filter with the active project scope (always live)."""
        if self.active_project is None:
            return doc_ids
        allowed = set(self.project_doc_ids(self.active_project) or [])
        if doc_ids is None:
            return sorted(allowed)
        return [d for d in doc_ids if d in allowed]

    def _scope_sql(self, column: str = "d.id") -> tuple[str, list]:
        """A live subquery, so documents added after the switch are still in scope."""
        if self.active_project is None:
            return "", []
        return f" AND {column} IN (SELECT id FROM documents WHERE project_id = ?)", [self.active_project]

    # ---------- meta ----------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
            self.conn.commit()

    # ---------- documents ----------
    def find_by_hash(self, sha: str, project_id: int | None = "active") -> dict | None:
        """A file already imported into ANOTHER project is not a duplicate here."""
        if project_id == "active":
            project_id = self.active_project
        if project_id is None:
            row = self.conn.execute("SELECT * FROM documents WHERE sha256=?", (sha,)).fetchone()
        else:
            row = self.conn.execute("SELECT * FROM documents WHERE sha256=? AND project_id=?",
                                    (sha, project_id)).fetchone()
        return dict(row) if row else None

    def add_document(self, name: str, path: str, sha: str, size: int, pages: int,
                     project_id: int | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO documents(name, path, sha256, size_bytes, pages, project_id, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, path, sha, size, pages, project_id if project_id is not None else self.active_project,
                 time.time()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def repair_counts(self) -> int:
        """Make the counters on each document agree with what is actually stored.

        A document interrupted mid-index keeps the chunks it had already written while its
        counter stays at 0 — the row then reads "ready, 0 chunks", or sits at "queued" with
        hundreds of orphan chunks. Counting the real rows is cheap and makes the numbers on
        screen true.
        """
        with self._lock:
            cur = self.conn.execute("""
                UPDATE documents SET
                    chunk_count = (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = documents.id),
                    table_count = (SELECT COUNT(*) FROM tables t WHERE t.doc_id = documents.id),
                    image_count = (SELECT COUNT(*) FROM images i WHERE i.doc_id = documents.id)
                WHERE chunk_count <> (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = documents.id)
                   OR table_count <> (SELECT COUNT(*) FROM tables t WHERE t.doc_id = documents.id)
                   OR image_count <> (SELECT COUNT(*) FROM images i WHERE i.doc_id = documents.id)""")
            self.conn.commit()
            return cur.rowcount or 0

    def set_status(self, doc_id: int, status: str, **fields) -> None:
        cols = ["status=?"] + [f"{k}=?" for k in fields]
        with self._lock:
            self.conn.execute(f"UPDATE documents SET {', '.join(cols)} WHERE id=?", (status, *fields.values(), doc_id))
            self.conn.commit()

    def get_document(self, doc_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, all_projects: bool = False) -> list[dict]:
        sql = "SELECT * FROM documents WHERE 1=1"
        args: list = []
        if not all_projects:
            clause, a = self._scope_sql("id")
            sql += clause
            args += a
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def queued_ids(self) -> list[int]:
        return [r[0] for r in self.conn.execute("SELECT id FROM documents WHERE status='queued' ORDER BY id")]

    def delete_document(self, doc_id: int) -> dict | None:
        doc = self.get_document(doc_id)
        if not doc:
            return None
        with self._lock:
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self.conn.commit()
            self._invalidate()
        return doc

    def path_in_use(self, path: str, ignore_doc_id: int | None = None) -> bool:
        """The same file imported into two workspaces shares one copy on disk: never delete it
        while another workspace still points at it."""
        sql = "SELECT COUNT(*) FROM documents WHERE path=?"
        args: list = [path]
        if ignore_doc_id is not None:
            sql += " AND id<>?"
            args.append(ignore_doc_id)
        return int(self.conn.execute(sql, args).fetchone()[0]) > 0

    # ---------- chunks + vectors ----------
    def replace_chunks(self, doc_id: int, chunks: list, vectors: np.ndarray, entities: list | None = None) -> None:
        """entities: list aligned with chunks, each [(kind, value, norm), ...]."""
        assert len(chunks) == len(vectors)
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                self.conn.execute("DELETE FROM entities WHERE doc_id=?", (doc_id,))
                for i, (ch, vec) in enumerate(zip(chunks, vectors)):
                    cur = self.conn.execute(
                        "INSERT INTO chunks(doc_id, page, ord, heading, text) VALUES (?,?,?,?,?)",
                        (doc_id, ch.page, ch.ord, ch.heading, ch.text),
                    )
                    self.conn.execute(
                        "INSERT INTO embeddings(chunk_id, vec) VALUES (?,?)",
                        (cur.lastrowid, np.asarray(vec, dtype=np.float32).tobytes()),
                    )
                    for kind, value, norm in (entities[i] if entities else []):
                        self.conn.execute(
                            "INSERT INTO entities(chunk_id, doc_id, kind, value, norm) VALUES (?,?,?,?,?)",
                            (cur.lastrowid, doc_id, kind, value, norm),
                        )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            self._invalidate()

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"""SELECT c.id, c.doc_id, c.page, c.ord, c.heading, c.text, d.name AS doc_name,
                       COALESCE(p.content_type, 'text') AS content_type
                FROM chunks c JOIN documents d ON d.id=c.doc_id
                LEFT JOIN pages p ON p.doc_id = c.doc_id AND p.page = c.page
                WHERE c.id IN ({q})""",
            ids,
        )
        return {r["id"]: dict(r) for r in rows}

    def chunk_ids(self, doc_id: int) -> list[int]:
        return [r[0] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE doc_id=? ORDER BY ord, id", (doc_id,))]

    def counts(self) -> dict:
        clause, args = self._scope_sql("id")
        d = self.conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(pages),0) FROM documents WHERE status='ready'{clause}", args).fetchone()
        cclause, cargs = self._scope_sql("c.doc_id")
        c = self.conn.execute(f"SELECT COUNT(*) FROM chunks c WHERE 1=1{cclause}", cargs).fetchone()[0]
        return {"documents": d[0], "pages": d[1], "chunks": c}

    def _invalidate(self) -> None:
        self._matrix = self._matrix_ids = self._matrix_docs = None

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All vectors as one (N, dim) matrix, cached until the index changes."""
        with self._lock:
            if self._matrix is None:
                clause, args = self._scope_sql("d.id")
                rows = self.conn.execute(
                    f"""SELECT e.chunk_id, c.doc_id, e.vec FROM embeddings e
                        JOIN chunks c ON c.id=e.chunk_id
                        JOIN documents d ON d.id=c.doc_id WHERE d.status='ready'{clause}""", args
                ).fetchall()
                if rows:
                    self._matrix_ids = np.array([r[0] for r in rows], dtype=np.int64)
                    self._matrix_docs = np.array([r[1] for r in rows], dtype=np.int64)
                    self._matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
                else:
                    self._matrix_ids = np.zeros(0, dtype=np.int64)
                    self._matrix_docs = np.zeros(0, dtype=np.int64)
                    self._matrix = np.zeros((0, 1), dtype=np.float32)
            return self._matrix, self._matrix_ids, self._matrix_docs

    def mark_ready(self, doc_id: int, **fields) -> None:
        self.set_status(doc_id, "ready", **fields)
        with self._lock:
            self._invalidate()

    def fts_search(self, match: str, limit: int, doc_ids: list[int] | None = None) -> list[int]:
        sql = """SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id=f.rowid
                 JOIN documents d ON d.id=c.doc_id
                 WHERE chunks_fts MATCH ? AND d.status='ready'"""
        args: list = [match]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            args += list(doc_ids)
        clause, cargs = self._scope_sql("c.doc_id")
        sql += clause
        args += cargs
        sql += " ORDER BY bm25(chunks_fts, 1.0, 0.3) LIMIT ?"
        args.append(limit)
        try:
            return [r[0] for r in self.conn.execute(sql, args)]
        except sqlite3.OperationalError:
            return []

    def entity_chunks(self, norms: list[str], limit: int = 10, doc_ids: list[int] | None = None) -> list[int]:
        doc_ids = self._scoped(doc_ids)
        """Chunk ids that literally contain one of these codes / part numbers."""
        if not norms:
            return []
        q = ",".join("?" * len(norms))
        sql = f"""SELECT DISTINCT e.chunk_id FROM entities e
                  JOIN chunks c ON c.id = e.chunk_id
                  JOIN documents d ON d.id = c.doc_id
                  WHERE e.norm IN ({q}) AND d.status='ready'"""
        args: list = list(norms)
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            args += list(doc_ids)
        clause, cargs = self._scope_sql("c.doc_id")
        sql += clause
        args += cargs
        sql += " LIMIT ?"
        args.append(limit)
        return [r[0] for r in self.conn.execute(sql, args)]

    def all_chunk_texts(self, limit: int = 4000) -> list[str]:
        clause, args = self._scope_sql("d.id")
        return [r[0] for r in self.conn.execute(
            f"SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id "
            f"WHERE d.status='ready'{clause} LIMIT ?", [*args, limit])]

    def entity_summary(self, kind: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.value, e.kind, d.name AS doc_name, c.page, e.chunk_id
               FROM entities e JOIN chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.doc_id
               WHERE e.kind=? ORDER BY e.value LIMIT ?""", (kind, limit))
        return [dict(r) for r in rows]

    def chunks_with_content_type(self, ids: list[int], types: list[str]) -> set[int]:
        """Metadata filter: keep only chunks whose page has one of these content types."""
        if not ids or not types:
            return set(ids)
        qi = ",".join("?" * len(ids))
        qt = ",".join("?" * len(types))
        rows = self.conn.execute(
            f"""SELECT c.id FROM chunks c JOIN pages p ON p.doc_id=c.doc_id AND p.page=c.page
                WHERE c.id IN ({qi}) AND p.content_type IN ({qt})""", [*ids, *types])
        return {r[0] for r in rows}

    def reset_index(self) -> None:
        """Drop all chunks/vectors and requeue every document (e.g. embedding model changed)."""
        with self._lock:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM entities")
            self.conn.execute("UPDATE documents SET status='queued', chunk_count=0")
            self.conn.commit()
            self._invalidate()


# ---------------------------------------------------------------------------
# Multimodal schema: pages, tables, images, vectors, entity graph, caches.
# Applied on top of SCHEMA at startup (all statements are IF NOT EXISTS).
# ---------------------------------------------------------------------------
SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  content_type TEXT NOT NULL DEFAULT 'text',   -- text | scanned_text | table | image | diagram | mixed | empty
  char_count INTEGER NOT NULL DEFAULT 0,
  ocr_used INTEGER NOT NULL DEFAULT 0,
  has_table INTEGER NOT NULL DEFAULT 0,
  has_image INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL DEFAULT '',
  UNIQUE(doc_id, page)
);
CREATE TABLE IF NOT EXISTS tables (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  headers TEXT NOT NULL,          -- JSON list[str]
  rows TEXT NOT NULL,             -- JSON list[list[str]]
  n_rows INTEGER NOT NULL DEFAULT 0,
  n_cols INTEGER NOT NULL DEFAULT 0,
  numeric_cols TEXT NOT NULL DEFAULT '[]',   -- JSON list[str]
  chunk_id INTEGER                -- the chunk holding this table's text form
);
CREATE INDEX IF NOT EXISTS idx_tables_doc ON tables(doc_id);
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL DEFAULT 1,
  path TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'embedded',   -- embedded | page_render | upload
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  sha256 TEXT NOT NULL,
  phash TEXT NOT NULL DEFAULT '',
  ocr_text TEXT NOT NULL DEFAULT '',
  caption TEXT NOT NULL DEFAULT '',
  caption_source TEXT NOT NULL DEFAULT '',   -- '' | vision_model_name | ocr
  objects TEXT NOT NULL DEFAULT '[]',        -- concepts the vision model found in the picture
  concept_score REAL NOT NULL DEFAULT 0,     -- how strongly the best concept describes it
  UNIQUE(sha256, doc_id, page)
);
CREATE INDEX IF NOT EXISTS idx_images_doc ON images(doc_id);
CREATE TABLE IF NOT EXISTS image_vectors (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(ocr_text, caption, content='');
CREATE TABLE IF NOT EXISTS graph_nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,             -- person | org | product | place | date | invoice | contract | project | amount | id
  name TEXT NOT NULL,
  norm TEXT NOT NULL,
  mentions INTEGER NOT NULL DEFAULT 0,
  UNIQUE(kind, norm)
);
CREATE TABLE IF NOT EXISTS graph_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  rel TEXT NOT NULL,
  dst INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  doc_id INTEGER,
  page INTEGER,
  chunk_id INTEGER,
  weight REAL NOT NULL DEFAULT 1.0,
  UNIQUE(src, rel, dst, doc_id, page)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON graph_edges(dst);
CREATE TABLE IF NOT EXISTS node_mentions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node_id INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  doc_id INTEGER NOT NULL,
  page INTEGER NOT NULL,
  chunk_id INTEGER,
  image_id INTEGER,
  UNIQUE(node_id, doc_id, page, chunk_id, image_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_node ON node_mentions(node_id);
CREATE TABLE IF NOT EXISTS embed_cache (
  key TEXT PRIMARY KEY,           -- sha256(model + text)
  vec BLOB NOT NULL,
  used_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ran_at REAL NOT NULL,
  task TEXT NOT NULL,             -- embed_text | embed_image | ocr | llm_generate | index | query
  model TEXT NOT NULL,
  backend TEXT NOT NULL,          -- the execution provider actually used
  latency_ms REAL,
  throughput REAL,
  memory_mb REAL,
  extra TEXT NOT NULL DEFAULT '{}'
);
"""


class MultimodalMixin:
    """Mixed into Store (see bottom of file) - keeps the multimodal SQL in one place."""

    # ---------------- pages ----------------
    def replace_pages(self, doc_id: int, pages: list[dict]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
            self.conn.executemany(
                """INSERT OR REPLACE INTO pages(doc_id, page, content_type, char_count, ocr_used,
                                                has_table, has_image, text)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(doc_id, p["page"], p.get("content_type", "text"), p.get("char_count", 0),
                  int(p.get("ocr_used", False)), int(p.get("has_table", False)),
                  int(p.get("has_image", False)), p.get("text", "")) for p in pages],
            )
            self.conn.commit()

    def get_pages(self, doc_id: int) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM pages WHERE doc_id=? ORDER BY page", (doc_id,))]

    def page_text(self, doc_id: int, page: int) -> str:
        row = self.conn.execute("SELECT text FROM pages WHERE doc_id=? AND page=?", (doc_id, page)).fetchone()
        return row[0] if row else ""

    def content_type_counts(self) -> dict:
        clause, args = self._scope_sql("d.id")
        rows = self.conn.execute(
            f"""SELECT p.content_type, COUNT(*) FROM pages p JOIN documents d ON d.id=p.doc_id
                WHERE d.status='ready'{clause} GROUP BY p.content_type""", args)
        return {r[0]: r[1] for r in rows}

    # ---------------- tables ----------------
    def replace_tables(self, doc_id: int, tables: list[dict]) -> None:
        import json as _json
        with self._lock:
            self.conn.execute("DELETE FROM tables WHERE doc_id=?", (doc_id,))
            for t in tables:
                self.conn.execute(
                    """INSERT INTO tables(doc_id, page, title, headers, rows, n_rows, n_cols, numeric_cols, chunk_id)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (doc_id, t["page"], t.get("title", ""), _json.dumps(t["headers"]), _json.dumps(t["rows"]),
                     len(t["rows"]), len(t["headers"]), _json.dumps(t.get("numeric_cols", [])), t.get("chunk_id")),
                )
            self.conn.commit()

    def list_tables(self, doc_ids: list[int] | None = None) -> list[dict]:
        import json as _json
        sql = """SELECT t.*, d.name AS doc_name FROM tables t JOIN documents d ON d.id=t.doc_id
                 WHERE d.status='ready'"""
        clause, args = self._scope_sql("t.doc_id")
        sql += clause
        if doc_ids is not None:
            if not doc_ids:
                return []
            sql += f" AND t.doc_id IN ({','.join('?' * len(doc_ids))})"
            args += list(doc_ids)
        out = []
        for r in self.conn.execute(sql, args):
            d = dict(r)
            d["headers"] = _json.loads(d["headers"])
            d["rows"] = _json.loads(d["rows"])
            d["numeric_cols"] = _json.loads(d["numeric_cols"])
            out.append(d)
        return out

    def get_table(self, table_id: int) -> dict | None:
        tables = [t for t in self.list_tables() if t["id"] == table_id]
        return tables[0] if tables else None

    # ---------------- images ----------------
    def add_image(self, rec: dict) -> int | None:
        with self._lock:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO images(doc_id, page, path, source, width, height, sha256,
                                                phash, ocr_text, caption, caption_source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.get("doc_id"), rec.get("page", 1), rec["path"], rec.get("source", "embedded"),
                 rec.get("width", 0), rec.get("height", 0), rec["sha256"], rec.get("phash", ""),
                 rec.get("ocr_text", ""), rec.get("caption", ""), rec.get("caption_source", "")),
            )
            self.conn.commit()
            if cur.lastrowid:
                self.conn.execute("INSERT INTO images_fts(rowid, ocr_text, caption) VALUES (?,?,?)",
                                  (cur.lastrowid, rec.get("ocr_text", ""), rec.get("caption", "")))
                self.conn.commit()
                return int(cur.lastrowid)
        row = self.conn.execute("SELECT id FROM images WHERE sha256=? AND doc_id IS ? AND page=?",
                                (rec["sha256"], rec.get("doc_id"), rec.get("page", 1))).fetchone()
        return int(row[0]) if row else None

    def set_image_vector(self, image_id: int, vec, model: str) -> None:
        import numpy as _np
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO image_vectors(image_id, model, dim, vec) VALUES (?,?,?,?)",
                (image_id, model, int(len(vec)), _np.asarray(vec, dtype=_np.float32).tobytes()),
            )
            self.conn.commit()
            self._img_matrix = None

    def image_matrix(self, model: str | None = None):
        """(N, dim) matrix of image vectors + their ids, cached."""
        import numpy as _np
        with self._lock:
            if getattr(self, "_img_matrix", None) is None:
                clause, args = self._scope_sql("i.doc_id")
                sql = ("SELECT v.image_id, v.vec, v.model FROM image_vectors v "
                       "JOIN images i ON i.id = v.image_id WHERE 1=1" + clause)
                rows = self.conn.execute(sql, args).fetchall()
                if model:
                    rows = [r for r in rows if r[2] == model]
                if rows:
                    self._img_ids = _np.array([r[0] for r in rows], dtype=_np.int64)
                    self._img_matrix = _np.vstack([_np.frombuffer(r[1], dtype=_np.float32) for r in rows])
                else:
                    self._img_ids = _np.zeros(0, dtype=_np.int64)
                    self._img_matrix = _np.zeros((0, 1), dtype=_np.float32)
            return self._img_matrix, self._img_ids

    def get_images(self, ids: list[int] | None = None, doc_id: int | None = None, limit: int = 200) -> list[dict]:
        scope_clause, scope_args = self._scope_sql("i.doc_id")
        sql = """SELECT i.*, d.name AS doc_name,
                        (SELECT 1 FROM image_vectors v WHERE v.image_id=i.id) AS has_vector
                 FROM images i LEFT JOIN documents d ON d.id=i.doc_id WHERE 1=1""" + scope_clause
        args: list = list(scope_args)
        if ids:
            sql += f" AND i.id IN ({','.join('?' * len(ids))})"
            args += list(ids)
        if doc_id is not None:
            sql += " AND i.doc_id=?"
            args.append(doc_id)
        sql += " ORDER BY i.doc_id, i.page LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def images_without_vectors(self, limit: int = 500) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            """SELECT i.* FROM images i LEFT JOIN image_vectors v ON v.image_id=i.id
               WHERE v.image_id IS NULL LIMIT ?""", (limit,))]

    def image_text_search(self, match: str, limit: int = 20) -> list[int]:
        try:
            ids = [r[0] for r in self.conn.execute(
                "SELECT rowid FROM images_fts WHERE images_fts MATCH ? ORDER BY rank LIMIT ?", (match, limit * 3))]
            return self._keep_scoped_images(ids)[:limit]
        except sqlite3.OperationalError:
            return []

    def _keep_scoped_images(self, image_ids: list[int]) -> list[int]:
        if self.active_project is None or not image_ids:
            return image_ids
        q = ",".join("?" * len(image_ids))
        clause, args = self._scope_sql("doc_id")
        rows = self.conn.execute(f"SELECT id FROM images WHERE id IN ({q}){clause}", [*image_ids, *args])
        keep = {r[0] for r in rows}
        return [i for i in image_ids if i in keep]

    def images_by_phash(self, phash: str, max_distance: int = 12, limit: int = 20) -> list[tuple[int, int]]:
        """Hamming distance on the perceptual hash - works with no vision model at all."""
        out = []
        clause, sargs = self._scope_sql("doc_id")
        for r in self.conn.execute(f"SELECT id, phash FROM images WHERE phash != ''{clause}", sargs):
            try:
                d = bin(int(r[1], 16) ^ int(phash, 16)).count("1")
            except ValueError:
                continue
            if d <= max_distance:
                out.append((int(r[0]), d))
        out.sort(key=lambda t: t[1])
        return out[:limit]

    def image_counts(self) -> dict:
        clause, args = self._scope_sql("doc_id")
        n = self.conn.execute(f"SELECT COUNT(*) FROM images WHERE 1=1{clause}", args).fetchone()[0]
        iclause, iargs = self._scope_sql("i.doc_id")
        v = self.conn.execute("SELECT COUNT(*) FROM image_vectors v JOIN images i ON i.id=v.image_id "
                              "WHERE 1=1" + iclause, iargs).fetchone()[0]
        return {"images": n, "with_vectors": v}

    def set_image_understanding(self, image_id: int, objects: list[str], description: str,
                                source: str = "", concept_score: float = 0.0) -> None:
        """What the vision model found in one picture: its concepts and a readable description."""
        import json as _json

        with self._lock:
            self.conn.execute(
                "UPDATE images SET objects=?, caption=?, caption_source=?, concept_score=? WHERE id=?",
                (_json.dumps(objects), description, source or "vision_model",
                 float(concept_score or 0.0), image_id))
            row = self.conn.execute("SELECT ocr_text FROM images WHERE id=?", (image_id,)).fetchone()
            ocr_text = (row[0] if row else "") or ""
            # the search index carries the words, so a concept is findable by name too
            try:
                self.conn.execute("DELETE FROM images_fts WHERE rowid=?", (image_id,))
            except sqlite3.OperationalError:
                pass
            self.conn.execute("INSERT INTO images_fts(rowid, ocr_text, caption) VALUES (?,?,?)",
                              (image_id, ocr_text, " ".join(objects) + " " + description))
            self.conn.commit()

    def delete_images_for_doc(self, doc_id: int) -> None:
        with self._lock:
            rows = [(r["id"], r["ocr_text"] or "", r["caption"] or "") for r in self.conn.execute(
                "SELECT id, ocr_text, caption FROM images WHERE doc_id=?", (doc_id,))]
            for image_id, ocr_text, caption in rows:
                # images_fts is a contentless FTS5 table, where a plain DELETE is rejected
                # ("cannot DELETE from contentless fts5 table"). Such a table is cleaned with
                # the 'delete' command, which needs the exact values that were indexed.
                try:
                    self.conn.execute("DELETE FROM images_fts WHERE rowid=?", (image_id,))
                except sqlite3.OperationalError:
                    try:
                        self.conn.execute(
                            "INSERT INTO images_fts(images_fts, rowid, ocr_text, caption) "
                            "VALUES ('delete', ?, ?, ?)", (image_id, ocr_text, caption))
                    except sqlite3.OperationalError as exc:
                        # A stale search-index row is harmless: results are joined back to the
                        # images table, so a deleted image can never surface. Never fail indexing.
                        logging.getLogger("ragly.store").debug(
                            "image %s left in the search index: %s", image_id, exc)
            self.conn.execute("DELETE FROM images WHERE doc_id=?", (doc_id,))
            self.conn.commit()
            self._img_matrix = None

    # ---------------- entity graph ----------------
    def clean_graph_nodes(self) -> int:
        """Repair entity names in a database indexed by an older build.

        Sentence words used to be swept into capitalised names ("No Axis Bank",
        "Profile Northstar Edge Analytics Pvt. Ltd."). Those names are trimmed here and
        merged into the real entity, mentions and edges included, so an existing database
        does not have to be re-indexed to get a readable graph. Returns how many were fixed.
        """
        from .graph import _is_junk, _norm_for_kind, _strip_determiners

        fixed = 0
        with self._lock:
            rows = [dict(r) for r in self.conn.execute("SELECT id, kind, name, norm, mentions FROM graph_nodes")]
            for row in rows:
                if row["kind"] not in ("org", "person", "place", "project", "contract"):
                    continue
                clean, _ = _strip_determiners(row["name"])
                clean = clean.strip().strip(",;:")
                if clean == row["name"] and not _is_junk(row["kind"], row["name"], row["norm"]):
                    continue
                if not clean or _is_junk(row["kind"], clean, _norm_for_kind(row["kind"], clean)):
                    self.conn.execute("DELETE FROM node_mentions WHERE node_id=?", (row["id"],))
                    self.conn.execute("DELETE FROM graph_edges WHERE src=? OR dst=?", (row["id"], row["id"]))
                    self.conn.execute("DELETE FROM graph_nodes WHERE id=?", (row["id"],))
                    fixed += 1
                    continue
                norm = _norm_for_kind(row["kind"], clean)
                existing = self.conn.execute("SELECT id, mentions FROM graph_nodes WHERE kind=? AND norm=? AND id<>?",
                                             (row["kind"], norm, row["id"])).fetchone()
                if existing:                      # merge into the entity that was already right
                    keep = int(existing[0])
                    self.conn.execute("UPDATE node_mentions SET node_id=? WHERE node_id=?", (keep, row["id"]))
                    self.conn.execute("UPDATE graph_edges SET src=? WHERE src=?", (keep, row["id"]))
                    self.conn.execute("UPDATE graph_edges SET dst=? WHERE dst=?", (keep, row["id"]))
                    self.conn.execute("UPDATE graph_nodes SET mentions=mentions+? WHERE id=?",
                                      (row["mentions"] or 0, keep))
                    self.conn.execute("DELETE FROM graph_nodes WHERE id=?", (row["id"],))
                else:
                    self.conn.execute("UPDATE graph_nodes SET name=?, norm=? WHERE id=?", (clean, norm, row["id"]))
                fixed += 1
            if fixed:
                self.conn.execute("DELETE FROM graph_edges WHERE src=dst")
                self.conn.commit()
        return fixed


    def upsert_node(self, kind: str, name: str, norm: str) -> int:
        with self._lock:
            self.conn.execute(
                "INSERT INTO graph_nodes(kind, name, norm, mentions) VALUES (?,?,?,0) "
                "ON CONFLICT(kind, norm) DO UPDATE SET mentions = mentions + 1",
                (kind, name, norm),
            )
            self.conn.commit()
        row = self.conn.execute("SELECT id FROM graph_nodes WHERE kind=? AND norm=?", (kind, norm)).fetchone()
        return int(row[0])

    def add_edge(self, src: int, rel: str, dst: int, doc_id: int | None = None,
                 page: int | None = None, chunk_id: int | None = None, weight: float = 1.0) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO graph_edges(src, rel, dst, doc_id, page, chunk_id, weight)
                   VALUES (?,?,?,?,?,?,?)""", (src, rel, dst, doc_id, page, chunk_id, weight))
            self.conn.commit()

    def add_mention(self, node_id: int, doc_id: int, page: int,
                    chunk_id: int | None = None, image_id: int | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO node_mentions(node_id, doc_id, page, chunk_id, image_id)
                   VALUES (?,?,?,?,?)""", (node_id, doc_id, page, chunk_id, image_id))
            self.conn.commit()

    def find_nodes(self, text: str, limit: int = 10) -> list[dict]:
        pattern = f"%{re_escape_like(text)}%"
        clause, sargs = self._scope_sql("m.doc_id")
        exists = (" AND EXISTS (SELECT 1 FROM node_mentions m WHERE m.node_id = n.id" + clause + ")"
                  if self.active_project is not None else "")
        return [dict(r) for r in self.conn.execute(
            f"""SELECT n.* FROM graph_nodes n
                WHERE (n.norm LIKE ? ESCAPE '\\' OR n.name LIKE ? ESCAPE '\\'){exists}
                ORDER BY n.mentions DESC LIMIT ?""", [pattern, pattern, *sargs, limit])]

    def node_neighbourhood(self, node_id: int, limit: int = 60) -> dict:
        edges = [dict(r) for r in self.conn.execute(
            """SELECT e.*, s.name AS src_name, s.kind AS src_kind, t.name AS dst_name, t.kind AS dst_kind
               FROM graph_edges e JOIN graph_nodes s ON s.id=e.src JOIN graph_nodes t ON t.id=e.dst
               WHERE e.src=? OR e.dst=? LIMIT ?""", (node_id, node_id, limit))]
        mentions = [dict(r) for r in self.conn.execute(
            """SELECT m.*, d.name AS doc_name FROM node_mentions m JOIN documents d ON d.id=m.doc_id
               WHERE m.node_id=? LIMIT ?""", (node_id, limit))]
        node = self.conn.execute("SELECT * FROM graph_nodes WHERE id=?", (node_id,)).fetchone()
        return {"node": dict(node) if node else None, "edges": edges, "mentions": mentions}

    def top_nodes(self, kind: str | None = None, limit: int = 40) -> list[dict]:
        clause, sargs = self._scope_sql("m.doc_id")
        sql = ("SELECT n.* FROM graph_nodes n WHERE EXISTS "
               "(SELECT 1 FROM node_mentions m WHERE m.node_id = n.id" + clause + ")")
        args: list = list(sargs)
        if kind:
            sql += " AND n.kind=?"
            args.append(kind)
        else:
            # Default view: the things worth browsing. Every amount and every date in a
            # corpus is an entity, but a list of 40 rupee figures tells the reader nothing —
            # those kinds stay one click away behind their own filter.
            sql += " AND n.kind IN ('org','person','project','place','invoice','contract','code','part_number')"
        sql += " ORDER BY n.mentions DESC, n.name LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def graph_counts(self) -> dict:
        mclause, margs = self._scope_sql("m.doc_id")
        eclause, eargs = self._scope_sql("e.doc_id")
        nodes = self.conn.execute(
            "SELECT COUNT(DISTINCT m.node_id) FROM node_mentions m WHERE 1=1" + mclause, margs).fetchone()[0]
        edges = self.conn.execute(
            "SELECT COUNT(*) FROM graph_edges e WHERE 1=1" + eclause, eargs).fetchone()[0]
        mentions = self.conn.execute(
            "SELECT COUNT(*) FROM node_mentions m WHERE 1=1" + mclause, margs).fetchone()[0]
        return {"nodes": nodes, "edges": edges, "mentions": mentions}

    def clear_graph_for_doc(self, doc_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM node_mentions WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM graph_edges WHERE doc_id=?", (doc_id,))
            self.conn.commit()

    # ---------------- caches / benchmarks ----------------
    def cache_get(self, key: str):
        import numpy as _np
        row = self.conn.execute("SELECT vec FROM embed_cache WHERE key=?", (key,)).fetchone()
        return _np.frombuffer(row[0], dtype=_np.float32) if row else None

    def cache_put(self, key: str, vec) -> None:
        import numpy as _np
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO embed_cache(key, vec, used_at) VALUES (?,?,?)",
                              (key, _np.asarray(vec, dtype=_np.float32).tobytes(), time.time()))
            self.conn.commit()

    def cache_size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]

    def record_benchmark(self, task: str, model: str, backend: str, latency_ms: float | None = None,
                         throughput: float | None = None, memory_mb: float | None = None,
                         extra: dict | None = None) -> None:
        import json as _json
        with self._lock:
            self.conn.execute(
                """INSERT INTO benchmarks(ran_at, task, model, backend, latency_ms, throughput, memory_mb, extra)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (time.time(), task, model, backend, latency_ms, throughput, memory_mb, _json.dumps(extra or {})))
            self.conn.commit()

    def benchmarks(self, limit: int = 200) -> list[dict]:
        import json as _json
        rows = self.conn.execute("SELECT * FROM benchmarks ORDER BY ran_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["extra"] = _json.loads(d["extra"])
            out.append(d)
        return out


def re_escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# attach the multimodal methods to Store (kept separate above for readability)
for _name, _fn in list(vars(MultimodalMixin).items()):
    if callable(_fn) and not _name.startswith("__"):
        setattr(Store, _name, _fn)
