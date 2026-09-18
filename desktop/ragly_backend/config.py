"""Central configuration. Everything is local; values can be overridden with env vars."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("RAGLY_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("RAGLY_DATA", ROOT / "data"))
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "ragly.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
ENGINES_PATH = Path(os.environ.get("RAGLY_ENGINES", ROOT / "engines.json"))
MODELS_DIR = ROOT / "models"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    host: str = os.environ.get("RAGLY_HOST", "127.0.0.1")
    port: int = _env_int("RAGLY_PORT", 8765)

    # Embeddings
    embed_model_dir: Path = Path(os.environ.get("RAGLY_EMBED_DIR", MODELS_DIR / "bge-small-en-v1.5"))
    embed_model_id: str = "bge-small-en-v1.5"
    embed_max_len: int = 512
    embed_batch: int = 16
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Chunking (tokens of the embedding tokenizer)
    chunk_tokens: int = _env_int("RAGLY_CHUNK_TOKENS", 380)
    chunk_overlap: int = _env_int("RAGLY_CHUNK_OVERLAP", 60)

    # OCR: pages with fewer characters than this are OCR'd
    ocr_min_chars: int = 30
    ocr_dpi: int = 200

    # Retrieval
    top_k: int = _env_int("RAGLY_TOP_K", 5)
    candidates: int = 20
    pool_factor: int = _env_int("RAGLY_POOL_FACTOR", 6)   # candidate pool = top_k * this, then reranked
    rrf_k: int = 60
    min_score: float = _env_float("RAGLY_MIN_SCORE", 0.45)  # cosine; below => "not found"
    keyword_bonus: float = 0.05  # threshold is relaxed by this much when keyword search also matched

    # Generation
    strict_grounding: bool = os.environ.get("RAGLY_STRICT", "1") != "0"
    require_sufficient_evidence: bool = os.environ.get("RAGLY_SUFFICIENCY", "1") != "0"
    temperature: float = _env_float("RAGLY_TEMPERATURE", 0.0)
    max_tokens: int = _env_int("RAGLY_MAX_TOKENS", 256)
    # How much text the model actually reads. Prompt reading is what a CPU spends its time on:
    # 3 passages of ~900 characters answer the same questions as 5 full ones, several times faster.
    prompt_passages: int = _env_int("RAGLY_PROMPT_PASSAGES", 3)
    prompt_chars: int = _env_int("RAGLY_PROMPT_CHARS", 900)
    request_timeout: float = 300.0

    # Offline guard: block every non-loopback socket connection in this process
    offline_guard: bool = os.environ.get("RAGLY_OFFLINE_GUARD", "1") != "0"

    extra: dict = field(default_factory=dict)


settings = Settings()

NOT_FOUND = "Not found in your documents."


def ensure_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)


def load_engines() -> dict:
    with open(ENGINES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_user_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_user_settings(values: dict) -> None:
    ensure_dirs()
    current = load_user_settings()
    current.update(values)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
