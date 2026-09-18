"""On-device text embeddings with ONNX Runtime (bge-small-en-v1.5, 384-dim, CLS pooling)."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import settings

QNN_OPTIONS = {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"}


class Embedder:
    def __init__(self, model_dir: Path | None = None, providers: list[str] | None = None,
                 model_id: str | None = None):
        self.model_dir = Path(model_dir or settings.embed_model_dir)
        model_path = self.model_dir / "model.onnx"
        tok_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"Embedding model missing in {self.model_dir}. Run scripts/download_models.py first."
            )
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=settings.embed_max_len)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._counter = Tokenizer.from_file(str(tok_path))  # no padding/truncation, for counting
        self._counter.no_truncation()
        self._counter.no_padding()
        self._lock = threading.Lock()
        self.session, self.provider = self._make_session(model_path, providers or ["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.model_id = model_id or self.model_dir.name or settings.embed_model_id
        self.quantization = "int8" if "int8" in self.model_dir.name else "float32"

    @staticmethod
    def _make_session(model_path: Path, wanted: list[str]):
        available = set(ort.get_available_providers())
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for prov in wanted:
            if prov not in available:
                continue
            try:
                if prov == "QNNExecutionProvider":
                    s = ort.InferenceSession(str(model_path), opts, providers=[(prov, QNN_OPTIONS), "CPUExecutionProvider"])
                else:
                    s = ort.InferenceSession(str(model_path), opts, providers=[prov])
                return s, s.get_providers()[0]
            except Exception:  # provider present but failed to init -> try next
                continue
        s = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        return s, "CPUExecutionProvider"

    def count_tokens(self, text: str) -> int:
        return len(self._counter.encode(text, add_special_tokens=False).ids)

    def token_windows(self, text: str, size: int, overlap: int) -> list[str]:
        """Split an over-long piece of text into token windows (used for giant sentences)."""
        enc = self._counter.encode(text, add_special_tokens=False)
        offsets = enc.offsets
        out, step = [], max(1, size - overlap)
        for start in range(0, len(offsets), step):
            window = offsets[start : start + size]
            if not window:
                break
            out.append(text[window[0][0] : window[-1][1]])
            if start + size >= len(offsets):
                break
        return out

    def _run(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        with self._lock:
            out = self.session.run(None, feeds)[0]
        vecs = out[:, 0, :] if out.ndim == 3 else out  # CLS pooling
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [self._run(texts[i : i + settings.embed_batch]) for i in range(0, len(texts), settings.embed_batch)]
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._run([settings.query_instruction + text])[0]
