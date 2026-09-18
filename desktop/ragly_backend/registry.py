"""Model registry: what each model is, where it runs, and whether it is actually installed.

The registry is the single place that knows about model files, so a Qualcomm AI Hub
optimised model can replace a generic ONNX one without touching application code.
Every entry reports installed / missing from the filesystem - never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import MODELS_DIR, resolve_path


@dataclass
class ModelEntry:
    key: str
    role: str                  # text_generation | text_embedding | image_embedding | ocr
    name: str
    files: list[str]           # relative to models/
    format: str                # GGUF | ONNX | QAIRT bundle | OS built-in
    quantization: str
    dim: int | None = None
    context: int | None = None
    memory_mb: int | None = None
    backends: list[str] = field(default_factory=lambda: ["CPU"])
    snapdragon: str = "cpu"    # npu | gpu | cpu | unknown
    notes: str = ""
    source: str = ""

    def status(self) -> dict:
        present, missing, size = True, [], 0
        for rel in self.files:
            p = resolve_path(MODELS_DIR / rel) if not Path(rel).is_absolute() else Path(rel)
            if p.exists():
                size += p.stat().st_size
            else:
                present = False
                missing.append(rel)
        return {
            "key": self.key, "role": self.role, "name": self.name, "format": self.format,
            "quantization": self.quantization, "dim": self.dim, "context": self.context,
            "backends": self.backends, "snapdragon": self.snapdragon, "notes": self.notes,
            "source": self.source, "installed": present if self.files else True,
            "missing_files": missing, "size_mb": round(size / 1e6, 1) if size else None,
            "expected_memory_mb": self.memory_mb,
        }


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="qwen2.5-3b-instruct-q4km", role="text_generation", name="Qwen2.5-3B-Instruct",
        files=["qwen2.5-3b-instruct-q4_k_m.gguf"], format="GGUF", quantization="Q4_K_M",
        context=4096, memory_mb=2600, backends=["CPU", "GPU (llama.cpp)", "NPU (via GenieX qairt bundle)"],
        snapdragon="npu", source="Qwen/Qwen2.5-3B-Instruct-GGUF",
        notes="Runs through llama.cpp on CPU. On Snapdragon, GenieX runs an AI Hub bundle on the Hexagon NPU; "
              "the GGUF path there uses CPU/GPU.",
    ),
    ModelEntry(
        key="geniex-qairt-llm", role="text_generation", name="AI Hub LLM bundle (GenieX qairt)",
        files=[], format="QAIRT bundle", quantization="int4/int8 (baked in)", context=4096,
        backends=["NPU"], snapdragon="npu", source="Qualcomm AI Hub via `geniex pull`",
        notes="Pulled by the GenieX CLI into its own store, not into models/. Only available on Snapdragon.",
    ),
    ModelEntry(
        key="bge-small-en-v1.5-fp32", role="text_embedding", name="BGE-small-en-v1.5 (fp32)",
        files=["bge-small-en-v1.5/model.onnx", "bge-small-en-v1.5/tokenizer.json"],
        format="ONNX", quantization="float32", dim=384, memory_mb=140, backends=["CPU", "GPU", "NPU (partial)"],
        snapdragon="gpu", source="BAAI/bge-small-en-v1.5",
        notes="Reference accuracy model. The Hexagon NPU prefers quantised graphs, so fp32 may fall back to CPU.",
    ),
    ModelEntry(
        key="bge-small-en-v1.5-int8", role="text_embedding", name="BGE-small-en-v1.5 (int8)",
        files=["bge-small-en-v1.5-int8/model.onnx", "bge-small-en-v1.5-int8/tokenizer.json"],
        format="ONNX", quantization="int8 (dynamic)", dim=384, memory_mb=45,
        backends=["CPU", "NPU"], snapdragon="npu", source="Xenova/bge-small-en-v1.5 (model_int8.onnx)",
        notes="Used automatically when a QNN provider is present; keeps the same 384-dim space as fp32 "
              "but vectors are not bit-identical, so switching triggers a re-index.",
    ),
    ModelEntry(
        key="clip-vit-b32", role="image_embedding", name="CLIP ViT-B/32 (visual + textual)",
        files=["clip-vit-base-patch32/visual.onnx", "clip-vit-base-patch32/textual.onnx",
               "clip-vit-base-patch32/tokenizer.json"],
        format="ONNX", quantization="int8 or float32", dim=512, memory_mb=170,
        backends=["CPU", "GPU", "NPU (int8)"], snapdragon="npu", source="Xenova/clip-vit-base-patch32",
        notes="Enables text→image and image→image search. Without it, image search falls back to OCR text "
              "plus perceptual hashing, and the UI says so.",
    ),
    ModelEntry(
        key="windows-ocr", role="ocr", name="Windows OCR (Windows.Media.OCR)",
        files=[], format="OS built-in", quantization="n/a", backends=["CPU (OS managed)"],
        snapdragon="cpu", source="Windows language pack",
        notes="Offline OS engine, works on x64 and ARM64. Qualcomm AI Hub OCR models (EasyOCR/TrOCR) can "
              "replace it later for NPU execution.",
    ),
    ModelEntry(
        key="rapidocr", role="ocr", name="RapidOCR (ONNX)", files=[], format="ONNX", quantization="int8",
        backends=["CPU"], snapdragon="cpu", source="rapidocr-onnxruntime (pip, models bundled)",
        notes="Used on Linux/macOS development machines; Python <= 3.12 only.",
    ),
]


def registry_status() -> list[dict]:
    return [m.status() for m in REGISTRY]


def get(key: str) -> ModelEntry | None:
    return next((m for m in REGISTRY if m.key == key), None)


def embedding_choice(npu_available: bool) -> tuple[str, str]:
    """Which text-embedding model to load: (models-dir subpath, reason). int8 only when an NPU is present."""
    int8 = get("bge-small-en-v1.5-int8")
    fp32 = get("bge-small-en-v1.5-fp32")
    if npu_available and int8 and int8.status()["installed"]:
        return "bge-small-en-v1.5-int8", "int8 model chosen: a QNN/NPU provider is available"
    if npu_available and int8:
        return "bge-small-en-v1.5", ("NPU present but the int8 model is not installed "
                                     "(python scripts/download_models.py --int8): using fp32")
    return "bge-small-en-v1.5", "fp32 model (reference accuracy) - no NPU provider detected"
