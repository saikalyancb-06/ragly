"""Download the offline models into ./models (run once, with internet).

    python scripts/download_models.py              # answer model + fp32 embeddings (recommended)
    python scripts/download_models.py --llm 1.5b   # smaller/faster answer model
    python scripts/download_models.py --int8       # + int8 embeddings (used when an NPU is present)
    python scripts/download_models.py --clip       # + CLIP image search (int8, ~155 MB)
    python scripts/download_models.py --clip-fp32  # + CLIP at full precision (~608 MB)
    python scripts/download_models.py --all        # everything

This is the ONLY place that touches the network. At runtime the offline guard blocks
every non-loopback connection, so nothing is ever downloaded while the app is running.
"""
import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LLMS = {
    "3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}
EMBED = ("BAAI/bge-small-en-v1.5", ["onnx/model.onnx", "tokenizer.json"], "bge-small-en-v1.5")
# (repo, [(repo_file, saved_as)], folder) - saved names are what the app expects
EMBED_INT8 = ("Xenova/bge-small-en-v1.5",
              [("onnx/model_int8.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")],
              "bge-small-en-v1.5-int8")
CLIP_INT8 = ("Xenova/clip-vit-base-patch32",
             [("onnx/vision_model_int8.onnx", "visual.onnx"),
              ("onnx/text_model_int8.onnx", "textual.onnx"),
              ("tokenizer.json", "tokenizer.json")],
             "clip-vit-base-patch32")
CLIP_FP32 = ("Xenova/clip-vit-base-patch32",
             [("onnx/vision_model.onnx", "visual.onnx"),
              ("onnx/text_model.onnx", "textual.onnx"),
              ("tokenizer.json", "tokenizer.json")],
             "clip-vit-base-patch32")


def fetch(repo: str, filename: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok (already present) {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo}/{filename} ...", flush=True)
    path = hf_hub_download(repo, filename, cache_dir=str(MODELS / ".cache"))
    shutil.copyfile(path, dest)
    print(f"  saved {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=sorted(LLMS), default="3b")
    ap.add_argument("--int8", action="store_true", help="int8 text embeddings (NPU deployment)")
    ap.add_argument("--clip", action="store_true", help="CLIP image search, int8 (~155 MB)")
    ap.add_argument("--clip-fp32", action="store_true", help="CLIP image search, float32 (~608 MB)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.int8 = args.clip = True

    print("Embedding model:")
    repo, files, folder = EMBED
    for f in files:
        fetch(repo, f, MODELS / folder / Path(f).name)

    if args.int8:
        print("Embedding model (int8, for the Hexagon NPU):")
        repo, pairs, folder = EMBED_INT8
        for src, dst in pairs:
            fetch(repo, src, MODELS / folder / dst)

    if args.clip or args.clip_fp32:
        repo, pairs, folder = CLIP_FP32 if args.clip_fp32 else CLIP_INT8
        print(f"Vision model for image search ({'float32' if args.clip_fp32 else 'int8'}):")
        for src, dst in pairs:
            fetch(repo, src, MODELS / folder / dst)

    if args.skip_llm:
        print("Skipping the answer model as requested.")
    else:
        print("Answer model (GGUF for llama.cpp):")
        repo, f = LLMS[args.llm]
        fetch(repo, f, MODELS / f)

    if args.llm != "3b" and not args.skip_llm:
        import json

        eng = ROOT / "engines.json"
        cfg = json.loads(eng.read_text())
        cfg["normal"]["gguf"] = f"models/{f}"
        eng.write_text(json.dumps(cfg, indent=2))
        print(f"  engines.json now points to models/{f}")

    if not args.keep_cache:
        shutil.rmtree(MODELS / ".cache", ignore_errors=True)
    print("Done. Models are stored locally; the app never downloads anything at runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
