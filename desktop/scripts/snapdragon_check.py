"""What this machine can actually run, and what is missing for the Hexagon NPU.

Reports only what it can verify: the execution providers ONNX Runtime really offers, the
models present on disk, and whether the GenieX CLI is installed. Nothing here claims NPU
execution - `ragly_backend.hardware` refuses to do that unless a QNN provider is active.

    python scripts/snapdragon_check.py            # human readable
    python scripts/snapdragon_check.py --json     # for scripts
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def report() -> dict:
    out: dict = {"machine": platform.machine(), "platform": platform.platform(),
                 "python": platform.python_version()}
    try:
        import onnxruntime as ort

        out["onnxruntime"] = ort.__version__
        out["providers"] = list(ort.get_available_providers())
    except Exception as exc:
        out["onnxruntime"] = None
        out["providers"] = []
        out["onnxruntime_error"] = str(exc)

    out["qnn_available"] = any(p.startswith("QNN") for p in out["providers"])
    out["is_arm64"] = out["machine"].lower() in ("arm64", "aarch64")

    try:
        from ragly_backend.hardware import detect

        hw = detect()
        out["cpu"] = hw.cpu
        out["is_snapdragon"] = hw.is_snapdragon
        out["npu_available"] = hw.npu_available
        out["summary"] = hw.summary()
    except Exception as exc:
        out["hardware_error"] = str(exc)

    models = ROOT / "models"
    out["models"] = {
        "answer_model": (models / "qwen2.5-3b-instruct-q4_k_m.gguf").exists(),
        "text_embeddings": (models / "bge-small-en-v1.5" / "model.onnx").exists(),
        "vision_model": (models / "clip-vit-base-patch32" / "visual.onnx").exists(),
    }
    out["llama_server"] = any((ROOT / "bin" / "llama" / name).exists()
                              for name in ("llama-server.exe", "llama-server"))
    out["geniex"] = bool(shutil.which("geniex"))
    return out


def main() -> int:
    data = report()
    if "--json" in sys.argv:
        print(json.dumps(data, indent=2))
        return 0

    print(f"machine          : {data['machine']}  ({data.get('cpu', 'unknown CPU')})")
    print(f"onnxruntime      : {data['onnxruntime'] or 'NOT INSTALLED'}")
    print(f"providers        : {', '.join(data['providers']) or 'none'}")
    print(f"answer engine    : {'llama.cpp present' if data['llama_server'] else 'MISSING'}"
          f" · GenieX {'installed' if data['geniex'] else 'not installed'}")
    for name, present in data["models"].items():
        print(f"{name:<17}: {'present' if present else 'MISSING'}")
    print()

    if data["qnn_available"]:
        print("Hexagon NPU      : available to ONNX Runtime (embeddings and image search can use it)")
    elif data.get("is_snapdragon"):
        print("Hexagon NPU      : NOT available — install it with:")
        print("                   .venv\\Scripts\\python.exe -m pip install onnxruntime-qnn")
        print("                   Embeddings run on the CPU until then; everything still works.")
    else:
        print("Hexagon NPU      : not applicable on this machine (no Snapdragon CPU detected)")

    if data.get("is_snapdragon") and not data["geniex"]:
        print("GenieX           : not installed — answers run on the CPU through llama.cpp.")
        print("                   scripts\\setup_windows.ps1 downloads the installer on Snapdragon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
