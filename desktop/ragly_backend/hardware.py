"""Hardware abstraction: detect what this machine really has, and pick a backend honestly.

Priority when RAGLY_BACKEND=auto (the default):
    1. Qualcomm NPU  (QNNExecutionProvider, Hexagon) - only when ONNX Runtime reports it
    2. GPU           (DirectML on Windows, CUDA if present, ROCm)
    3. CPU           (always available, always functional)

Nothing here fabricates capability: every backend claim comes from
onnxruntime.get_available_providers(), a successful session creation, or the
provider a live session reports. On an Intel i5 with no accelerator this module
says exactly that, and the application runs on CPU.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import onnxruntime as ort

BACKEND_MODES = ("auto", "cpu", "gpu", "npu")
NPU_PROVIDERS = ("QNNExecutionProvider",)
GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                 "OpenVINOExecutionProvider", "CoreMLExecutionProvider")
CPU_PROVIDER = "CPUExecutionProvider"

# QNN on Windows on Snapdragon: the HTP (Hexagon Tensor Processor) backend library
QNN_OPTIONS = {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst",
               "htp_graph_finalization_optimization_mode": "3"}


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            txt = Path("/proc/cpuinfo").read_text(errors="ignore")
            m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M) or re.search(r"^Hardware\s*:\s*(.+)$", txt, re.M)
            if m:
                name = m.group(1).strip()
            if "0x51" in txt and "qualcomm" not in name.lower():
                name = f"Qualcomm ARM ({name})"
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=3).stdout.strip() or name
        except Exception:
            pass
    return name.strip()


def total_ram_mb() -> float:
    try:
        if sys.platform == "win32":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1e6, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e6, 1)
    except Exception:
        return 0.0


def process_memory_mb() -> float:
    """Resident memory of this process - used for honest memory numbers in benchmarks."""
    try:
        if sys.platform == "win32":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = Counters()
            c.cb = ctypes.sizeof(Counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                     ctypes.byref(c), c.cb)
            return round(c.WorkingSetSize / 1e6, 1)
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return 0.0


def gpu_names() -> list[str]:
    """Best-effort GPU listing. Empty list simply means 'none detected'."""
    out: list[str] = []
    try:
        if sys.platform == "win32":
            ps = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=8)
            out = [ln.strip() for ln in ps.stdout.splitlines() if ln.strip()]
        elif sys.platform.startswith("linux"):
            for p in Path("/sys/class/drm").glob("card*/device/uevent"):
                txt = p.read_text(errors="ignore")
                m = re.search(r"DRIVER=(\w+)", txt)
                if m:
                    out.append(m.group(1))
            out = sorted(set(out))
    except Exception:
        pass
    return out


@dataclass
class HardwareInfo:
    os: str
    machine: str
    cpu: str
    cores: int
    ram_mb: float
    is_snapdragon: bool
    snapdragon_forced: bool
    gpus: list[str] = field(default_factory=list)
    ort_version: str = ""
    available_providers: list[str] = field(default_factory=list)
    npu_providers: list[str] = field(default_factory=list)
    gpu_providers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def npu_available(self) -> bool:
        return bool(self.npu_providers)

    @property
    def gpu_available(self) -> bool:
        return bool(self.gpu_providers)

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "npu_available": self.npu_available,
            "gpu_available": self.gpu_available,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        if self.is_snapdragon and self.npu_available:
            return f"Qualcomm Snapdragon detected · NPU available ({', '.join(self.npu_providers)})"
        if self.is_snapdragon:
            return ("Qualcomm Snapdragon detected · NPU runtime not installed "
                    "(install onnxruntime-qnn for Hexagon NPU execution)")
        return f"Snapdragon/NPU: not detected · running in CPU compatibility mode ({self.cpu})"


def detect() -> HardwareInfo:
    cpu = _cpu_name()
    machine = platform.machine()
    forced = os.environ.get("RAGLY_FORCE_SNAPDRAGON") == "1"
    qcom = any(k in cpu.lower() for k in ("snapdragon", "qualcomm", "oryon", "kryo"))
    is_arm = machine.lower() in ("arm64", "aarch64")
    is_sd = forced or (qcom and (is_arm or sys.platform == "win32"))

    available = list(ort.get_available_providers())
    info = HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        machine=machine,
        cpu=cpu,
        cores=os.cpu_count() or 1,
        ram_mb=total_ram_mb(),
        is_snapdragon=is_sd,
        snapdragon_forced=forced,
        gpus=gpu_names(),
        ort_version=ort.__version__,
        available_providers=available,
        npu_providers=[p for p in NPU_PROVIDERS if p in available],
        gpu_providers=[p for p in GPU_PROVIDERS if p in available],
    )
    if is_sd and not info.npu_providers:
        info.notes.append("Snapdragon CPU detected but no QNN execution provider: "
                          "pip install onnxruntime-qnn to use the Hexagon NPU.")
    if forced:
        info.notes.append("RAGLY_FORCE_SNAPDRAGON=1 is set: Snapdragon mode is allowed for testing, "
                          "but NPU execution is still only claimed when the runtime reports it.")
    if not is_sd:
        info.notes.append("Development machine: CPU execution. The same code selects the NPU "
                          "automatically on a Snapdragon device.")
    return info


class InferenceManager:
    """Chooses an ordered provider list per model kind, and records what was actually used."""

    KINDS = ("text_embedding", "image_embedding", "ocr", "llm")

    def __init__(self, info: HardwareInfo | None = None, mode: str | None = None):
        self.info = info or detect()
        self.mode = (mode or os.environ.get("RAGLY_BACKEND") or os.environ.get("MODEL_BACKEND") or "auto").lower()
        if self.mode not in BACKEND_MODES:
            self.mode = "auto"
        self.actual: dict[str, str] = {}      # kind -> provider/runtime actually in use
        self.reasons: dict[str, str] = {}

    # ---------- provider selection ----------
    def providers(self, kind: str = "text_embedding") -> list[str]:
        """Ordered ONNX Runtime providers to try for this kind of model."""
        npu = self.info.npu_providers
        gpu = self.info.gpu_providers
        if self.mode == "cpu":
            self.reasons[kind] = "forced CPU (RAGLY_BACKEND=cpu)"
            return [CPU_PROVIDER]
        if self.mode == "npu":
            if npu:
                self.reasons[kind] = "forced NPU (RAGLY_BACKEND=npu)"
                return [*npu, CPU_PROVIDER]
            self.reasons[kind] = "NPU requested but no QNN provider installed: falling back to CPU"
            return [CPU_PROVIDER]
        if self.mode == "gpu":
            if gpu:
                self.reasons[kind] = "forced GPU (RAGLY_BACKEND=gpu)"
                return [*gpu, CPU_PROVIDER]
            self.reasons[kind] = "GPU requested but no GPU provider installed: falling back to CPU"
            return [CPU_PROVIDER]
        # auto
        order: list[str] = []
        if npu and kind in ("text_embedding", "image_embedding", "ocr"):
            order += npu
            self.reasons[kind] = "auto: Qualcomm NPU preferred"
        if gpu:
            order += gpu
            self.reasons.setdefault(kind, "auto: GPU preferred (no NPU provider)")
        order.append(CPU_PROVIDER)
        self.reasons.setdefault(kind, "auto: CPU (no accelerator detected)")
        return order

    def qnn_options(self) -> dict:
        return dict(QNN_OPTIONS)

    def record(self, kind: str, provider_or_runtime: str) -> None:
        """Called by model wrappers with the provider the live session reports."""
        self.actual[kind] = provider_or_runtime

    def backend_label(self, kind: str) -> str:
        p = self.actual.get(kind, "")
        if p.startswith("QNN"):
            return "NPU"
        if p in GPU_PROVIDERS:
            return "GPU"
        if p == CPU_PROVIDER:
            return "CPU"
        return p or "not loaded"

    # ---------- honest reporting ----------
    def report(self, extra: dict | None = None) -> dict:
        return {
            "mode": self.mode,
            "device": self.info.to_dict(),
            "selected": {k: self.providers(k)[0] for k in self.KINDS},
            "actual": {k: {"provider": v, "backend": self.backend_label(k)} for k, v in self.actual.items()},
            "reasons": dict(self.reasons),
            "npu_claimed": any(v.startswith("QNN") for v in self.actual.values()),
            "cloud_api_calls": 0,
            **(extra or {}),
        }


@dataclass
class BenchResult:
    task: str
    model: str
    backend: str
    latency_ms: float
    throughput: float | None
    memory_mb: float
    iterations: int

    def to_dict(self) -> dict:
        return self.__dict__


def bench(task: str, model: str, backend: str, fn: Callable[[], object],
          iterations: int = 5, warmup: int = 1, units: float | None = None) -> BenchResult:
    """Time a real callable. `units` (e.g. tokens or images per call) gives throughput."""
    for _ in range(max(0, warmup)):
        fn()
    before = process_memory_mb()
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / max(1, iterations) * 1000
    throughput = (units * iterations / elapsed) if units else None
    return BenchResult(task=task, model=model, backend=backend, latency_ms=round(per_call_ms, 2),
                       throughput=round(throughput, 2) if throughput else None,
                       memory_mb=round(max(process_memory_mb(), before), 1), iterations=iterations)
