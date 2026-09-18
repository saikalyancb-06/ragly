"""Engine manager: the Normal (llama.cpp) <-> Snapdragon (GenieX) toggle.

Both engines expose the same OpenAI-style API, so switching = stop one local server,
start the other, wait for /v1/models, and point the client at the new address.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import DATA_DIR, ROOT, load_engines, load_user_settings, resolve_path, save_user_settings
from .llm import LLMClient

log = logging.getLogger("ragly.engine")
MODES = ("normal", "snapdragon")


def _cpu_name() -> str:
    name = platform.processor() or ""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            txt = Path("/proc/cpuinfo").read_text(errors="ignore")
            for line in txt.splitlines():
                if line.lower().startswith(("model name", "hardware", "cpu implementer")):
                    name = name or line.split(":", 1)[-1].strip()
            if "0x51" in txt:  # Qualcomm ARM implementer id
                name = f"Qualcomm ({name})"
        except OSError:
            pass
    return name.strip()


def device_info() -> dict:
    machine = platform.machine()
    cpu = _cpu_name()
    forced = os.environ.get("RAGLY_FORCE_SNAPDRAGON") == "1"
    is_arm = machine.lower() in ("arm64", "aarch64")
    qcom = any(k in cpu.lower() for k in ("snapdragon", "qualcomm"))
    # On Windows the CPU name is authoritative (x64 Python under emulation still reports AMD64)
    is_sd = forced or (qcom and (is_arm or sys.platform == "win32"))
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": machine,
        "cpu": cpu,
        "is_snapdragon": is_sd,
        "snapdragon_forced": forced,
        "python": platform.python_version(),
    }


class EngineManager:
    def __init__(self):
        self.cfg = load_engines()
        self.device = device_info()
        saved = load_user_settings().get("engine_mode")
        self.mode = saved if saved in MODES else self.cfg.get("default_mode", "normal")
        if self.mode == "snapdragon" and not self.device["is_snapdragon"]:
            self.mode = "normal"
        self.proc: subprocess.Popen | None = None
        self.client: LLMClient | None = None
        self.state = "stopped"  # stopped | starting | ready | error
        self.error: str | None = None
        self._lock = threading.RLock()
        self._log_file = None

    # ---------- commands ----------
    def _command(self, mode: str) -> list[str]:
        c = self.cfg[mode]
        if mode == "snapdragon":
            cmd = list(c.get("start_command") or [])
            if not cmd:
                raise RuntimeError("engines.json: snapdragon.start_command is empty")
            if not shutil.which(cmd[0]) and not Path(cmd[0]).exists():
                raise RuntimeError(f"GenieX not found ('{cmd[0]}'). Install the GenieX CLI (scripts/setup_windows.ps1 does this on Snapdragon)")
            return cmd

        gguf = resolve_path(c["gguf"])
        if not gguf.exists():
            raise RuntimeError(f"Model file not found: {gguf}. Run scripts/download_models.py")
        port = c["base_url"].rsplit(":", 1)[-1].split("/")[0]
        exe = resolve_path(c.get("server_bin", "bin/llama/llama-server"))
        candidates = [exe, exe.with_suffix(".exe")]
        found = next((p for p in candidates if p.exists()), None) or (
            Path(shutil.which("llama-server")) if shutil.which("llama-server") else None
        )
        cpus = os.cpu_count() or 4
        # leave a core for the app on a real machine; on a 1-2 core box use everything
        threads = int(c.get("threads") or 0) or (max(1, cpus - 1) if cpus > 2 else cpus)
        if found:
            # --cache-reuse lets llama.cpp keep the part of the prompt that did not change
            # (the instructions, and any passage repeated between questions) instead of
            # reading it again, which is where a CPU spends most of its time.
            return [str(found), "-m", str(gguf), "--host", "127.0.0.1", "--port", port,
                    "-c", str(c.get("ctx", 4096)), "-t", str(threads),
                    "--cache-reuse", "256", "-b", "1024", "-ub", "512"]
        try:  # fallback: llama-cpp-python's server (pip install "llama-cpp-python[server]")
            import llama_cpp.server  # noqa: F401

            return [sys.executable, "-m", "llama_cpp.server", "--model", str(gguf), "--host", "127.0.0.1",
                    "--port", port, "--n_ctx", str(c.get("ctx", 4096)), "--n_threads", str(threads)]
        except ImportError:
            raise RuntimeError("llama-server not found. Run scripts/setup_windows.ps1 (downloads llama.cpp into bin/llama)")

    # ---------- lifecycle ----------
    def start(self, mode: str | None = None, wait: float = 420.0) -> dict:
        with self._lock:
            mode = mode or self.mode
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            if mode == "snapdragon" and not self.device["is_snapdragon"]:
                raise RuntimeError("Snapdragon mode needs a Snapdragon device (set RAGLY_FORCE_SNAPDRAGON=1 to override)")
            c = self.cfg[mode]
            client = LLMClient(c["base_url"], c.get("model", ""))
            self.state, self.error = "starting", None

            if client.healthy():
                log.info("%s engine already running at %s", mode, c["base_url"])
            elif c.get("manage_process", True):
                try:
                    cmd = self._command(mode)
                except RuntimeError as exc:
                    self.state, self.error = "error", str(exc)
                    raise
                log_path = DATA_DIR / f"engine-{mode}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = open(log_path, "ab")
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                log.info("starting %s engine: %s", mode, " ".join(cmd))
                self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=self._log_file,
                                             stderr=subprocess.STDOUT, creationflags=flags)
                deadline = time.time() + wait
                while time.time() < deadline:
                    if self.proc.poll() is not None:
                        self.state = "error"
                        self.error = f"engine exited with code {self.proc.returncode}; see {log_path}"
                        self.proc = None
                        raise RuntimeError(self.error)
                    if client.healthy():
                        break
                    time.sleep(0.5)
                else:
                    self.stop()
                    self.state, self.error = "error", f"engine did not become ready in {wait:.0f}s"
                    raise RuntimeError(self.error)
            else:
                self.state = "error"
                self.error = f"No engine at {c['base_url']} (manage_process is false: start it yourself)"
                raise RuntimeError(self.error)

            self.client, self.mode, self.state = client, mode, "ready"
            save_user_settings({"engine_mode": mode})
            return self.status()

    def stop(self) -> None:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            self.client = None
            self.state = "stopped"

    def switch(self, mode: str) -> dict:
        with self._lock:
            if mode == self.mode and self.state == "ready":
                return self.status()
            if mode == "snapdragon" and not self.device["is_snapdragon"]:
                raise RuntimeError("Snapdragon mode needs a Snapdragon device")
            previous = self.mode
            self.stop()
            try:
                return self.start(mode)
            except Exception as exc:
                log.warning("switch to %s failed (%s); restoring %s", mode, exc, previous)
                err = str(exc)
                try:
                    self.start(previous)
                except Exception:
                    pass
                self.error = err
                raise RuntimeError(err) from exc

    def require_client(self) -> LLMClient:
        if self.client is None or self.state != "ready":
            raise RuntimeError(f"Answer engine is not ready ({self.state}): {self.error or 'start it first'}")
        return self.client

    def status(self) -> dict:
        c = self.cfg[self.mode]
        model = None
        if self.client is not None:
            try:
                model = self.client.model
            except Exception:
                model = None
        return {
            "mode": self.mode,
            "label": c.get("label", self.mode),
            "state": self.state,
            "error": self.error,
            "base_url": c["base_url"],
            "model": model,
            "managed_pid": self.proc.pid if self.proc else None,
            "available_modes": [m for m in MODES if m == "normal" or self.device["is_snapdragon"]],
            "compute": "Hexagon NPU / GenieX" if self.mode == "snapdragon" else "CPU / llama.cpp",
        }

    def embed_providers(self) -> list[str]:
        if self.device["is_snapdragon"]:
            return list(self.cfg["snapdragon"].get("embed_providers", ["CPUExecutionProvider"]))
        return ["CPUExecutionProvider"]
