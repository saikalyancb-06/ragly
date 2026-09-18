"""Offline guard: refuses any outbound connection that is not to this machine.

Installed at startup so a bug or a library can never send document data off-device.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

_installed = False
_blocked: list[str] = []
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.strip("[]").lower()
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_local_url(url: str) -> None:
    host = urlparse(url).hostname
    if not is_loopback_host(host):
        raise ValueError(f"Refusing non-local endpoint {url!r}: Ragly only talks to localhost.")


def _check(address) -> None:
    host = None
    if isinstance(address, tuple) and address:
        host = str(address[0])
    elif isinstance(address, (str, bytes)):
        return  # AF_UNIX socket path: local by definition
    if host is not None and not is_loopback_host(host):
        _blocked.append(host)
        raise ConnectionRefusedError(f"Offline guard blocked a connection to {host}")


def install() -> None:
    """Patch socket.connect / connect_ex process-wide."""
    global _installed
    if _installed:
        return
    # Make sure no library tries to fetch models at runtime.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(address)
        return orig_connect(self, address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(address)
        return orig_connect_ex(self, address)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    _installed = True


def status() -> dict:
    return {"installed": _installed, "blocked_attempts": len(_blocked), "last_blocked": _blocked[-5:]}
