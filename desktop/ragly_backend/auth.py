"""Local passcode lock for the workspace.

Honest description of what this is: the backend already refuses anything that is not
localhost, so this is a lock on the app on this machine, not network security. It stops
someone who walks up to an unlocked laptop from reading the indexed documents through the
UI. The passcode is stored as a PBKDF2-SHA256 hash (never in plain text) in the local data
folder, and sessions are signed with a key generated on this machine.

It does NOT encrypt the database — that is a separate feature, and the UI says so.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, ensure_dirs

AUTH_PATH = DATA_DIR / "auth.json"
ITERATIONS = 200_000
SESSION_HOURS = 12
MIN_LENGTH = 4


def _hash(passcode: str, salt: bytes) -> str:
    return base64.b64encode(
        hashlib.pbkdf2_hmac("sha256", passcode.encode("utf-8"), salt, ITERATIONS)).decode()


@dataclass
class AuthState:
    enabled: bool
    display_name: str
    created_at: float | None = None

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "display_name": self.display_name, "created_at": self.created_at}


class Auth:
    """File-backed local account. One user per installation, which matches a desktop app."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or AUTH_PATH)
        self._data = self._load()

    # ---------- storage ----------
    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        ensure_dirs()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # ---------- state ----------
    @property
    def enabled(self) -> bool:
        return bool(self._data.get("hash"))

    def state(self) -> AuthState:
        return AuthState(enabled=self.enabled,
                         display_name=self._data.get("display_name", ""),
                         created_at=self._data.get("created_at"))

    # ---------- account ----------
    def register(self, passcode: str, display_name: str = "") -> str:
        if self.enabled:
            raise ValueError("an account already exists on this device")
        if len(passcode) < MIN_LENGTH:
            raise ValueError(f"the passcode needs at least {MIN_LENGTH} characters")
        salt = secrets.token_bytes(16)
        self._data = {
            "salt": base64.b64encode(salt).decode(),
            "hash": _hash(passcode, salt),
            "secret": secrets.token_hex(32),
            "display_name": display_name.strip() or "Local user",
            "created_at": time.time(),
        }
        self._save()
        return self.issue_token()

    def verify(self, passcode: str) -> bool:
        if not self.enabled:
            return True
        salt = base64.b64decode(self._data["salt"])
        return hmac.compare_digest(_hash(passcode, salt), self._data["hash"])

    def change_passcode(self, old: str, new: str) -> str:
        if not self.verify(old):
            raise ValueError("the current passcode is wrong")
        if len(new) < MIN_LENGTH:
            raise ValueError(f"the passcode needs at least {MIN_LENGTH} characters")
        salt = secrets.token_bytes(16)
        self._data.update({"salt": base64.b64encode(salt).decode(), "hash": _hash(new, salt),
                           "secret": secrets.token_hex(32)})
        self._save()
        return self.issue_token()

    def disable(self, passcode: str) -> None:
        if not self.verify(passcode):
            raise ValueError("the passcode is wrong")
        self._data = {}
        self._save()

    def set_display_name(self, name: str) -> None:
        self._data["display_name"] = name.strip() or "Local user"
        self._save()

    # ---------- sessions ----------
    def issue_token(self, hours: int = SESSION_HOURS) -> str:
        expires = int(time.time() + hours * 3600)
        payload = f"{expires}"
        signature = hmac.new(self._data.get("secret", "none").encode(), payload.encode(),
                             hashlib.sha256).hexdigest()[:32]
        return f"{payload}.{signature}"

    def valid(self, token: str | None) -> bool:
        if not self.enabled:
            return True
        if not token or "." not in token:
            return False
        payload, signature = token.rsplit(".", 1)
        expected = hmac.new(self._data.get("secret", "none").encode(), payload.encode(),
                            hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(signature, expected):
            return False
        try:
            return int(payload) > time.time()
        except ValueError:
            return False
