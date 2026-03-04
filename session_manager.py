from __future__ import annotations
import secrets
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Session:
    client_site: str
    max_context_chars: int
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _messages: list[str] = field(default_factory=list)

    def add_turn(self, user_message: str | None, assistant_message: str | None) -> None:
        if user_message:
            self._add_line("User", user_message)
        if assistant_message:
            self._add_line("Assistant", assistant_message)
        self.updated_at = time.time()

    def _add_line(self, speaker: str, text: str) -> None:
        line = f"{speaker}: {text}".strip()
        if not line:
            return
        self._messages.append(line)
        self._trim_to_limit()

    def _trim_to_limit(self) -> None:
        if self.max_context_chars <= 0:
            return
        # Drop oldest messages until the serialized transcript fits the limit.
        while self._messages and len("\n".join(self._messages)) > self.max_context_chars:
            self._messages.pop(0)

    def context(self) -> str:
        if not self._messages:
            return ""
        return "\n".join(self._messages).strip()


class SessionManager:
    def __init__(self, default_max_context_chars: int = 5000):
        self.default_max_context_chars = max(int(default_max_context_chars), 0)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def create_session(self, client_site: str | None = None, max_context_chars: int | None = None) -> str:
        max_chars = self._coerce_max_chars(max_context_chars)
        session_id = secrets.token_urlsafe(16)
        normalized_site = (client_site or "").strip().lower()
        with self._lock:
            self._sessions[session_id] = _Session(client_site=normalized_site, max_context_chars=max_chars)
        return session_id

    def end_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def get_session(self, session_id: str) -> _Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_context(self, session_id: str) -> str | None:
        session = self.get_session(session_id)
        if not session:
            return None
        session.updated_at = time.time()
        return session.context()

    def append_turn(self, session_id: str, user_message: str | None, assistant_message: str | None) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        session.add_turn(user_message, assistant_message)
        return True

    def _coerce_max_chars(self, max_context_chars: int | None) -> int:
        if max_context_chars is None:
            return self.default_max_context_chars
        try:
            val = int(max_context_chars)
            if val < 0:
                return self.default_max_context_chars
            return val
        except Exception:
            return self.default_max_context_chars

