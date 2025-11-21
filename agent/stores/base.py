from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Optional, Any


@dataclass
class SessionRecord:
    id: int
    user_id: str
    topic: str
    goal_kind: str
    created_at: str
    thread_id: int | None = None
    language: str | None = None


class SessionRepository(abc.ABC):
    """Abstract repository for session persistence.

    This abstraction allows swapping SQLite → Redis/Postgres without changing callers.
    """

    @abc.abstractmethod
    async def init(self) -> None:
        """Initialize storage (create tables/indexes)."""

    @abc.abstractmethod
    async def create_session(self, *, user_id: str, topic: str, goal_kind: str, created_at_iso: str, thread_id: int | None = None, language: str | None = None) -> int:
        """Create a session and return its integer id."""

    @abc.abstractmethod
    async def get_latest_session_by_user(self, user_id: str) -> Optional[SessionRecord]:
        """Return the most recent session for the user if exists."""

    @abc.abstractmethod
    async def get_session_by_thread(self, *, user_id: str, thread_id: int) -> Optional[SessionRecord]:
        """Return the most recent session for a (user, thread_id) pair if exists."""

    @abc.abstractmethod
    async def add_message(self, *, session_id: int, ts_iso: str, role: str, text: str) -> None:
        """Append a message to the session transcript."""

    @abc.abstractmethod
    async def iter_messages(self, *, session_id: int) -> Iterable[tuple[str, str, str]]:
        """Yield (ts_iso, role, text) for the session in order."""

    # --- Slots (M1b) ---
    @abc.abstractmethod
    async def upsert_slot(self, *, session_id: int, slot: dict[str, Any]) -> None:
        """Insert or update a slot row for the given session."""

    @abc.abstractmethod
    async def set_slot_value(self, *, session_id: int, slot_name: str, value: str, source_kind: str = "user", filled_ratio: float | None = None, last_filled_ts: float | None = None) -> None:
        """Set a slot value (and optionally ratio/timestamp)."""

    @abc.abstractmethod
    async def get_slots(self, *, session_id: int) -> list[dict[str, Any]]:
        """Return all slot rows for the session."""


