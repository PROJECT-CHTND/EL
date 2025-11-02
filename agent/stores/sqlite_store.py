from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from agent.stores.base import SessionRecord, SessionRepository


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


class SqliteSessionRepository(SessionRepository):
    def __init__(self, db_path: Optional[str] = None) -> None:
        default_path = os.getenv("EL_SQLITE_PATH") or str(Path.cwd() / "data" / "el_sessions.db")
        self.db_path = db_path or default_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        def _init() -> None:
            conn = _connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        goal_kind TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        ts_iso TEXT NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);")
                conn.close()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        await asyncio.to_thread(_init)

    async def create_session(self, *, user_id: str, topic: str, goal_kind: str, created_at_iso: str) -> int:
        def _create() -> int:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO sessions(user_id, topic, goal_kind, created_at) VALUES (?, ?, ?, ?)",
                (user_id, topic, goal_kind, created_at_iso),
            )
            sid = int(cur.lastrowid)
            conn.close()
            return sid

        return await asyncio.to_thread(_create)

    async def get_latest_session_by_user(self, user_id: str) -> Optional[SessionRecord]:
        def _get() -> Optional[SessionRecord]:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT id, user_id, topic, goal_kind, created_at FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            return SessionRecord(id=int(row[0]), user_id=row[1], topic=row[2], goal_kind=row[3], created_at=row[4])

        return await asyncio.to_thread(_get)

    async def add_message(self, *, session_id: int, ts_iso: str, role: str, text: str) -> None:
        def _add() -> None:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages(session_id, ts_iso, role, text) VALUES (?, ?, ?, ?)",
                (session_id, ts_iso, role, text),
            )
            conn.close()

        await asyncio.to_thread(_add)

    async def iter_messages(self, *, session_id: int) -> Iterable[tuple[str, str, str]]:
        def _iter() -> list[tuple[str, str, str]]:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT ts_iso, role, text FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,),
            )
            rows = [(str(r[0]), str(r[1]), str(r[2])) for r in cur.fetchall()]
            conn.close()
            return rows

        rows = await asyncio.to_thread(_iter)
        for r in rows:
            yield r


