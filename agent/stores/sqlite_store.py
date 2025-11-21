from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Any

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
                        thread_id INTEGER,
                        topic TEXT NOT NULL,
                        goal_kind TEXT NOT NULL,
                        language TEXT,
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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS slots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        type TEXT,
                        importance REAL,
                        filled_ratio REAL,
                        last_filled_ts REAL,
                        value TEXT,
                        evidence_ids_json TEXT,
                        source_kind TEXT,
                        UNIQUE(session_id, name),
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_slots_session ON slots(session_id, name);")
                conn.close()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        await asyncio.to_thread(_init)

    async def create_session(self, *, user_id: str, topic: str, goal_kind: str, created_at_iso: str, thread_id: int | None = None, language: str | None = None) -> int:
        def _create() -> int:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO sessions(user_id, thread_id, topic, goal_kind, language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, thread_id, topic, goal_kind, language, created_at_iso),
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
                "SELECT id, user_id, topic, goal_kind, created_at, thread_id, language FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            return SessionRecord(id=int(row[0]), user_id=row[1], topic=row[2], goal_kind=row[3], created_at=row[4], thread_id=row[5], language=row[6])

        return await asyncio.to_thread(_get)

    async def get_session_by_thread(self, *, user_id: str, thread_id: int) -> Optional[SessionRecord]:
        def _get() -> Optional[SessionRecord]:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT id, user_id, topic, goal_kind, created_at, thread_id, language FROM sessions WHERE user_id=? AND thread_id=? ORDER BY id DESC LIMIT 1",
                (user_id, thread_id),
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            return SessionRecord(id=int(row[0]), user_id=row[1], topic=row[2], goal_kind=row[3], created_at=row[4], thread_id=row[5], language=row[6])

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

    # --- Slots (M1b) ---
    async def upsert_slot(self, *, session_id: int, slot: dict[str, Any]) -> None:
        def _upsert() -> None:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO slots(session_id, name, description, type, importance, filled_ratio, last_filled_ts, value, evidence_ids_json, source_kind)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, name) DO UPDATE SET
                  description=excluded.description,
                  type=excluded.type,
                  importance=excluded.importance,
                  filled_ratio=COALESCE(excluded.filled_ratio, slots.filled_ratio),
                  last_filled_ts=COALESCE(excluded.last_filled_ts, slots.last_filled_ts),
                  value=COALESCE(excluded.value, slots.value),
                  evidence_ids_json=COALESCE(excluded.evidence_ids_json, slots.evidence_ids_json),
                  source_kind=COALESCE(excluded.source_kind, slots.source_kind)
                ;
                """
            ,
                (
                    session_id,
                    slot.get("name"),
                    slot.get("description"),
                    slot.get("type"),
                    float(slot.get("importance")) if slot.get("importance") is not None else None,
                    float(slot.get("filled_ratio")) if slot.get("filled_ratio") is not None else None,
                    float(slot.get("last_filled_ts")) if slot.get("last_filled_ts") is not None else None,
                    slot.get("value"),
                    slot.get("evidence_ids_json"),
                    slot.get("source_kind"),
                ),
            )
            conn.close()

        await asyncio.to_thread(_upsert)

    async def set_slot_value(self, *, session_id: int, slot_name: str, value: str, source_kind: str = "user", filled_ratio: float | None = None, last_filled_ts: float | None = None) -> None:
        def _set() -> None:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE slots
                SET value=?, source_kind=?, filled_ratio=COALESCE(?, filled_ratio), last_filled_ts=COALESCE(?, last_filled_ts)
                WHERE session_id=? AND name=?
                """,
                (value, source_kind, filled_ratio, last_filled_ts, session_id, slot_name),
            )
            if cur.rowcount == 0:
                cur.execute(
                    """
                    INSERT INTO slots(session_id, name, value, source_kind, filled_ratio, last_filled_ts)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (session_id, slot_name, value, source_kind, filled_ratio, last_filled_ts),
                )
            conn.close()

        await asyncio.to_thread(_set)

    async def get_slots(self, *, session_id: int) -> list[dict[str, Any]]:
        def _get() -> list[dict[str, Any]]:
            conn = _connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT name, description, type, importance, filled_ratio, last_filled_ts, value, evidence_ids_json, source_kind
                FROM slots WHERE session_id=? ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            conn.close()
            cols = ["name", "description", "type", "importance", "filled_ratio", "last_filled_ts", "value", "evidence_ids_json", "source_kind"]
            return [dict(zip(cols, r)) for r in rows]

        return await asyncio.to_thread(_get)


