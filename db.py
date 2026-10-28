"""SQLite-backed storage for streaming conversations and transcript segments."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


class ConversationStore:
    """Persists conversations (grouped by date) and their transcript segments."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with _LOCK, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    title TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    audio_path TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_date ON conversations(date)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_conversation ON segments(conversation_id)")

    def create_conversation(self, session_id: str) -> int:
        now = datetime.now()
        with _LOCK, self._conn:
            cur = self._conn.execute(
                "INSERT INTO conversations (session_id, date, started_at) VALUES (?, ?, ?)",
                (session_id, now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")),
            )
            return int(cur.lastrowid)

    def add_segment(self, conversation_id: int, seq: int, text: str) -> None:
        if not text.strip():
            return
        now = datetime.now().isoformat(timespec="seconds")
        with _LOCK, self._conn:
            self._conn.execute(
                "INSERT INTO segments (conversation_id, seq, text, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, seq, text, now),
            )

    def finalize_conversation(self, conversation_id: int, audio_path: str | None = None) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with _LOCK, self._conn:
            rows = self._conn.execute(
                "SELECT text FROM segments WHERE conversation_id = ? ORDER BY seq", (conversation_id,)
            ).fetchall()
            title = None
            if rows:
                combined = " ".join(r["text"] for r in rows)
                title = combined[:60] + ("…" if len(combined) > 60 else "")
            self._conn.execute(
                "UPDATE conversations SET ended_at = ?, audio_path = COALESCE(?, audio_path), "
                "title = COALESCE(title, ?) WHERE id = ?",
                (now, audio_path, title, conversation_id),
            )

    def list_conversations(self, date: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT c.id, c.date, c.title, c.started_at, c.ended_at, c.audio_path, "
            "COUNT(s.id) AS segment_count "
            "FROM conversations c LEFT JOIN segments s ON s.conversation_id = c.id"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if date:
            clauses.append("c.date = ?")
            params.append(date)
        if search:
            clauses.append("c.id IN (SELECT conversation_id FROM segments WHERE text LIKE ?)")
            params.append(f"%{search}%")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY c.id ORDER BY c.started_at DESC"
        with _LOCK:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        with _LOCK:
            conv = self._conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if conv is None:
                return None
            segments = self._conn.execute(
                "SELECT seq, text, created_at FROM segments WHERE conversation_id = ? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        result = dict(conv)
        result["segments"] = [dict(s) for s in segments]
        return result

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with _LOCK, self._conn:
            self._conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def delete_conversation(self, conversation_id: int) -> str | None:
        with _LOCK, self._conn:
            row = self._conn.execute(
                "SELECT audio_path FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            self._conn.execute("DELETE FROM segments WHERE conversation_id = ?", (conversation_id,))
            self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return row["audio_path"] if row else None
