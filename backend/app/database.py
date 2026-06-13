import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import settings


def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                audio_path TEXT NOT NULL,
                transcript TEXT NOT NULL,
                generated_note_json TEXT NOT NULL
            )
            """
        )


def create_consultation(audio_path: str, transcript: str, note: dict[str, str]) -> dict[str, Any]:
    note_json = json.dumps(note, ensure_ascii=False)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO consultations (audio_path, transcript, generated_note_json)
            VALUES (?, ?, ?)
            """,
            (audio_path, transcript, note_json),
        )
        consultation_id = cursor.lastrowid
        row = conn.execute(
            """
            SELECT id, created_at, audio_path, transcript, generated_note_json
            FROM consultations
            WHERE id = ?
            """,
            (consultation_id,),
        ).fetchone()
    return row_to_consultation(row)


def list_consultations() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, audio_path, transcript, generated_note_json
            FROM consultations
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()
    return [row_to_consultation(row) for row in rows]


def row_to_consultation(row: sqlite3.Row) -> dict[str, Any]:
    generated_note = json.loads(row["generated_note_json"])
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "audio_path": row["audio_path"],
        "transcript": row["transcript"],
        "note": generated_note,
        "generated_note_json": generated_note,
    }
