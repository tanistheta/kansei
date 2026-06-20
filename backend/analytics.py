"""
Kansei analytics — minimal self-rolled event logging.

Design choices, explained:
- SQLite, not Postgres: zero setup, file-based, fine up to tens of thousands
  of events/day. Railway gives you a persistent volume if you mount one —
  see deployment note at bottom of this file.
- One table, not a "users" + "events" schema: you don't have auth, you don't
  need joins yet. A session_id (random string set client-side) is enough to
  reconstruct a funnel per visitor. Don't over-normalize data you don't have
  a query for yet.
- Fire-and-forget: the endpoint never raises in a way that breaks the
  frontend. If logging fails, the user's quiz still works. Analytics should
  never be allowed to take down the product it's measuring.
"""

import sqlite3
import time
import json
from pathlib import Path
from fastapi import APIRouter, Request
from pydantic import BaseModel

DB_PATH = Path(__file__).parent / "analytics.db"

router = APIRouter()


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            ts REAL NOT NULL,
            meta TEXT
        )
    """)
    # Index on session_id since every funnel query groups by it
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_name ON events(event_name)")
    return conn


class EventIn(BaseModel):
    session_id: str
    event_name: str
    meta: dict | None = None


@router.post("/api/event")
async def log_event(event: EventIn):
    """
    Fire-and-forget event log. Always returns 200 even if write fails —
    a 500 here would show up in browser dev tools and is pointless noise
    for something the user never sees.
    """
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO events (session_id, event_name, ts, meta) VALUES (?, ?, ?, ?)",
            (event.session_id, event.event_name, time.time(),
             json.dumps(event.meta) if event.meta else None)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # analytics must never break the app
    return {"ok": True}


@router.get("/api/event/summary")
async def summary():
    """
    Quick funnel readout. Hit this in your browser at
    kansei.up.railway.app/api/event/summary — no dashboard needed yet.
    """
    conn = get_conn()
    rows = conn.execute("""
        SELECT event_name, COUNT(*) as count, COUNT(DISTINCT session_id) as unique_sessions
        FROM events
        GROUP BY event_name
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return {
        "funnel": [
            {"event": r[0], "total": r[1], "unique_sessions": r[2]}
            for r in rows
        ]
    }