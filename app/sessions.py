"""Review sessions: the checkpoint that lets a review survive a closed tab, a
dropped connection or a pod restart.

An upload is parsed and classified once, then saved here as a session. Every
answer on the review screen is written back as you make it, so reopening the
session's URL (or picking it from the home page) resumes at the first merchant
you haven't answered yet, with your XP and streak intact.

Unlike training_pairs.csv (app/store.py), a session holds the full parsed
statement - dates and amounts included - because the final CSV is built from
it. It lives on the same data volume, never leaves the machine, and is
deleted once it's older than SESSION_RETENTION_DAYS, or straight away with
"Discard" on the home page.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.store import DATA_DIR

SESSIONS_DIR = DATA_DIR / "sessions"
RETENTION_DAYS = int(os.environ.get("SESSION_RETENTION_DAYS", "30"))

# Session ids become filenames: accept only what new_id() produces, so a
# crafted id can never point outside SESSIONS_DIR.
_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")

# One uvicorn worker, but the sync file handlers below run in FastAPI's thread
# pool: serialise read-modify-write so two quick answers can't drop each other.
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(session_id: str) -> Path | None:
    if not _ID_PATTERN.match(session_id):
        return None
    return SESSIONS_DIR / f"{session_id}.json"


def _write(session: dict[str, Any]) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{session['id']}.json"
    # Write-then-rename: a crash mid-write leaves the previous checkpoint
    # intact instead of a truncated file.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session), encoding="utf-8")
    os.replace(tmp, path)


def create(
    filename: str, transactions: list[dict], merchants: list[dict], status: str = "ready"
) -> dict[str, Any]:
    session = {
        "id": secrets.token_hex(8),
        # "categorizing" while app/jobs.py fills in suggestions, then "ready".
        # Sessions saved before this field existed are ready.
        "status": status,
        "filename": filename,
        "created_at": now(),
        "updated_at": now(),
        "completed_at": None,
        "transactions": transactions,
        "merchants": merchants,
        "stats": {"xp": 0, "streak": 0, "best_streak": 0},
        # merchant key -> category last fed to the classifier, so downloading
        # twice doesn't add the same training pairs twice.
        "trained": {},
    }
    with _lock:
        _write(session)
    return session


def load(session_id: str) -> dict[str, Any] | None:
    path = _path(session_id)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update(session_id: str, mutate) -> dict[str, Any] | None:
    """Loads, applies mutate(session) and saves, atomically with respect to
    other updates. Returns the saved session, or None if it doesn't exist."""
    with _lock:
        session = load(session_id)
        if session is None:
            return None
        mutate(session)
        session["updated_at"] = now()
        _write(session)
        return session


def delete(session_id: str) -> None:
    path = _path(session_id)
    if path is not None:
        with _lock:
            path.unlink(missing_ok=True)


def summaries() -> list[dict[str, Any]]:
    """Every session, newest first, for the home page. Prunes expired ones."""
    if not SESSIONS_DIR.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    result = []
    for path in SESSIONS_DIR.glob("*.json"):
        session = json.loads(path.read_text(encoding="utf-8"))
        if datetime.fromisoformat(session["updated_at"]) < cutoff:
            delete(session["id"])
            continue
        deck = [m for m in session["merchants"] if m["source"] != "memory"]
        result.append(
            {
                "id": session["id"],
                "filename": session["filename"],
                "created_at": session["created_at"],
                "completed": session["completed_at"] is not None,
                "categorizing": session.get("status") == "categorizing",
                "answered": sum(1 for m in deck if m["category"] is not None),
                "deck_size": len(deck),
                "transaction_count": len(session["transactions"]),
            }
        )
    return sorted(result, key=lambda s: s["created_at"], reverse=True)
