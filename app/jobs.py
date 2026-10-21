"""Categorizes an uploaded export in the background, with live progress.

Upload used to classify every merchant inside the request, so the browser sat
on a blank page for as long as Ollama took - a minute or more for a year of
transactions - with no sign anything was happening, and a dropped connection
lost the whole upload. Now upload saves the session straight away (status
"categorizing") and starts this job; the review URL shows a progress screen
that polls progress() until the job marks the session "ready".

Each answer is written to the session as it arrives, so a pod restart
mid-run loses nothing: opening the review again restarts the job for just
the merchants still without an answer (see resume_if_stalled()).

Live detail that isn't worth persisting - which merchants Ollama is working
on right now, timings, the recent-answers feed - lives in _progress, in
memory only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from app import classifier, sessions
from app.classifier import Classification

logger = logging.getLogger(__name__)

# Deck order once everything is answered: what the classifier was confident
# about first (quick confirms that build a streak), then what only Ollama
# could answer; within each, the merchants with the most transactions first.
_SOURCE_ORDER = {"classifier": 0, "ollama": 1}

_tasks: dict[str, asyncio.Task] = {}
_progress: dict[str, dict[str, Any]] = {}


def running(session_id: str) -> bool:
    task = _tasks.get(session_id)
    return task is not None and not task.done()


def start(session_id: str) -> None:
    if not running(session_id):
        _tasks[session_id] = asyncio.create_task(_run(session_id))


def resume_if_stalled(session: dict[str, Any]) -> None:
    """A session still marked categorizing with no job behind it was cut off
    by a restart: pick up where it stopped."""
    if session.get("status") == "categorizing" and not running(session["id"]):
        logger.info("Resuming categorization of session %s", session["id"])
        start(session["id"])


def progress(session_id: str) -> dict[str, Any] | None:
    session = sessions.load(session_id)
    if session is None:
        return None
    merchants = session["merchants"]
    done = sum(1 for m in merchants if m["suggested"] is not None)
    live = _progress.get(session_id, {})
    now = time.monotonic()
    started = live.get("started")
    ollama_started = live.get("ollama_started")
    ollama_done = live.get("ollama_done", 0)
    remaining = len(merchants) - done

    # ETA from Ollama's actual throughput so far: memory and classifier
    # answers are instant and all land in the first moment, so whatever is
    # still open is waiting on Ollama.
    eta = None
    if ollama_started is not None and ollama_done:
        eta = (now - ollama_started) / ollama_done * remaining

    return {
        "status": session.get("status", "ready"),
        "running": running(session_id),
        "total": len(merchants),
        "done": done,
        "transactions": len(session["transactions"]),
        "counts": live.get("counts", {}),
        "model": classifier.OLLAMA_MODEL,
        "elapsed": round(now - started) if started is not None else None,
        "eta": round(eta) if eta is not None else None,
        "in_flight": [
            {"merchant": key, "seconds": round(now - since, 1)}
            for key, since in live.get("in_flight", {}).items()
        ],
        "recent": list(live.get("recent", [])),
        "error": live.get("error"),
        "error_count": live.get("error_count", 0),
    }


async def _run(session_id: str) -> None:
    session = sessions.load(session_id)
    if session is None:
        return
    pending = {m["key"]: m.get("direction") for m in session["merchants"] if m["suggested"] is None}
    live: dict[str, Any] = {
        "started": time.monotonic(),
        "ollama_started": None,
        "ollama_done": 0,
        "in_flight": {},
        "recent": deque(maxlen=12),
        "counts": {"memory": 0, "classifier": 0, "ollama": 0},
        "error": None,
        "error_count": 0,
    }
    _progress[session_id] = live

    def on_start(key: str) -> None:
        if live["ollama_started"] is None:
            live["ollama_started"] = time.monotonic()
        live["in_flight"][key] = time.monotonic()

    def on_result(key: str, c: Classification, seconds: float, error: str | None) -> None:
        live["in_flight"].pop(key, None)
        live["counts"][c.source] += 1
        if c.source == "ollama":
            live["ollama_done"] += 1
        if error:
            live["error"] = error
            live["error_count"] += 1
        live["recent"].appendleft(
            {"merchant": key, "category": c.category, "source": c.source, "seconds": round(seconds, 1), "error": bool(error)}
        )

        def record(s: dict[str, Any]) -> None:
            m = next(m for m in s["merchants"] if m["key"] == key)
            m["suggested"] = c.category
            m["source"] = c.source
            m["confidence"] = round(c.confidence * 100) if c.confidence is not None else None
            # Remembered merchants count as already answered: that's the
            # whole point of remembering them. Everything else waits for you.
            m["category"] = c.category if c.source == "memory" else None

        sessions.update(session_id, record)

    try:
        await classifier.classify_merchants(pending, on_start, on_result)
    except Exception:
        logger.exception("Categorizing session %s failed", session_id)
        live["error"] = "Categorizing stopped unexpectedly. Reopen this page to try again."
        return

    def finish(s: dict[str, Any]) -> None:
        s["merchants"].sort(key=lambda m: (_SOURCE_ORDER.get(m["source"], -1), -m["count"]))
        s["status"] = "ready"
        # Kept on the session so the review screen can say why some cards
        # came back Uncategorized, after the progress screen is gone.
        s["categorize_error"] = (
            {"message": live["error"], "count": live["error_count"]} if live["error_count"] else None
        )

    sessions.update(session_id, finish)
    counts = live["counts"]
    logger.info(
        "Session %s: %d merchants categorized in %.0fs: %d remembered, %d from the local "
        "classifier, %d from Ollama (%d failed)",
        session_id,
        len(pending),
        time.monotonic() - live["started"],
        counts["memory"],
        counts["classifier"],
        counts["ollama"],
        live["error_count"],
    )
