"""Second opinion on remembered merchants, for the Known merchants page.

Memory is trusted completely - a remembered merchant is filled in without
asking - so a wrong answer that got accepted once (an AI pick confirmed
without a close look) repeats on every upload. A recheck asks Ollama about
every remembered merchant and the page flags the ones where it disagrees.

Unlike a normal upload, the prompt carries none of your past answers: those
are what's being checked, and a wrong one would vouch for its own siblings
("MCDONALDS CASTLE TWR -> Shopping" convincing the model that every other
McDonald's is Shopping too). Your deliberate conventions (say, a phone plan
filed as Subscriptions) will get flagged as a result - keep them as they are.

It runs as a background task, not inside the request: with an 8B model a
full pass takes a minute or more, longer than the gateway lets a request
hang. Results go to DATA_DIR so they survive a restart; the page reads them
back and polls while a run is in progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from app import classifier, store

logger = logging.getLogger(__name__)

RESULTS_PATH = store.DATA_DIR / "recheck.json"

_task: asyncio.Task | None = None


def load() -> dict[str, Any] | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _save(state: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, RESULTS_PATH)


def running() -> bool:
    return _task is not None and not _task.done()


def start() -> None:
    global _task
    if not running():
        _task = asyncio.create_task(_run())


async def _run() -> None:
    memory = dict(store.known_merchants())
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_at": None,
        "model": classifier.OLLAMA_MODEL,
        "total": len(memory),
        "done": 0,
        "suggestions": {},
    }
    _save(state)
    limit = asyncio.Semaphore(classifier.OLLAMA_CONCURRENCY)

    async def one(client: httpx.AsyncClient, key: str) -> None:
        async with limit:
            category = await classifier.classify(client, key, None, [])
        state["suggestions"][key] = category
        state["done"] += 1
        # Saved as it goes, so the page's progress count moves.
        _save(state)

    try:
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(one(client, key) for key in memory))
    except Exception:
        logger.exception("Known-merchant recheck failed")
    state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save(state)
    logger.info("Rechecked %d remembered merchants with %s", state["done"], state["model"])
