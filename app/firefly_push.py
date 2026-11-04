"""Runs a send-to-Firefly in the background and records how it went.

A year of transactions takes the importer a while, longer than a browser
request should hang, so the review screen starts this job and shows a status
page that polls it. The outcome is stored on the session ("firefly"), so it
survives a reload and the review screen can say when it was last sent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app import firefly, sessions

logger = logging.getLogger(__name__)

# Rows listed by name when some are missing; the count is always exact.
_MISSING_SHOWN = 50

_tasks: dict[str, asyncio.Task] = {}


def running(session_id: str) -> bool:
    task = _tasks.get(session_id)
    return task is not None and not task.done()


def start(session_id: str, account_id: str, account_name: str, csv_text: str, expected: list[dict[str, str]], filename: str) -> None:
    if running(session_id):
        return
    state = {
        "state": "uploading",
        "account_id": account_id,
        "account_name": account_name,
        "account_url": firefly.account_url(account_id),
        "started_at": sessions.now(),
        "finished_at": None,
        "expected": len(expected),
        "present": None,
        "missing": [],
        "missing_count": None,
        "balance": None,
        "currency": "",
        "importer_message": None,
        "error": None,
    }
    sessions.update(session_id, lambda s: s.update(firefly=state))
    _tasks[session_id] = asyncio.create_task(_run(session_id, account_id, csv_text, expected, filename))


def _record(session_id: str, **changes: Any) -> None:
    sessions.update(session_id, lambda s: s["firefly"].update(changes))


async def _run(session_id: str, account_id: str, csv_text: str, expected: list[dict[str, str]], filename: str) -> None:
    try:
        message = await firefly.upload(csv_text, firefly.import_config(csv_text, account_id, filename), filename)
        _record(session_id, state="verifying", importer_message=message)

        dates = sorted(row["date"] for row in expected if row["date"])
        present = await firefly.external_ids_on_account(account_id, dates[0], dates[-1]) if dates else set()
        missing = [row for row in expected if row["uid"] not in present]
        details = await firefly.account(account_id)
        _record(
            session_id,
            state="done",
            finished_at=sessions.now(),
            present=len(expected) - len(missing),
            missing_count=len(missing),
            missing=missing[:_MISSING_SHOWN],
            balance=details["balance"],
            currency=details["currency"],
        )
        logger.info("Session %s sent to Firefly account %s: %d of %d present", session_id, account_id, len(expected) - len(missing), len(expected))
    except firefly.FireflyError as exc:
        logger.error("Sending session %s to Firefly failed: %s", session_id, exc)
        _record(session_id, state="failed", finished_at=sessions.now(), error=str(exc))
    except Exception:
        logger.exception("Sending session %s to Firefly failed unexpectedly", session_id)
        _record(session_id, state="failed", finished_at=sessions.now(), error="Something went wrong while sending. The categorizer's log has the details.")
