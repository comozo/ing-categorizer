"""Calls a local Ollama model to classify one *merchant* at a time.

Uses Ollama's structured-output mode (a JSON schema passed as `format`) so a
small model's output is constrained to exactly one of the fixed categories,
rather than parsed out of free text.

Two things this module is deliberately careful about, both load-bearing for
why this is safe to point at a shared/remote Ollama at all:

- **Minimal payload.** Only the transaction description crosses the network -
  never the date, the amount, or the account it came from. The description
  text is already the only field the categorization rules below actually
  need; everything else would just be extra information leaving the machine
  for no benefit.
- **Deduplication.** A statement commonly has the same merchant many times
  over (the same supermarket every week, the same subscription every
  month), each with a unique receipt/date tail. Classification works on
  merchant keys (app/merchant.py), never on transactions: Ollama is called
  once per *unknown merchant*, with one representative description.
  Those calls run concurrently, capped by OLLAMA_CONCURRENCY - an unbounded
  burst just queues inside Ollama until the later requests hit their
  timeout and come back Uncategorized.

classify_merchants() is the entry point main.py calls. Each merchant goes to
the first source that can answer it:

1. memory - an exact merchant-key match against what you've confirmed
   before (store.known_merchants()). No guessing, no network call.
2. classifier - the local scikit-learn model (app/classifier_ml.py), when
   it's confident enough.
3. ollama - everything else.

Each result carries its source and confidence, which main.py uses to decide
what goes on the review deck at all: memory hits are filled in without
asking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import httpx

from app import classifier_ml, store
from app.categories import CATEGORIES

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_CONCURRENCY = int(os.environ.get("OLLAMA_CONCURRENCY", "4"))

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": CATEGORIES}},
    "required": ["category"],
}

_SYSTEM_PROMPT = (
    "You categorize a bank transaction description from an Australian ING account "
    "(an everyday transaction account, a term deposit, or a variable-rate savings "
    "account) into exactly one category from this fixed list:\n"
    f"{', '.join(CATEGORIES)}\n\n"
    "You are given only the transaction description - no date, no amount, no account "
    "details. Rules:\n"
    "- If the description indicates money moving between the user's own ING "
    "accounts (e.g. contains 'transfer', 'TFR', 'Everyday', 'Savings Maximiser', "
    "'Term Deposit'), use Transfers.\n"
    "- If the description mentions interest credited, use Interest.\n"
    "- If you are not reasonably confident, use Uncategorized rather than guessing.\n"
    "Respond with only the JSON object described by the schema."
)


@dataclass(frozen=True)
class Classification:
    category: str
    # None for Ollama - it commits to one answer via structured output, it
    # doesn't expose a calibrated probability the way the local classifier does.
    confidence: float | None
    source: Literal["memory", "classifier", "ollama"]


async def classify(client: httpx.AsyncClient, description: str) -> str:
    """Classifies a single description. No date, amount, or account info is sent -
    this is the only thing that leaves the machine, ever, for any transaction."""
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Description: {description}"},
                ],
                "format": _RESPONSE_SCHEMA,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        category = json.loads(content).get("category", "Uncategorized")
        if category not in CATEGORIES:
            logger.warning(
                "Ollama returned a category outside the fixed list for %r: %r",
                description,
                category,
            )
            return "Uncategorized"
        return category
    except httpx.ConnectError as exc:
        logger.error(
            "Couldn't connect to Ollama at %s while classifying %r: %s. "
            "Check OLLAMA_URL is correct and reachable from this cluster.",
            OLLAMA_URL,
            description,
            exc,
        )
        return "Uncategorized"
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ollama returned HTTP %s while classifying %r: %s",
            exc.response.status_code,
            description,
            exc.response.text[:500],
        )
        return "Uncategorized"
    except httpx.HTTPError as exc:
        logger.error("Request to Ollama failed while classifying %r: %s", description, exc)
        return "Uncategorized"
    except (KeyError, ValueError) as exc:
        logger.error(
            "Couldn't parse Ollama's response while classifying %r: %s", description, exc
        )
        return "Uncategorized"


async def classify_merchants(merchants: dict[str, str]) -> dict[str, Classification]:
    """Classifies each merchant key, given one representative description per
    key: memory first, then the local classifier, then Ollama (concurrent,
    capped at OLLAMA_CONCURRENCY) for whatever's left."""
    memory = store.known_merchants()
    results: dict[str, Classification] = {}
    needs_ollama: list[str] = []

    for key in merchants:
        if key in memory:
            results[key] = Classification(memory[key], 1.0, "memory")
        elif (guess := classifier_ml.predict(key)) is not None:
            category, confidence = guess
            results[key] = Classification(category, confidence, "classifier")
        else:
            needs_ollama.append(key)

    if needs_ollama:
        limit = asyncio.Semaphore(OLLAMA_CONCURRENCY)

        async def bounded(client: httpx.AsyncClient, description: str) -> str:
            async with limit:
                return await classify(client, description)

        async with httpx.AsyncClient() as client:
            categories = await asyncio.gather(
                *(bounded(client, merchants[key]) for key in needs_ollama)
            )
        for key, category in zip(needs_ollama, categories):
            results[key] = Classification(category, None, "ollama")

    return results
