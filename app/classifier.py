"""Calls a local Ollama model to classify one *merchant* at a time.

Uses Ollama's structured-output mode (a JSON schema passed as `format`) so a
small model's output is constrained to exactly one of the fixed categories,
rather than parsed out of free text.

Two things this module is deliberately careful about, both load-bearing for
why this is safe to point at a shared/remote Ollama at all:

- **Minimal payload.** Per merchant, only the cleaned merchant name
  (app/merchant.py - receipt numbers, dates and card digits already
  stripped) and whether money went out or came in cross the network - never
  a date, an amount, or the account. Direction is the one extra field worth
  its cost: without it a small model can't tell a refund from a purchase or
  a salary from a transfer. The prompt also carries up to FEW_SHOT_EXAMPLES
  of your own confirmed (merchant -> category) answers, nearest to the
  merchant being asked about, so the model follows your conventions instead
  of guessing at them.
- **Deduplication.** A statement commonly has the same merchant many times
  over (the same supermarket every week, the same subscription every
  month), each with a unique receipt/date tail. Classification works on
  merchant keys (app/merchant.py), never on transactions: Ollama is called
  once per *unknown merchant*, never once per transaction.
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
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app import classifier_ml, store
from app.categories import CATEGORIES, CATEGORY_HINTS

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_CONCURRENCY = int(os.environ.get("OLLAMA_CONCURRENCY", "4"))
FEW_SHOT_EXAMPLES = int(os.environ.get("FEW_SHOT_EXAMPLES", "30"))

Direction = Literal["out", "in"]

# Uncategorized is left out on purpose: offered an "I'm not sure" answer, a 3B
# model takes it for anything slightly unusual, which is the review screen's
# job to catch, not the model's. Other is the honest "nothing fits".
_ANSWERS = [c for c in CATEGORIES if c != "Uncategorized"]
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": _ANSWERS}},
    "required": ["category"],
}

_INSTRUCTIONS = (
    "You categorize transactions from an Australian ING bank account. For each one you get "
    "the merchant name as it appears on the statement (receipt numbers, dates and card "
    "digits removed) and whether money went out or came in. Answer with the single "
    "category below that fits best; use Other only when none of them does.\n\n"
    "Categories:\n"
    + "\n".join(f"- {name}: {CATEGORY_HINTS[name]}" for name in _ANSWERS)
    + "\n\nRules:\n"
    "- Money moving between the user's own ING accounts (transfer, TFR, Everyday, Savings "
    "Maximiser, Term Deposit) is Transfers.\n"
    "- Money coming in is usually Salary & Income, Interest or Transfers; a refund goes in "
    "the category of whatever was refunded."
)


def build_messages(
    merchant: str, direction: Direction | None, examples: list[tuple[str, str]]
) -> list[dict[str, str]]:
    system = _INSTRUCTIONS
    if examples:
        system += (
            "\n\nThis user's own past answers for similar merchants. Follow them: the same "
            "merchant, or the same chain in another suburb, gets the same category.\n"
            + "\n".join(f"- {key} -> {category}" for key, category in examples)
        )
    user = f"Merchant: {merchant}"
    if direction is not None:
        user += f"\nMoney: {'went out' if direction == 'out' else 'came in'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_examples_index: tuple[dict[str, str], list[str], TfidfVectorizer, object] | None = None


def similar_examples(merchant: str, memory: dict[str, str], k: int = FEW_SHOT_EXAMPLES) -> list[tuple[str, str]]:
    """The k confirmed merchants whose names look most like this one, by
    character n-grams - so "WOOLWORTHS METRO PYRMONT" pulls in "WOOLWORTHS
    SYDNEY", and a merchant's own past answers outrank unrelated ones."""
    global _examples_index
    if not memory or k <= 0:
        return []
    # Rebuilt only when store.known_merchants() hands back a new dict, i.e.
    # after new answers were saved.
    if _examples_index is None or _examples_index[0] is not memory:
        keys = list(memory)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        _examples_index = (memory, keys, vectorizer, vectorizer.fit_transform(keys))
    _, keys, vectorizer, matrix = _examples_index
    scores = linear_kernel(vectorizer.transform([merchant]), matrix)[0]
    nearest = scores.argsort()[::-1][:k]
    # Most similar last, right before the question, where a small model
    # weighs it most.
    return [(keys[i], memory[keys[i]]) for i in reversed(nearest) if keys[i] != merchant]


@dataclass(frozen=True)
class Classification:
    category: str
    # None for Ollama - it commits to one answer via structured output, it
    # doesn't expose a calibrated probability the way the local classifier does.
    confidence: float | None
    source: Literal["memory", "classifier", "ollama"]


class OllamaError(Exception):
    """Ollama couldn't give an answer: unreachable, an HTTP error, or a reply
    that doesn't parse. The message is what the progress screen shows."""


async def ask_ollama(
    client: httpx.AsyncClient,
    merchant: str,
    direction: Direction | None = None,
    examples: list[tuple[str, str]] | None = None,
) -> str:
    """Classifies one merchant, raising OllamaError when there's no answer.
    See "Minimal payload" above for exactly what is sent."""
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": build_messages(merchant, direction, examples or []),
                "format": _RESPONSE_SCHEMA,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        category = json.loads(content).get("category", "Uncategorized")
    except httpx.ConnectError as exc:
        logger.error(
            "Couldn't connect to Ollama at %s while classifying %r: %s. "
            "Check OLLAMA_URL is correct and reachable from this cluster.",
            OLLAMA_URL,
            merchant,
            exc,
        )
        raise OllamaError(f"Can't reach Ollama at {OLLAMA_URL}. Is the machine running it awake?") from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ollama returned HTTP %s while classifying %r: %s",
            exc.response.status_code,
            merchant,
            exc.response.text[:500],
        )
        raise OllamaError(f"Ollama answered HTTP {exc.response.status_code}. Is {OLLAMA_MODEL} pulled?") from exc
    except httpx.HTTPError as exc:
        logger.error("Request to Ollama failed while classifying %r: %s", merchant, exc)
        raise OllamaError(f"The request to Ollama failed ({type(exc).__name__}).") from exc
    except (KeyError, ValueError) as exc:
        logger.error(
            "Couldn't parse Ollama's response while classifying %r: %s", merchant, exc
        )
        raise OllamaError("Ollama's reply couldn't be read.") from exc
    if category not in _ANSWERS:
        logger.warning(
            "Ollama returned a category outside the fixed list for %r: %r",
            merchant,
            category,
        )
        return "Uncategorized"
    return category


async def classify(
    client: httpx.AsyncClient,
    merchant: str,
    direction: Direction | None = None,
    examples: list[tuple[str, str]] | None = None,
) -> str:
    """ask_ollama(), with any failure answered as Uncategorized."""
    try:
        return await ask_ollama(client, merchant, direction, examples)
    except OllamaError:
        return "Uncategorized"


async def classify_merchants(
    merchants: dict[str, Direction | None],
    on_start: Callable[[str], None] | None = None,
    on_result: Callable[[str, Classification, float, str | None], None] | None = None,
) -> dict[str, Classification]:
    """Classifies each merchant key (with whether its money went out or came
    in): memory first, then the local classifier, then Ollama (concurrent,
    capped at OLLAMA_CONCURRENCY) for whatever's left.

    on_start(key) fires when a merchant's Ollama request is sent; on_result
    (key, classification, seconds, error) fires as each merchant gets its
    answer, from any source, with the error message when Ollama failed and
    the answer fell back to Uncategorized. The progress screen is built on
    these two."""
    memory = store.known_merchants()
    results: dict[str, Classification] = {}
    needs_ollama: list[str] = []

    def report(key: str, result: Classification, seconds: float = 0.0, error: str | None = None) -> None:
        results[key] = result
        if on_result is not None:
            on_result(key, result, seconds, error)

    for key in merchants:
        if key in memory:
            report(key, Classification(memory[key], 1.0, "memory"))
        elif (guess := classifier_ml.predict(key)) is not None:
            category, confidence = guess
            report(key, Classification(category, confidence, "classifier"))
        else:
            needs_ollama.append(key)

    if needs_ollama:
        limit = asyncio.Semaphore(OLLAMA_CONCURRENCY)

        async def bounded(client: httpx.AsyncClient, key: str) -> None:
            async with limit:
                if on_start is not None:
                    on_start(key)
                started = time.monotonic()
                error = None
                try:
                    category = await ask_ollama(client, key, merchants[key], similar_examples(key, memory))
                except OllamaError as exc:
                    category, error = "Uncategorized", str(exc)
                report(key, Classification(category, None, "ollama"), time.monotonic() - started, error)

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(bounded(client, key) for key in needs_ollama))

    return results
