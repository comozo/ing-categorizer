"""Calls a local Ollama model to classify one *unique description* at a time.

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
- **Deduplication.** A statement commonly has the same merchant description
  repeated many times (the same supermarket every week, the same
  subscription every month). Ollama is called once per *unique* description
  in a batch, never once per transaction - classify_all() below does the
  dedup and fans the single result back out to every matching transaction.

classify_all() is the hybrid entry point main.py actually calls: the local
scikit-learn classifier (app/classifier_ml.py) gets first refusal on every
transaction - trained on past human-confirmed categorizations, no network
call at all - and only descriptions it isn't confident about (or, on a
fresh install, everything) go to Ollama, deduplicated as above.
"""

from __future__ import annotations

import json
import os

import httpx

from app import classifier_ml
from app.categories import CATEGORIES
from app.csv_io import Transaction

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

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
    "'Term Deposit'), use Transfer.\n"
    "- If the description mentions interest credited, use Interest.\n"
    "- If you are not reasonably confident, use Uncategorized rather than guessing.\n"
    "Respond with only the JSON object described by the schema."
)


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
        return category if category in CATEGORIES else "Uncategorized"
    except (httpx.HTTPError, KeyError, ValueError):
        return "Uncategorized"


async def classify_all(transactions: list[Transaction]) -> list[str]:
    """Local classifier first, Ollama only for what it won't commit to - and even
    then, one Ollama call per unique description in the batch, not per transaction."""
    results: list[str | None] = [None] * len(transactions)
    needs_ollama: list[int] = []

    for i, txn in enumerate(transactions):
        guess = classifier_ml.predict(txn.description)
        if guess is not None:
            category, _confidence = guess
            results[i] = category
        else:
            needs_ollama.append(i)

    if needs_ollama:
        unique_descriptions = {transactions[i].description for i in needs_ollama}
        classified: dict[str, str] = {}
        async with httpx.AsyncClient() as client:
            for description in unique_descriptions:
                classified[description] = await classify(client, description)
        for i in needs_ollama:
            results[i] = classified[transactions[i].description]

    return results  # type: ignore[return-value]
