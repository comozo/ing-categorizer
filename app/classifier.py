"""Calls a local Ollama model to classify one transaction at a time.

Uses Ollama's structured-output mode (a JSON schema passed as `format`) so a
small model's output is constrained to exactly one of the fixed categories,
rather than parsed out of free text.

classify_all() is the hybrid entry point main.py actually calls: the local
scikit-learn classifier (app/classifier_ml.py) gets first refusal on every
transaction - trained on past human-confirmed categorizations, no network
call at all - and only what it isn't confident about (or, on a fresh
install, everything) falls through to Ollama here.
"""

from __future__ import annotations

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
    "You categorize a single bank transaction from an Australian ING account "
    "(an everyday transaction account, a term deposit, or a variable-rate savings "
    "account) into exactly one category from this fixed list:\n"
    f"{', '.join(CATEGORIES)}\n\n"
    "Rules:\n"
    "- If the description indicates money moving between the user's own ING "
    "accounts (e.g. contains 'transfer', 'TFR', 'Everyday', 'Savings Maximiser', "
    "'Term Deposit'), use Transfer.\n"
    "- If the description mentions interest credited, use Interest.\n"
    "- If you are not reasonably confident, use Uncategorized rather than guessing.\n"
    "Respond with only the JSON object described by the schema."
)


async def classify(client: httpx.AsyncClient, txn: Transaction) -> str:
    prompt = f"Date: {txn.date}\nDescription: {txn.description}\nAmount: {txn.amount}"
    try:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "format": _RESPONSE_SCHEMA,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        import json

        category = json.loads(content).get("category", "Uncategorized")
        return category if category in CATEGORIES else "Uncategorized"
    except (httpx.HTTPError, KeyError, ValueError):
        return "Uncategorized"


async def classify_all(transactions: list[Transaction]) -> list[str]:
    """Local classifier first, Ollama only for what it won't commit to."""
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
        async with httpx.AsyncClient() as client:
            for i in needs_ollama:
                results[i] = await classify(client, transactions[i])

    return results  # type: ignore[return-value]
