"""Scores categorization accuracy against your own confirmed answers.

Every merchant in training_pairs.csv becomes a test case, with the category
you confirmed as the right answer. Each one is asked leave-one-out: its own
past answer is hidden from memory, the few-shot examples and the local
classifier, so the score measures what happens with a merchant the app has
never seen - the only case where accuracy matters, since known merchants
are filled in from memory.

    python -m scripts.evaluate data/training_pairs.csv \\
        --sessions data/sessions --models qwen2.5:3b llama3:latest

--sessions is optional: review sessions are the only place amounts are
kept, so without them every merchant is asked without its money direction.
Runs against OLLAMA_URL (default http://localhost:11434).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from app import classifier, store  # noqa: E402
from app.categories import CATEGORIES  # noqa: E402
from app.merchant import merchant_key  # noqa: E402

# The prompt as it was before category hints, direction and few-shot examples,
# kept here only so every run can report the improvement over it.
_BASELINE_PROMPT = (
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
_BASELINE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": CATEGORIES}},
    "required": ["category"],
}


def load_cases(pairs_path: Path, sessions_dir: Path | None):
    """One case per merchant key: (key, a sample raw description, your
    category, direction or None)."""
    store.TRAINING_DATA_PATH = pairs_path
    labels: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, str] = {}
    for description, category in store.load_pairs():
        key = merchant_key(description)
        labels[key][category] += 1
        samples.setdefault(key, description)

    totals: dict[str, float] = defaultdict(float)
    if sessions_dir and sessions_dir.exists():
        for path in sessions_dir.glob("*.json"):
            for t in json.loads(path.read_text())["transactions"]:
                try:
                    totals[t["merchant"]] += float(t["amount"])
                except ValueError:
                    pass

    cases = []
    for key, counts in labels.items():
        category = counts.most_common(1)[0][0]
        if category == "Uncategorized":
            continue
        direction = ("in" if totals[key] >= 0 else "out") if key in totals else None
        cases.append((key, samples[key], category, direction))
    return cases


async def ask(client, model, messages, schema):
    resp = await client.post(
        f"{classifier.OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"]).get("category", "Uncategorized")


async def run_llm(cases, model, variant, concurrency):
    memory = {key: category for key, _, category, _ in cases}
    limit = asyncio.Semaphore(concurrency)

    async def one(case):
        key, sample, _, direction = case
        if variant == "baseline":
            messages = [
                {"role": "system", "content": _BASELINE_PROMPT},
                {"role": "user", "content": f"Description: {sample}"},
            ]
            schema = _BASELINE_SCHEMA
        else:
            others = {k: c for k, c in memory.items() if k != key}
            examples = classifier.similar_examples(key, others) if variant == "few-shot" else []
            messages = classifier.build_messages(key, direction, examples)
            schema = classifier._RESPONSE_SCHEMA
        async with limit:
            try:
                return await ask(client, model, messages, schema)
            except (httpx.HTTPError, KeyError, ValueError):
                return "Uncategorized"

    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*(one(c) for c in cases))


def run_local_classifier(cases, analyzer, ngram_range, threshold):
    """Leave-one-merchant-out: returns (prediction or None if below the
    confidence threshold) per case."""
    keys = [c[0] for c in cases]
    labels = [c[2] for c in cases]
    predictions = []
    for i in range(len(cases)):
        train_x = keys[:i] + keys[i + 1 :]
        train_y = labels[:i] + labels[i + 1 :]
        if len(set(train_y)) < 2:
            predictions.append(None)
            continue
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        model.fit(train_x, train_y)
        probabilities = model.predict_proba([keys[i]])[0]
        best = int(np.argmax(probabilities))
        predictions.append(model.classes_[best] if probabilities[best] >= threshold else None)
    return predictions


def report(name, cases, predictions, seconds=None):
    answered = [(c, p) for c, p in zip(cases, predictions) if p is not None]
    correct = sum(1 for c, p in answered if p == c[2])
    timing = f"  {seconds:5.1f}s" if seconds is not None else ""
    coverage = f"  answers {len(answered)}/{len(cases)}" if len(answered) != len(cases) else ""
    accuracy = correct / len(answered) if answered else 0
    print(f"{name:42} {correct:4}/{len(answered):<4} = {accuracy:6.1%}{coverage}{timing}")
    return [(c[0], c[2], p) for c, p in answered if p != c[2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--models", nargs="+", default=[classifier.OLLAMA_MODEL])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--show-errors", type=int, default=15)
    args = parser.parse_args()

    cases = load_cases(args.pairs, args.sessions)
    with_direction = sum(1 for c in cases if c[3] is not None)
    print(f"{len(cases)} merchants, {with_direction} with money direction\n")
    print("Category spread:", dict(Counter(c[2] for c in cases).most_common()), "\n")

    print("Local classifier (leave-one-merchant-out, threshold 0.65):")
    for label, analyzer, ngrams in (("word 1-2 grams (current)", "word", (1, 2)), ("char 3-5 grams", "char_wb", (3, 5))):
        report(f"  {label}", cases, run_local_classifier(cases, analyzer, ngrams, 0.65))

    errors = {}
    for model in args.models:
        print(f"\nOllama {model}:")
        for variant in ("baseline", "hints+direction", "few-shot"):
            start = time.monotonic()
            predictions = asyncio.run(run_llm(cases, model, variant, args.concurrency))
            errors[(model, variant)] = report(f"  {variant}", cases, predictions, time.monotonic() - start)

    for (model, variant), wrong in errors.items():
        if variant == "few-shot" and wrong:
            print(f"\nSample mistakes, {model} few-shot (merchant: yours -> model's):")
            for key, truth, predicted in wrong[: args.show_errors]:
                print(f"  {key}: {truth} -> {predicted}")


if __name__ == "__main__":
    main()
