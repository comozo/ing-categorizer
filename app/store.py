"""Persists the (description -> category) pairs a human has confirmed, and the
classifier trained on them. Never leaves the machine it runs on.

Review sessions (app/sessions.py) are the other piece of state, and the only
one holding dates and amounts - this module stays description-and-category
only.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from app.merchant import merchant_key

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TRAINING_DATA_PATH = DATA_DIR / "training_pairs.csv"
# v2: trained on merchant keys rather than raw descriptions. A new filename
# instead of an in-place overwrite, so a v1 model left on the volume is simply
# ignored and the first prediction retrains from training_pairs.csv.
MODEL_PATH = DATA_DIR / "classifier-v2.joblib"

_known_merchants: dict[str, str] | None = None


def append_pairs(pairs: list[tuple[str, str]]) -> None:
    global _known_merchants
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not TRAINING_DATA_PATH.exists()
    with open(TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["description", "category"])
        writer.writerows(pairs)
    _known_merchants = None


def load_pairs() -> list[tuple[str, str]]:
    if not TRAINING_DATA_PATH.exists():
        return []
    with open(TRAINING_DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return [(row[0], row[1]) for row in reader if len(row) == 2]


def relabel(updates: dict[str, str | None]) -> None:
    """Rewrites every stored pair of each merchant key in updates to the new
    category, or drops them all when the new category is None ("forget").

    Corrections rewrite rather than append so a merchant only ever has one
    category on disk: appending would fix memory (latest answer wins) but
    leave the old, wrong rows training the classifier.
    """
    global _known_merchants
    if not updates or not TRAINING_DATA_PATH.exists():
        return
    rows = []
    for description, category in load_pairs():
        key = merchant_key(description)
        if key in updates:
            if updates[key] is None:
                continue
            category = updates[key]
        rows.append((description, category))
    tmp = TRAINING_DATA_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "category"])
        writer.writerows(rows)
    os.replace(tmp, TRAINING_DATA_PATH)
    _known_merchants = None


def merchant_counts() -> dict[str, int]:
    """Merchant key -> how many confirmed transactions it has."""
    counts: dict[str, int] = {}
    for description, _ in load_pairs():
        key = merchant_key(description)
        counts[key] = counts.get(key, 0) + 1
    return counts


def known_merchants() -> dict[str, str]:
    """Merchant key -> the category it was most recently confirmed as.

    An exact match here skips both the classifier and Ollama: once you've told
    the app what a merchant is, it never asks again. Later confirmations win,
    so correcting a merchant on a later upload overrides the earlier answer.
    """
    global _known_merchants
    if _known_merchants is None:
        _known_merchants = {merchant_key(d): c for d, c in load_pairs()}
    return _known_merchants
