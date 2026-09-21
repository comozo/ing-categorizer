"""Persists the (description -> category) pairs a human has confirmed, and the
classifier trained on them. This is the only state ing-categorizer keeps -
no amounts, no dates, no account details, nothing beyond what's needed to
train a text classifier. Still never leaves the machine it runs on.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TRAINING_DATA_PATH = DATA_DIR / "training_pairs.csv"
MODEL_PATH = DATA_DIR / "classifier.joblib"


def append_pairs(pairs: list[tuple[str, str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not TRAINING_DATA_PATH.exists()
    with open(TRAINING_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["description", "category"])
        writer.writerows(pairs)


def load_pairs() -> list[tuple[str, str]]:
    if not TRAINING_DATA_PATH.exists():
        return []
    with open(TRAINING_DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return [(row[0], row[1]) for row in reader if len(row) == 2]
