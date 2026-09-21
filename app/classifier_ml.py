"""A small local classifier trained on human-confirmed categorizations, so
that most transactions can be classified without a network call to Ollama
at all. Bootstrapped by Ollama's own output on the first few uploads (see
app/classifier.py) - this is the "avoid outbound request" half of that
LLM-preload-then-distill pattern.
"""

from __future__ import annotations

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.categories import CATEGORIES
from app.store import MODEL_PATH, load_pairs

# Below this many confirmed examples, don't trust the classifier at all - too
# little data for TF-IDF + logistic regression to have learned anything
# meaningful, so everything still goes to Ollama.
MIN_TRAINING_EXAMPLES = int(os.environ.get("MIN_TRAINING_EXAMPLES", "20"))

# Below this predicted probability, treat the classifier as "not sure" and
# fall back to Ollama rather than guess.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def retrain() -> Pipeline | None:
    """Retrains from every confirmed pair on disk and persists the result.

    Called once per download - the training set is small enough (a personal
    account's transaction history) that refitting from scratch each time is
    simpler and fast enough to not need incremental/online learning.
    """
    pairs = load_pairs()
    if len(pairs) < MIN_TRAINING_EXAMPLES:
        return None

    descriptions, categories = zip(*pairs)
    pipeline = _build_pipeline()
    pipeline.fit(list(descriptions), list(categories))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


_cached_model: Pipeline | None = None
_cached_model_loaded = False


def _get_model() -> Pipeline | None:
    global _cached_model, _cached_model_loaded
    if not _cached_model_loaded:
        _cached_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
        _cached_model_loaded = True
    return _cached_model


def invalidate_cache() -> None:
    """Called after retrain() so the next prediction picks up the new model."""
    global _cached_model_loaded
    _cached_model_loaded = False


def predict(description: str) -> tuple[str, float] | None:
    """Returns (category, confidence) if the classifier is confident enough,
    or None if it should fall back to Ollama (no model yet, or unsure)."""
    model = _get_model()
    if model is None:
        return None

    probabilities = model.predict_proba([description])[0]
    best_idx = probabilities.argmax()
    confidence = float(probabilities[best_idx])
    category = model.classes_[best_idx]

    if confidence < CONFIDENCE_THRESHOLD or category not in CATEGORIES:
        return None
    return category, confidence
