# ing-categorizer

Upload a CSV export from ING Australia online banking, get it back with a suggested spending category per transaction, review/edit on your phone, download the result. No transaction data ever leaves the machine it runs on — categorization runs against a local classifier and a local [Ollama](https://ollama.com) model, both in-cluster, never an external API.

Built for evaluating self-hosted personal finance apps (Securo, Actual Budget, Firefly III) without paying per-transaction for categorization, and without any bank data touching a third party.

## How it works

1. Upload a CSV export (Date / Description / Amount, or separate Debit/Credit columns — column names are matched loosely).
2. Each transaction is classified against a fixed, hand-picked category list (see `app/categories.py`) by a **hybrid** pipeline:
   - a small local scikit-learn classifier (TF-IDF + logistic regression), trained on categorizations you've confirmed on past uploads, gets first refusal — no network call at all;
   - anything it isn't confident about (or, on a fresh install, everything — there's nothing to train on yet) falls through to a local Ollama model, using Ollama's structured-output mode so it can only answer with one of the fixed categories.
3. Review and fix any miscategorized rows in a mobile-friendly table.
4. Download the same CSV with a `Category` column appended, ready to import into whichever finance app you're using.
5. Every row you confirmed on the review screen (except `Uncategorized`) is added to the classifier's training set and it retrains immediately — so the more you use this, the less it needs Ollama at all. This is the only state kept between requests: a small CSV of (description, category) pairs and the trained model, nothing else.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Base URL of the Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to classify with — keep this small, this is meant to run on CPU-only hardware |
| `DATA_DIR` | `/data` | Where the training pairs and trained classifier are persisted |
| `MIN_TRAINING_EXAMPLES` | `20` | Below this many confirmed examples, skip the classifier entirely and use Ollama |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this predicted probability, treat the classifier as unsure and fall back to Ollama |

## Local development

```bash
pip install -r requirements.txt
OLLAMA_URL=http://localhost:11434 uvicorn app.main:app --reload
```

Requires a running Ollama instance with the configured model pulled (`ollama pull qwen2.5:3b`).

## Deployment

Deployed via Flux in [comozo/homelab-cluster](https://github.com/comozo/homelab-cluster) under `kubernetes/apps/ing-categorizer/`, alongside a matching in-cluster Ollama deployment under `kubernetes/apps/ollama/`.
