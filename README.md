# ing-categorizer

Upload an OFX/QFX export from ING Australia online banking, get back a CSV with a suggested spending category per transaction, review/edit on your phone, download the result. No transaction data ever leaves the LAN — categorization runs against a local classifier plus an [Ollama](https://ollama.com) server on the network (not a hosted/third-party API), controlled entirely by `OLLAMA_URL`.

Built for evaluating self-hosted personal finance apps (Securo, Actual Budget, Firefly III) without paying per-transaction for categorization, and without any bank data touching a third party.

## How it works

1. Upload an OFX or QFX export. Unlike CSV, OFX has typed fields — a signed amount, a real date, a stable per-transaction `FITID` — so there's no column-name guesswork the way there was with CSV (that parser existed briefly; see git history if you need a CSV-shaped starting point again).
2. Each transaction is classified against a fixed, hand-picked category list (see `app/categories.py`) by a **hybrid** pipeline:
   - a small local scikit-learn classifier (TF-IDF + logistic regression), trained on categorizations you've confirmed on past uploads, gets first refusal — no network call at all;
   - anything it isn't confident about (or, on a fresh install, everything — there's nothing to train on yet) falls through to Ollama (`OLLAMA_URL`), using its structured-output mode so it can only answer with one of the fixed categories.
   - Ollama only ever sees the **description text** — never the date, the amount, or which account a transaction came from. And it's only called once per **unique** description in a batch, not once per transaction: five transactions at the same supermarket become one Ollama call, not five, with the result applied to all five.
3. Review screen is triaged, not one flat table: rows the classifier was confident about (shown with its confidence %) are grouped into a collapsed "Already confident" section, and only what actually needed Ollama sits open under "Needs your review" by default. Nothing is hidden or auto-applied — every row is still there and still editable — it's just sorted by how much attention it's worth.
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

Deployed via Flux in [comozo/homelab-cluster](https://github.com/comozo/homelab-cluster) under `kubernetes/apps/ing-categorizer/`. Ollama itself is **not** deployed in-cluster (that was tried first, see the repo's git history) — `OLLAMA_URL` currently points at a plain `ollama serve` running on the operator's own laptop, reachable over the LAN. That's a DHCP-leased IP, not a stable address; see the comment on `OLLAMA_URL` in `kubernetes/apps/ing-categorizer/ing-categorizer/app/helmrelease.yaml` before assuming a classification failure is a bug in this app rather than a stale IP.
