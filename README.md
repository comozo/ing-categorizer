# ing-categorizer

Upload a CSV export from ING Australia online banking, get it back with a suggested spending category per transaction, review/edit on your phone, download the result. Nothing is stored between requests and no transaction data ever leaves the machine it runs on — categorization calls a local [Ollama](https://ollama.com) model over the cluster network, never an external API.

Built for evaluating self-hosted personal finance apps (Securo, Actual Budget, Firefly III) without paying per-transaction for categorization, and without any bank data touching a third party.

## How it works

1. Upload a CSV export (Date / Description / Amount, or separate Debit/Credit columns — column names are matched loosely).
2. Each transaction is sent to a local Ollama model with a fixed, hand-picked category list (see `app/categories.py`), using Ollama's structured-output mode so the model can only answer with one of those categories.
3. Review and fix any miscategorized rows in a mobile-friendly table.
4. Download the same CSV with a `Category` column appended, ready to import into whichever finance app you're using.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Base URL of the Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to classify with — keep this small, this is meant to run on CPU-only hardware |

## Local development

```bash
pip install -r requirements.txt
OLLAMA_URL=http://localhost:11434 uvicorn app.main:app --reload
```

Requires a running Ollama instance with the configured model pulled (`ollama pull qwen2.5:3b`).

## Deployment

Deployed via Flux in [comozo/homelab-cluster](https://github.com/comozo/homelab-cluster) under `kubernetes/apps/ing-categorizer/`, alongside a matching in-cluster Ollama deployment under `kubernetes/apps/ollama/`.
