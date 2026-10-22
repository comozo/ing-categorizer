# ing-categorizer

Upload an OFX/QFX export from ING Australia online banking, get back a CSV with a suggested spending category per transaction, review/edit on your phone, download the result. No transaction data ever leaves the LAN — categorization runs against a local classifier plus an [Ollama](https://ollama.com) server on the network (not a hosted/third-party API), controlled entirely by `OLLAMA_URL`.

Built for evaluating self-hosted personal finance apps (Securo, Actual Budget, Firefly III) without paying per-transaction for categorization, and without any bank data touching a third party.

## How it works

1. Upload an OFX or QFX export. Unlike CSV, OFX has typed fields — a signed amount, a real date, a stable per-transaction `FITID` — so there's no column-name guesswork the way there was with CSV (that parser existed briefly; see git history if you need a CSV-shaped starting point again). ING Australia's own export omits several fields that `ofxtools` otherwise treats as hard-required — `<BANKTRANLIST>`'s `DTSTART`/`DTEND` (the statement period), `<STMTRS>`'s `BANKACCTFROM` (which bank/account the statement is for), and `<STMTRS>`'s `LEDGERBAL` (balance amount/as-of date) — worked around by patching in harmless placeholders before parsing whenever a field is genuinely missing (see `app/ofx_io.py` if a future export starts omitting something else — the fix is a one-line addition to a list there). OFX 1.x carries no schema in the file itself to check exports against up front, so these were found by cross-checking a real export against `ofxtools`' own model spec for every aggregate a bank statement message touches, rather than one at a time as each became a production crash.
2. Transactions are grouped by **merchant** before anything else happens. ING appends a receipt number, date and card number to every description, so the same shop never repeats verbatim; `app/merchant.py` strips that tail (and store numbers) so "WOOLWORTHS 1234 SYDNEY - Visa Purchase - Receipt …" and "WOOLWORTHS 5678 SYDNEY - …" both become `WOOLWORTHS SYDNEY`. Everything downstream (classification, review, learning) works per merchant, not per transaction.
3. Each merchant is classified against a fixed, grouped category list (see `app/categories.py` — six groups: Housing, Food & Dining, Transport, Lifestyle, Other, Income) by the first source that can answer:
   - **memory** — a merchant you've confirmed on a past upload is filled in automatically and never asked about again;
   - a small local scikit-learn classifier (TF-IDF + logistic regression on merchant keys), trained on everything you've confirmed — no network call;
   - anything it isn't confident about (or, on a fresh install, everything) falls through to Ollama (`OLLAMA_URL`), using its structured-output mode so it can only answer with one of the fixed categories. One call per unknown merchant, at most `OLLAMA_CONCURRENCY` at a time. The prompt defines every category with typical Australian merchants (the `hint`s in `app/categories.py`), and includes up to `FEW_SHOT_EXAMPLES` of your own past answers for the most similar-looking merchants, so it follows your conventions rather than guessing.
   - Ollama only ever sees the **cleaned merchant name and whether money went out or came in** — never the date, the amount, or which account a transaction came from.
4. The review screen shows **one card per merchant** — the classifier's confident picks first, then whatever needed Ollama, biggest merchants first within each. One tap on "Looks right" confirms the AI pick for every transaction of that merchant; the chip grid picks something else; "Accept every remaining AI pick" finishes the deck in one go. On a keyboard, Enter accepts, S skips, ← goes back. Remembered merchants skip the deck and are listed on the final card, where any of them can still be changed. The plain `<select>` per merchant is the no-JS fallback (see `app/static/game.js`).
5. **Categorizing runs in the background with live progress** (`app/jobs.py`). The upload is saved straight away and the review URL shows a progress screen: merchants done out of the total, elapsed time and an estimate of what's left, which merchants Ollama is working on right now with how long each has taken, and a feed of the latest answers with where each came from. If Ollama can't be reached, the screen says so, and the review shows how many merchants were left Uncategorized. You can close the page; a restart mid-run carries on from the merchants still without an answer the next time the review is opened.
6. **Every answer is checkpointed** to a review session on the data volume as you make it (`app/sessions.py`). Closing the tab, losing the connection or a pod restart loses nothing: the home page lists every review, and "Resume" reopens it at the first unanswered merchant with XP and streak intact. Sessions hold the parsed statement (dates and amounts included, needed to build the CSV) and are deleted after `SESSION_RETENTION_DAYS`, or immediately with "Discard".
7. Download the CSV with a `Category` column appended, ready to import into whichever finance app you're using. Every confirmed merchant (except `Uncategorized`) is added to the training set and the classifier retrains — so each upload leaves fewer merchants to review on the next one. Training data is just (description, category) pairs.

8. **Known merchants** (`/merchants`, linked from the home page) lists every remembered merchant — the ones filled in without asking. Change a category to rewrite every past answer for that merchant (and retrain), or forget it to be asked again next time. "Check with AI" asks Ollama about each one in the background, without showing it your past answers, and flags where it disagrees: the quick way to find AI picks that were accepted without a close look and have been repeating since.

9. **Check against a statement** (`/reconcile`, linked from the home page and each review) answers "does this file add up?" with no finance app involved: statement opening balance + every transaction dated inside the period must equal the closing balance. Give it the categorizer's CSV, the original OFX, a saved review, or a bank CSV with Credit/Debit columns. When it's off, it lists transactions dated outside the period, repeated transaction IDs, look-alike rows, transactions that are exactly the gap (a duplicate?) or half of it (wrong sign?), whether money in or money out is the side that's wrong (when you enter the statement's totals), and a month-by-month running balance to compare with ING's monthly statements. With both the OFX and the CSV it also confirms, transaction by transaction, that the CSV matches the export. `python -m scripts.reconcile` does the same from the command line.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Base URL of the Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to classify with — keep this small, this is meant to run on CPU-only hardware |
| `DATA_DIR` | `/data` | Where the training pairs, trained classifier and review sessions are persisted |
| `MIN_TRAINING_EXAMPLES` | `20` | Below this many confirmed examples, skip the classifier entirely and use Ollama |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this predicted probability, treat the classifier as unsure and fall back to Ollama |
| `OLLAMA_CONCURRENCY` | `4` | Maximum Ollama requests in flight at once; more just queue inside Ollama until they time out |
| `FEW_SHOT_EXAMPLES` | `30` | How many of your past answers (most similar merchants first) go into each Ollama prompt |
| `SESSION_RETENTION_DAYS` | `30` | Review sessions untouched for longer than this are deleted |

## Local development

```bash
pip install -r requirements.txt
OLLAMA_URL=http://localhost:11434 uvicorn app.main:app --reload
```

Requires a running Ollama instance with the configured model pulled (`ollama pull qwen2.5:3b`).

## Measuring accuracy

`scripts/evaluate.py` replays every merchant you've confirmed through the local classifier and Ollama, leave-one-out (its own answer hidden), and scores the result against your answer — use it before changing the prompt, the category hints or the model:

```bash
kubectl -n ing-categorizer exec deploy/ing-categorizer -- cat /data/training_pairs.csv > /tmp/pairs.csv
python -m scripts.evaluate /tmp/pairs.csv --models qwen2.5:3b qwen2.5:7b
```

Pass `--sessions <dir>` with a copy of `/data/sessions` to include money direction, which only sessions record.

## Deployment

Deployed via Flux in [comozo/homelab-cluster](https://github.com/comozo/homelab-cluster) under `kubernetes/apps/ing-categorizer/`. Ollama itself is **not** deployed in-cluster (that was tried first, see the repo's git history) — `OLLAMA_URL` currently points at a plain `ollama serve` running on the operator's own laptop, reachable over the LAN. That's a DHCP-leased IP, not a stable address; see the comment on `OLLAMA_URL` in `kubernetes/apps/ing-categorizer/ing-categorizer/app/helmrelease.yaml` before assuming a classification failure is a bug in this app rather than a stale IP.
