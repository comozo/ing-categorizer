"""ing-categorizer: upload an ING Australia OFX export, get it back with an
AI-suggested category per transaction, review/edit on a phone or desktop,
download the result. No transaction data ever leaves the LAN - the local
classifier runs in-process and Ollama only ever sees one description per
unknown merchant.

Two kinds of state are persisted, both on DATA_DIR:

- the (description -> category) pairs you've confirmed, and the classifier
  trained on them - see app/store.py and app/classifier_ml.py;
- review sessions, the checkpoint that makes a half-finished review
  resumable - see app/sessions.py.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import classifier_ml, sessions, store
from app.categories import CATEGORIES, CATEGORY_GROUPS
from app.classifier import classify_merchants
from app.merchant import merchant_key
from app.ofx_io import OfxFormatError, Transaction, parse_ofx, write_categorized_csv

# Plain stdout logging with timestamps - this is a container, `kubectl logs` is the
# only place these are ever read from, no need for anything fancier than that.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ing-categorizer")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Review order: what the classifier is confident about first (quick confirms
# that build a streak), then what only Ollama could answer (worth a closer
# look). Within each, the merchants with the most transactions come first, so
# the early answers cover the most rows.
_SOURCE_ORDER = {"classifier": 0, "ollama": 1}


def _index(request: Request, error: str | None = None, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"error": error, "sessions": sessions.summaries(), "retention_days": sessions.RETENTION_DAYS},
        status_code=status_code,
    )


def _total(amounts: list[str]) -> str:
    total = Decimal(0)
    for amount in amounts:
        try:
            total += Decimal(amount)
        except InvalidOperation:
            pass
    return f"{total:.2f}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _index(request)


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    logger.info("Upload received: filename=%r, size=%d bytes", file.filename, len(raw))
    try:
        transactions = parse_ofx(raw)
    except OfxFormatError as exc:
        return _index(request, error=str(exc), status_code=400)

    by_merchant: dict[str, list[Transaction]] = {}
    for txn in transactions:
        by_merchant.setdefault(merchant_key(txn.description), []).append(txn)

    classifications = await classify_merchants(
        {key: txns[0].description for key, txns in by_merchant.items()}
    )
    counts = {source: 0 for source in ("memory", "classifier", "ollama")}
    for c in classifications.values():
        counts[c.source] += 1
    logger.info(
        "Parsed %d transactions into %d merchants: %d remembered, %d from the local "
        "classifier, %d needed Ollama",
        len(transactions),
        len(by_merchant),
        counts["memory"],
        counts["classifier"],
        counts["ollama"],
    )

    merchants = []
    for key, txns in by_merchant.items():
        c = classifications[key]
        dates = sorted(t.date for t in txns if t.date)
        merchants.append(
            {
                "key": key,
                "sample": txns[0].description,
                "count": len(txns),
                "total": _total([t.amount for t in txns]),
                "first_date": dates[0] if dates else "",
                "last_date": dates[-1] if dates else "",
                "suggested": c.category,
                "source": c.source,
                "confidence": round(c.confidence * 100) if c.confidence is not None else None,
                # Remembered merchants count as already answered: that's the
                # whole point of remembering them. Everything else waits for you.
                "category": c.category if c.source == "memory" else None,
            }
        )
    merchants.sort(key=lambda m: (_SOURCE_ORDER.get(m["source"], -1), -m["count"]))

    session = sessions.create(
        filename=file.filename or "export.ofx",
        transactions=[
            {
                "row_id": t.row_id,
                "date": t.date,
                "description": t.description,
                "amount": t.amount,
                "raw": t.raw,
                "merchant": merchant_key(t.description),
            }
            for t in transactions
        ],
        merchants=merchants,
    )
    # Redirect rather than render: the review page now has its own URL, so a
    # refresh or a reopened tab resumes the session instead of re-uploading.
    return RedirectResponse(f"/review/{session['id']}", status_code=303)


@app.get("/review/{session_id}", response_class=HTMLResponse)
def review(request: Request, session_id: str):
    session = sessions.load(session_id)
    if session is None:
        return _index(request, error="That review no longer exists. Upload the export again.", status_code=404)

    indexed = list(enumerate(session["merchants"]))
    deck = [(i, m) for i, m in indexed if m["source"] != "memory"]
    remembered = [(i, m) for i, m in indexed if m["source"] == "memory"]

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "session": session,
            "deck": deck,
            "remembered": remembered,
            "remembered_txn_count": sum(m["count"] for _, m in remembered),
            "categories": CATEGORIES,
            "category_groups": CATEGORY_GROUPS,
            "transaction_count": len(session["transactions"]),
        },
    )


@app.post("/review/{session_id}/answers")
async def save_answers(session_id: str, request: Request):
    """Checkpoint endpoint the review screen calls on every pick:
    {"answers": [{"index": 3, "category": "Groceries"}, ...], "stats": {...}}."""
    payload = await request.json()
    answers = payload.get("answers", [])
    stats = payload.get("stats")

    def apply(session):
        merchants = session["merchants"]
        for answer in answers:
            index, category = answer.get("index"), answer.get("category")
            if isinstance(index, int) and 0 <= index < len(merchants) and category in CATEGORIES:
                merchants[index]["category"] = category
        if isinstance(stats, dict):
            for field in ("xp", "streak", "best_streak"):
                if isinstance(stats.get(field), int):
                    session["stats"][field] = stats[field]

    if sessions.update(session_id, apply) is None:
        return JSONResponse({"ok": False, "error": "session not found"}, status_code=404)
    return {"ok": True}


@app.post("/review/{session_id}/download")
async def download(session_id: str, request: Request):
    form = await request.form()

    def finish(session):
        # The form carries every merchant's <select>: with JS these already
        # match what was checkpointed; without JS this is the only place the
        # answers arrive. Either way, the submitted form is the final word.
        for i, merchant in enumerate(session["merchants"]):
            category = form.get(f"category_{i}")
            if category in CATEGORIES:
                merchant["category"] = category
            elif merchant["category"] is None:
                merchant["category"] = merchant["suggested"]
        session["completed_at"] = sessions.now()

    session = sessions.update(session_id, finish)
    if session is None:
        return PlainTextResponse("That review no longer exists.", status_code=404)

    by_key = {m["key"]: m for m in session["merchants"]}
    transactions = [
        Transaction(row_id=t["row_id"], date=t["date"], description=t["description"], amount=t["amount"], raw=t["raw"])
        for t in session["transactions"]
    ]
    categories = [by_key[t["merchant"]]["category"] for t in session["transactions"]]
    csv_text = write_categorized_csv(transactions, categories)

    # Feed confirmed labels back into training - only the merchants whose
    # answer changed since the last download of this session, so downloading
    # again doesn't duplicate pairs.
    trained = session["trained"]
    changed = {
        m["key"]: m["category"]
        for m in session["merchants"]
        if m["category"] != "Uncategorized" and trained.get(m["key"]) != m["category"]
    }
    if changed:
        store.append_pairs(
            [(t["description"], changed[t["merchant"]]) for t in session["transactions"] if t["merchant"] in changed]
        )
        classifier_ml.retrain()
        classifier_ml.invalidate_cache()
        sessions.update(session_id, lambda s: s["trained"].update(changed))

    stem = re.sub(r"[^\w.-]", "_", session["filename"].rsplit(".", 1)[0])
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}-categorized.csv"'},
    )


@app.post("/review/{session_id}/discard")
def discard(session_id: str):
    sessions.delete(session_id)
    return RedirectResponse("/", status_code=303)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
