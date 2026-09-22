"""ing-categorizer: upload an ING Australia OFX export, get it back with an
AI-suggested category per transaction, review/edit on a phone or desktop,
download the result. No transaction data ever leaves the cluster - the
local classifier and Ollama both run in-cluster, no external API calls.

The only thing persisted between requests is the (description -> category)
pairs you've confirmed on the review screen, and the classifier trained on
them - see app/store.py and app/classifier_ml.py. Everything else about a
request is forgotten once the response is sent."""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import classifier_ml, store
from app.categories import CATEGORIES
from app.classifier import classify_all
from app.ofx_io import OfxFormatError, Transaction, parse_ofx, write_categorized_csv

app = FastAPI(title="ing-categorizer")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"error": None})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    try:
        transactions = parse_ofx(raw)
    except OfxFormatError as exc:
        return templates.TemplateResponse(request, "index.html", {"error": str(exc)})

    classifications = await classify_all(transactions)

    rows = [
        {
            "id": txn.row_id,
            "date": txn.date,
            "description": txn.description,
            "amount": txn.amount,
            "category": c.category,
            "confidence": round(c.confidence * 100) if c.confidence is not None else None,
            "raw_json": json.dumps(txn.raw),
        }
        for txn, c in zip(transactions, classifications)
    ]

    # Triage, not filtering: everything is still here and still editable, this
    # just decides what a human needs to actually look closely at. The local
    # classifier only ever returns a result once it's already confident enough
    # (see CONFIDENCE_THRESHOLD in classifier_ml.py), so "came from the
    # classifier" already means "confident" - anything Ollama had to answer is,
    # by construction, either genuinely novel or something the classifier
    # wasn't sure about, which is exactly what's worth a second look.
    needs_review = [r for r, c in zip(rows, classifications) if c.source == "ollama"]
    confident = [r for r, c in zip(rows, classifications) if c.source == "classifier"]

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "needs_review": needs_review,
            "confident": confident,
            "categories": CATEGORIES,
            "count": len(rows),
        },
    )


@app.post("/download")
async def download(request: Request):
    form = await request.form()
    row_ids = sorted(
        {key.removeprefix("raw_") for key in form if key.startswith("raw_")},
        key=int,
    )

    transactions: list[Transaction] = []
    categories: list[str] = []
    training_pairs: list[tuple[str, str]] = []
    for row_id in row_ids:
        raw = json.loads(form[f"raw_{row_id}"])
        category = form.get(f"category_{row_id}", "Uncategorized")
        description = form.get(f"desc_{row_id}", "")

        transactions.append(Transaction(row_id=int(row_id), date="", description="", amount="", raw=raw))
        categories.append(category)
        if description and category != "Uncategorized":
            training_pairs.append((description, category))

    csv_text = write_categorized_csv(transactions, categories)

    # Every download is a batch of human-confirmed labels - feed them straight
    # back into the classifier's training set and retrain before responding.
    if training_pairs:
        store.append_pairs(training_pairs)
        classifier_ml.retrain()
        classifier_ml.invalidate_cache()

    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=categorized.csv"},
    )


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
