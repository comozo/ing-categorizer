"""ing-categorizer: upload an ING Australia CSV export, get it back with an
AI-suggested category per transaction, review/edit on a phone or desktop,
download the result. Stateless - nothing is stored server-side between
requests, and no transaction data ever leaves the cluster (categorization
runs against a local Ollama model, no external API calls)."""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.categories import CATEGORIES
from app.classifier import classify_all
from app.csv_io import CsvFormatError, Transaction, parse_csv, write_categorized_csv

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
        transactions = parse_csv(raw)
    except CsvFormatError as exc:
        return templates.TemplateResponse(request, "index.html", {"error": str(exc)})

    categories = await classify_all(transactions)

    rows = [
        {
            "id": txn.row_id,
            "date": txn.date,
            "description": txn.description,
            "amount": txn.amount,
            "category": category,
            "raw_json": json.dumps(txn.raw),
        }
        for txn, category in zip(transactions, categories)
    ]

    return templates.TemplateResponse(
        request,
        "review.html",
        {"rows": rows, "categories": CATEGORIES, "count": len(rows)},
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
    for row_id in row_ids:
        raw = json.loads(form[f"raw_{row_id}"])
        transactions.append(Transaction(row_id=int(row_id), date="", description="", amount="", raw=raw))
        categories.append(form.get(f"category_{row_id}", "Uncategorized"))

    csv_text = write_categorized_csv(transactions, categories)
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=categorized.csv"},
    )


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
