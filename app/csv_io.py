"""Parsing and re-serializing ING Australia CSV exports.

ING's own export format isn't pinned down by a verified sample yet, so
column detection is deliberately loose (matches on header keywords rather
than exact names) and fails with a clear, user-facing message rather than
a stack trace when it can't confidently find what it needs.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass


class CsvFormatError(ValueError):
    """Raised when the uploaded file doesn't look like a bank export we can read."""


@dataclass
class Transaction:
    row_id: int
    date: str
    description: str
    amount: str  # kept as the original string for lossless round-tripping
    raw: dict[str, str]  # every original column, untouched, for the final export


def _find_column(fieldnames: list[str], *keywords: str) -> str | None:
    for name in fieldnames:
        lowered = name.strip().lower()
        if any(keyword in lowered for keyword in keywords):
            return name
    return None


def parse_csv(raw_bytes: bytes) -> list[Transaction]:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvFormatError("Could not read the file as text (unexpected encoding).") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise CsvFormatError("The file doesn't have a header row Securo can recognise.")

    date_col = _find_column(reader.fieldnames, "date")
    desc_col = _find_column(reader.fieldnames, "description", "narrative", "details", "memo")
    debit_col = _find_column(reader.fieldnames, "debit")
    credit_col = _find_column(reader.fieldnames, "credit")
    # A generic "Amount" column only counts if it's not actually "Debit Amount" /
    # "Credit Amount" - both contain the substring "amount" too, so debit/credit
    # must be detected first and excluded here, or this always matches "Debit Amount".
    non_split_fields = [f for f in reader.fieldnames if f not in (debit_col, credit_col)]
    amount_col = _find_column(non_split_fields, "amount")

    if not date_col or not desc_col:
        raise CsvFormatError(
            "Couldn't find a Date and Description column in this file. "
            f"Columns found: {', '.join(reader.fieldnames)}"
        )
    if not amount_col and not (debit_col or credit_col):
        raise CsvFormatError(
            "Couldn't find an Amount column (or separate Debit/Credit columns) in this file. "
            f"Columns found: {', '.join(reader.fieldnames)}"
        )

    transactions: list[Transaction] = []
    for i, row in enumerate(reader):
        if amount_col:
            amount = row.get(amount_col, "").strip()
        else:
            debit = (row.get(debit_col, "") or "").strip() if debit_col else ""
            credit = (row.get(credit_col, "") or "").strip() if credit_col else ""
            amount = f"-{debit}" if debit else credit

        transactions.append(
            Transaction(
                row_id=i,
                date=row.get(date_col, "").strip(),
                description=row.get(desc_col, "").strip(),
                amount=amount,
                raw=row,
            )
        )

    if not transactions:
        raise CsvFormatError("This file has a header row but no transactions under it.")

    return transactions


def write_categorized_csv(transactions: list[Transaction], categories: list[str]) -> str:
    """Re-emit every original column plus a trailing Category column."""
    fieldnames = list(transactions[0].raw.keys()) + ["Category"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for txn, category in zip(transactions, categories):
        row = dict(txn.raw)
        row["Category"] = category
        writer.writerow(row)
    return buffer.getvalue()
