"""Parsing OFX/QFX exports and re-serializing the result as CSV.

OFX has typed fields (a signed amount, a real date, a stable per-transaction
FITID) instead of CSV's guesswork over which column is which - this replaced
the earlier CSV-only parser for exactly that reason. Output stays CSV
regardless: it's the simplest common format for whichever finance app the
categorized result ends up imported into, and nothing about the input
format needs to leak into the output shape.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from ofxtools.Parser import OFXTree


class OfxFormatError(ValueError):
    """Raised when the uploaded file doesn't look like an OFX/QFX export we can read."""


@dataclass
class Transaction:
    row_id: int
    date: str
    description: str
    amount: str  # kept as a string for lossless round-tripping into the output CSV
    raw: dict[str, str]  # flattened OFX fields, for the final export


def parse_ofx(raw_bytes: bytes) -> list[Transaction]:
    tree = OFXTree()
    try:
        tree.parse(io.BytesIO(raw_bytes))
        ofx = tree.convert()
    except Exception as exc:
        raise OfxFormatError(
            "Couldn't read this as an OFX/QFX file. Make sure it's exported in that format, "
            "not CSV."
        ) from exc

    if not ofx.statements:
        raise OfxFormatError("This OFX file has no account statements in it.")

    transactions: list[Transaction] = []
    row_id = 0
    for statement in ofx.statements:
        for txn in statement.transactions:
            # NAME is the usual payee/merchant field; some banks put extra detail
            # in MEMO that NAME doesn't have. Combine when both are present.
            name = (txn.name or "").strip()
            memo = (txn.memo or "").strip()
            if name and memo and memo != name:
                description = f"{name} {memo}"
            else:
                description = name or memo

            date_str = txn.dtposted.date().isoformat() if txn.dtposted else ""
            amount_str = str(txn.trnamt) if txn.trnamt is not None else ""

            transactions.append(
                Transaction(
                    row_id=row_id,
                    date=date_str,
                    description=description,
                    amount=amount_str,
                    raw={
                        "Date": date_str,
                        "Description": description,
                        "Amount": amount_str,
                        "Type": txn.trntype or "",
                        "FITID": txn.fitid or "",
                    },
                )
            )
            row_id += 1

    if not transactions:
        raise OfxFormatError("This OFX file has an account statement but no transactions in it.")

    return transactions


def write_categorized_csv(transactions: list[Transaction], categories: list[str]) -> str:
    """Re-emit every flattened field plus a trailing Category column."""
    fieldnames = list(transactions[0].raw.keys()) + ["Category"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for txn, category in zip(transactions, categories):
        row = dict(txn.raw)
        row["Category"] = category
        writer.writerow(row)
    return buffer.getvalue()
