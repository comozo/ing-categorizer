"""Checks an export against a bank statement, with no finance app involved.

Opening balance + every transaction dated inside the statement period must
equal the statement's closing balance. When it doesn't, the report narrows
down why: transactions dated outside the period, repeated transaction IDs,
identical-looking rows, a single transaction (or a duplicated one) the size
of the gap, amounts with the wrong sign, or whether credits or debits are
the side that's off. A month-by-month running balance pins the problem to
the first month that disagrees with ING's monthly statements.

Sources, any combination: the original OFX/QFX export, a CSV (this app's
output, or a bank CSV with Credit/Debit columns), or a saved review session.
Given both an OFX and a CSV, it also checks the CSV against the OFX
transaction by transaction, which is what proves the categorizer neither
dropped nor changed anything.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.ofx_io import parse_ofx, transaction_ids

CENT = Decimal("0.01")


@dataclass(frozen=True)
class Row:
    date: date
    description: str
    amount: Decimal  # signed: negative is money out
    fitid: str  # ING's own ID - not unique, see ofx_io.transaction_ids()
    uid: str  # unique per transaction; what rows are matched on


class SourceError(ValueError):
    """A file that can't be read as transactions. The message is shown."""


def parse_amount(text: str) -> Decimal | None:
    cleaned = (text or "").strip().replace("$", "").replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    try:
        value = Decimal(cleaned.strip("()"))
    except InvalidOperation:
        return None
    return -value if negative else value


def _parse_date(text: str) -> date:
    text = (text or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise SourceError(f"Couldn't read the date {text!r}. Expected 2025-07-01 or 01/07/2025.")


def _rows_from_raws(raws: list[dict[str, str]]) -> list[Row]:
    return [
        Row(date.fromisoformat(raw["Date"]), raw["Description"], Decimal(raw["Amount"]), raw.get("FITID", ""), uid)
        for raw, uid in zip(raws, transaction_ids(raws))
    ]


def rows_from_ofx(raw: bytes) -> list[Row]:
    return _rows_from_raws([t.raw for t in parse_ofx(raw)])


def rows_from_csv(raw: bytes) -> list[Row]:
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}

    def col(*names: str) -> str | None:
        return next((headers[n] for n in names if n in headers), None)

    date_col = col("date", "transaction date")
    desc_col = col("description", "details", "memo")
    amount_col = col("amount")
    credit_col, debit_col = col("credit"), col("debit")
    fitid_col = col("fitid", "id")
    uid_col = col("transaction id")
    if date_col is None or (amount_col is None and credit_col is None and debit_col is None):
        raise SourceError("The CSV needs a Date column and either an Amount column or Credit/Debit columns.")

    rows = []
    raw_amounts = []
    for line, record in enumerate(reader, start=2):
        if amount_col is not None:
            raw_amounts.append(record.get(amount_col, "").strip())
            amount = parse_amount(record.get(amount_col, ""))
        else:
            # Bank CSVs with split columns: debits may be written positive or
            # negative, so take their size and make them money out.
            credit = parse_amount(record.get(credit_col, "")) if credit_col else None
            debit = parse_amount(record.get(debit_col, "")) if debit_col else None
            amount = (credit or Decimal(0)) - abs(debit or Decimal(0)) if (credit or debit) else None
        if amount is None:
            raise SourceError(f"Line {line}: no amount could be read.")
        rows.append(
            Row(
                _parse_date(record.get(date_col, "")),
                (record.get(desc_col, "") if desc_col else "").strip(),
                amount,
                (record.get(fitid_col, "") if fitid_col else "").strip(),
                (record.get(uid_col, "") if uid_col else "").strip(),
            )
        )
    # A CSV without the Transaction ID column (made before it existed) gets
    # the same IDs the categorizer would give it now, from its own fields.
    if uid_col is None and amount_col is not None and desc_col is not None:
        raws = [{"Date": r.date.isoformat(), "Description": r.description, "Amount": a, "FITID": r.fitid} for r, a in zip(rows, raw_amounts)]
        rows = [Row(r.date, r.description, r.amount, r.fitid, uid) for r, uid in zip(rows, transaction_ids(raws))]
    return rows


def rows_from_session(session: dict[str, Any]) -> list[Row]:
    return _rows_from_raws([t["raw"] for t in session["transactions"]])


def _money(value: Decimal) -> str:
    return f"{value.quantize(CENT):,.2f}"


def _row_dict(row: Row) -> dict[str, str]:
    return {"date": row.date.isoformat(), "description": row.description, "amount": _money(row.amount), "fitid": row.fitid, "uid": row.uid}


def reconcile(
    rows: list[Row],
    start: date,
    end: date,
    opening: Decimal,
    closing: Decimal,
    total_credits: Decimal | None = None,
    total_debits: Decimal | None = None,
) -> dict[str, Any]:
    inside = [r for r in rows if start <= r.date <= end]
    outside = [r for r in rows if not start <= r.date <= end]
    credits = sum((r.amount for r in inside if r.amount > 0), Decimal(0))
    debits = sum((r.amount for r in inside if r.amount < 0), Decimal(0))
    net = credits + debits
    computed = opening + net
    difference = computed - closing
    findings: list[str] = []

    if difference == 0:
        findings.append("Opening balance plus the transactions in the period lands exactly on the closing balance.")
    else:
        size = abs(difference)
        findings.append(
            f"The transactions add up to {_money(computed)}, which is {_money(size)} "
            f"{'more' if difference > 0 else 'less'} than the statement's closing balance."
        )
        # Signed matches only, so every hint is a cause that actually fits:
        # removing a row changes the total by its amount, flipping its sign
        # by twice its amount.
        extra = [r for r in inside if r.amount == difference]
        if extra:
            findings.append(
                f"{len(extra)} transaction(s) of exactly {_money(difference)} are listed below. If one is in the file "
                "by mistake (a duplicate, say), removing it closes the gap."
            )
        flipped = [r for r in inside if r.amount * 2 == difference]
        if flipped:
            findings.append(
                f"{len(flipped)} transaction(s) of exactly {_money(difference / 2)}, half the gap, are listed below. "
                "One of them recorded with the wrong sign would explain it."
            )
        findings.append(
            f"Or the file is missing {'money out' if difference > 0 else 'money in'} totalling {_money(size)}: "
            "compare the month-by-month table with ING's monthly statements to find where."
        )
        if outside:
            outside_net = sum((r.amount for r in outside), Decimal(0))
            if opening + net + outside_net == closing:
                findings.append(
                    f"Counting the {len(outside)} transaction(s) dated outside the period closes the gap exactly: the "
                    "statement period and the export's dates don't line up."
                )
        if opening - net == closing:
            findings.append("Every amount has the wrong sign: with the signs flipped, the closing balance matches.")
        if total_credits is not None and credits != total_credits:
            findings.append(
                f"Money in is off: the file has {_money(credits)}, the statement {_money(total_credits)} "
                f"(difference {_money(credits - total_credits)})."
            )
        if total_debits is not None and abs(debits) != abs(total_debits):
            findings.append(
                f"Money out is off: the file has {_money(abs(debits))}, the statement {_money(abs(total_debits))} "
                f"(difference {_money(abs(debits) - abs(total_debits))})."
            )

    if total_credits is not None and total_debits is not None and difference != 0:
        if credits == total_credits and abs(debits) == abs(total_debits):
            findings.append(
                "Money in and money out both match the statement, so the transactions are right and the opening balance is the problem."
            )

    # ING reuses one FITID for an international purchase, its fee and the
    # fee's refund, so a shared FITID is normal there - but it's exactly what
    # makes a finance app that de-duplicates on FITID drop transactions.
    fitid_counts = Counter(r.fitid for r in rows if r.fitid)
    repeated_ids = [r for r in rows if r.fitid and fitid_counts[r.fitid] > 1]
    if repeated_ids:
        findings.append(
            f"{len(repeated_ids)} rows share ING's FITID with another row. ING does this for an international purchase, "
            "its fee and the fee refund, so they're usually all genuine. But a finance app that skips duplicates by "
            "FITID keeps only one of each group: import with the Transaction ID column mapped instead."
        )
    uid_counts = Counter(r.uid for r in rows if r.uid)
    true_duplicates = [r for r in rows if r.uid and uid_counts[r.uid] > 1]
    if true_duplicates:
        findings.append(
            f"{len(true_duplicates)} rows share a Transaction ID: same ID, date, amount and description. "
            "These are duplicates."
        )
    look_alike_counts = Counter((r.date, r.amount, r.description) for r in rows)
    look_alikes = [r for r in rows if look_alike_counts[(r.date, r.amount, r.description)] > 1 and not (r.uid and uid_counts[r.uid] > 1)]
    if look_alikes:
        findings.append(
            f"{len(look_alikes)} rows have the same date, amount and description as another row but different IDs. "
            "Often genuine (two identical coffees); check them against the statement."
        )

    by_month: dict[str, list[Decimal]] = defaultdict(lambda: [Decimal(0), Decimal(0), 0])
    for r in sorted(inside, key=lambda r: r.date):
        bucket = by_month[r.date.strftime("%Y-%m")]
        bucket[0 if r.amount > 0 else 1] += r.amount
        bucket[2] += 1
    months = []
    running = opening
    for month in sorted(by_month):
        money_in, money_out, count = by_month[month]
        running += money_in + money_out
        months.append(
            {
                "month": datetime.strptime(month, "%Y-%m").strftime("%b %Y"),
                "count": count,
                "money_in": _money(money_in),
                "money_out": _money(abs(money_out)),
                "closing": _money(running),
            }
        )

    return {
        "matches": difference == 0,
        "count": len(rows),
        "inside_count": len(inside),
        "first_date": min((r.date for r in rows), default=None),
        "last_date": max((r.date for r in rows), default=None),
        "opening": _money(opening),
        "money_in": _money(credits),
        "money_out": _money(abs(debits)),
        "computed": _money(computed),
        "closing": _money(closing),
        "difference": _money(difference),
        "findings": findings,
        "months": months,
        "outside": [_row_dict(r) for r in sorted(outside, key=lambda r: r.date)],
        "suspects": [_row_dict(r) for r in inside if difference and (r.amount == difference or r.amount * 2 == difference)],
        "repeated_ids": [_row_dict(r) for r in sorted(repeated_ids, key=lambda r: (r.fitid, r.date))],
        "true_duplicates": [_row_dict(r) for r in sorted(true_duplicates, key=lambda r: (r.uid, r.date))],
        "look_alikes": [_row_dict(r) for r in sorted(look_alikes, key=lambda r: (r.date, r.amount))],
    }


def compare(ofx_rows: list[Row], csv_rows: list[Row]) -> dict[str, Any]:
    """The CSV against the OFX it was made from, matched on the unique
    Transaction ID (ING's FITID repeats, so it can't be the key)."""
    ofx_by_id = {r.uid: r for r in ofx_rows if r.uid}
    csv_by_id = {r.uid: r for r in csv_rows if r.uid}
    missing = [r for uid, r in ofx_by_id.items() if uid not in csv_by_id]
    extra = [r for uid, r in csv_by_id.items() if uid not in ofx_by_id]
    changed = [
        {"fitid": uid, "ofx": _row_dict(r), "csv": _row_dict(csv_by_id[uid])}
        for uid, r in ofx_by_id.items()
        if uid in csv_by_id and (csv_by_id[uid].amount != r.amount or csv_by_id[uid].date != r.date)
    ]
    return {
        "ofx_count": len(ofx_rows),
        "csv_count": len(csv_rows),
        "ofx_total": _money(sum((r.amount for r in ofx_rows), Decimal(0))),
        "csv_total": _money(sum((r.amount for r in csv_rows), Decimal(0))),
        "identical": not missing and not extra and not changed and len(ofx_rows) == len(csv_rows),
        "missing": [_row_dict(r) for r in missing],
        "extra": [_row_dict(r) for r in extra],
        "changed": changed,
    }
