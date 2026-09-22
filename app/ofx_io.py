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
import logging
import re
from dataclasses import dataclass

from ofxtools.Parser import OFXTree

logger = logging.getLogger(__name__)

# ofxtools treats <BANKTRANLIST>'s DTSTART/DTEND as hard-required, even though they're just
# the statement period, not anything about individual transactions - we never read them. ING
# Australia's own export omits them entirely (confirmed against a real file, which failed with
# "Can't set BANKTRANLIST.dtstart to None: DateTime: Value is required"), so inject harmless
# placeholders before parsing rather than depend on ofxtools relaxing a spec requirement it has
# no option to relax. Only touches a <BANKTRANLIST> that's missing DTSTART - a well-formed one
# is left untouched.
_MISSING_BANKTRANLIST_DATES = re.compile(r"(<BANKTRANLIST>)(?!\s*<DTSTART)")
_BANKTRANLIST_DATE_PLACEHOLDER = (
    r"\1<DTSTART>19700101000000</DTSTART><DTEND>20991231000000</DTEND>"
)


def _patch_missing_bank_tranlist_dates(text: str) -> tuple[str, int]:
    """Returns the patched text and how many <BANKTRANLIST> blocks were patched."""
    return _MISSING_BANKTRANLIST_DATES.subn(_BANKTRANLIST_DATE_PLACEHOLDER, text)


class OfxFormatError(ValueError):
    """Raised when the uploaded file doesn't look like an OFX/QFX export we can read."""


def _diagnose(raw_bytes: bytes, exc: Exception) -> str:
    """Builds a log line describing *why* parsing failed, without echoing anything
    that could be transaction data - only format-level structure: byte count, the
    file's OFX header block (this is metadata like VERSION/ENCODING, never
    transaction content, whether SGML- or XML-flavored), and whether it looks like
    CSV was uploaded by mistake despite the prompt. The exception message itself
    is logged too, but ofxtools' own errors sometimes echo back a snippet of
    whatever it choked on - so that part only goes to the log, never into the
    response shown in the browser.
    """
    size = len(raw_bytes)
    try:
        text = raw_bytes.decode("utf-8-sig", errors="replace")
    except Exception:
        text = raw_bytes.decode("latin-1", errors="replace")

    first_line = text.splitlines()[0] if text.splitlines() else ""
    looks_like_csv = "," in first_line and "<" not in first_line and "OFXHEADER" not in text[:200]
    header_end = text.find("<OFX")
    header_block = text[:header_end].strip() if header_end > 0 else "(no <OFX> tag found)"

    return (
        f"OFX parse failed: {type(exc).__name__}: {exc}\n"
        f"  file size: {size} bytes\n"
        f"  first line: {first_line[:200]!r}\n"
        f"  looks like CSV was uploaded instead of OFX: {looks_like_csv}\n"
        f"  header block before <OFX>: {header_block[:500]!r}"
    )


@dataclass
class Transaction:
    row_id: int
    date: str
    description: str
    amount: str  # kept as a string for lossless round-tripping into the output CSV
    raw: dict[str, str]  # flattened OFX fields, for the final export


def parse_ofx(raw_bytes: bytes) -> list[Transaction]:
    # Patch on the decoded text, then re-encode with the same encoding declared in the OFX
    # header (or a safe default) - re-encoding as UTF-8 regardless of what the header
    # declares would leave the two disagreeing, and ofxtools reads the header itself.
    try:
        text = raw_bytes.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw_bytes.decode("cp1252", errors="replace")
        encoding = "cp1252"

    patched_text, patched_count = _patch_missing_bank_tranlist_dates(text)
    if patched_count:
        logger.info(
            "Patched %d <BANKTRANLIST> block(s) missing DTSTART/DTEND with placeholder dates",
            patched_count,
        )
    parse_bytes = patched_text.encode(encoding) if patched_count else raw_bytes

    tree = OFXTree()
    try:
        tree.parse(io.BytesIO(parse_bytes))
        ofx = tree.convert()
    except Exception as exc:
        logger.error(_diagnose(raw_bytes, exc))
        raise OfxFormatError(
            "Couldn't read this as an OFX/QFX file. Make sure it's exported in that format, "
            "not CSV. More detail was written to the server log to help track down why."
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
