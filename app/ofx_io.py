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

# ofxtools enforces several fields as hard-required that we never actually read - we only ever
# pull statement.transactions out of a parsed file, nothing about the statement period, which
# account it's for, or its balance. ING Australia's real export omits all three of the fields
# below entirely. The first two were each confirmed against a real upload, one at a time, as
# each became the next error in turn; LEDGERBAL was caught ahead of time instead, by walking
# ofxtools' own model spec for every aggregate a bank STMTTRNRS message touches and checking
# ING's export against each one's required fields, rather than waiting for it to crash:
#
#   - <BANKTRANLIST>'s DTSTART/DTEND (the statement period)
#   - <STMTRS>'s BANKACCTFROM (bank id / account id / account type)
#   - <STMTRS>'s LEDGERBAL (balance amount / as-of date) - has no optional fields of its own
#     to fall back on, unlike the other two, so a missing LEDGERBAL has no lenient path at all
#
# ofxtools has no API to relax these, so each gets a harmless placeholder injected before
# parsing - but only when genuinely missing; a well-formed block is left untouched. ofxtools
# also enforces field *order* (confirmed by trial: inserting BANKACCTFROM before CURDEF instead
# of after raises "Elements out of order"), so each patch's anchor and insertion point respects
# where the chart's own spec says that field belongs - STMTRS.spec order is curdef, bankacctfrom,
# banktranlist, banktranlistp, ledgerbal, availbal, ..., so LEDGERBAL anchors on BANKTRANLIST's
# closing tag (ING never sends BANKTRANLISTP or AVAILBAL, which would otherwise sit between them).
#
# Add another entry here, following the same shape, if a real file surfaces a fourth one -
# that's the expected way this list grows, not a sign something's wrong with the approach.
_REQUIRED_FIELD_PATCHES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "<BANKTRANLIST> missing DTSTART/DTEND",
        re.compile(r"(<BANKTRANLIST>)(?!\s*<DTSTART)"),
        r"\1<DTSTART>19700101000000</DTSTART><DTEND>20991231000000</DTEND>",
    ),
    (
        "<STMTRS> missing BANKACCTFROM",
        # CURDEF's value is always a 3-letter ISO currency code (OFX spec) - anchoring on the
        # whole tag+value, not just <CURDEF>, keeps the insertion point after it as the spec
        # requires (BANKACCTFROM follows CURDEF in STMTRS's own field order).
        re.compile(r"(<CURDEF>[A-Z]{3})(?!\s*<BANKACCTFROM)"),
        r"\1<BANKACCTFROM><BANKID>000000000</BANKID><ACCTID>UNKNOWN</ACCTID>"
        r"<ACCTTYPE>CHECKING</ACCTTYPE></BANKACCTFROM>",
    ),
    (
        "<STMTRS> missing LEDGERBAL",
        re.compile(r"(</BANKTRANLIST>)(?!\s*<LEDGERBAL>)"),
        r"\1<LEDGERBAL><BALAMT>0.00</BALAMT><DTASOF>19700101000000</DTASOF></LEDGERBAL>",
    ),
]


def _patch_missing_required_fields(text: str) -> tuple[str, list[str]]:
    """Returns the patched text and a list naming which patches actually fired."""
    applied: list[str] = []
    for description, pattern, placeholder in _REQUIRED_FIELD_PATCHES:
        text, count = pattern.subn(placeholder, text)
        if count:
            applied.append(f"{description} (x{count})")
    return text, applied


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

    patched_text, applied = _patch_missing_required_fields(text)
    if applied:
        logger.info("Patched missing required OFX field(s): %s", "; ".join(applied))
    parse_bytes = patched_text.encode(encoding) if applied else raw_bytes

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
