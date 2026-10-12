"""Reduces a raw ING transaction description to a stable merchant key.

ING Australia descriptions carry a per-transaction tail - "WOOLWORTHS 1234
SYDNEY - Visa Purchase - Receipt 123456 Date 12 Mar 2026 Card 462263xxxx1234"
- so the same shop never produces the same description twice. Keyed on the
raw text, every weekly grocery run looked like a brand-new merchant: its own
Ollama call, its own review card, and nothing learned from last time. The key
keeps only the merchant segment, with digits and punctuation stripped, so
"WOOLWORTHS 1234 SYDNEY" and "WOOLWORTHS 5678 SYDNEY" both become
"WOOLWORTHS SYDNEY".
"""

from __future__ import annotations

import re

# Segments ING inserts between " - " separators that describe the payment
# rail, not who was paid. Skipped when picking the merchant segment.
_RAIL_SEGMENTS = {
    "VISA PURCHASE",
    "VISA REFUND",
    "EFTPOS PURCHASE",
    "EFTPOS REFUND",
    "DIRECT DEBIT",
    "DIRECT CREDIT",
    "OSKO PAYMENT",
    "BPAY",
    "ATM WITHDRAWAL",
}

# Everything from one of these words onward is per-transaction noise.
_TAIL = re.compile(r"\b(RECEIPT|DATE|CARD|REF|REFERENCE)\b.*$")
_NON_LETTERS = re.compile(r"[^A-Z& ]+")


def merchant_key(description: str) -> str:
    text = description.upper()
    segments = [s.strip() for s in re.split(r"\s+-\s+", text) if s.strip()]
    merchant = next(
        (s for s in segments if s not in _RAIL_SEGMENTS and not _TAIL.match(s)),
        text,
    )
    merchant = _TAIL.sub("", merchant)
    words = _NON_LETTERS.sub(" ", merchant).split()
    # "AMZNPRIMEA* AMZNPRIMEA" -> "AMZNPRIMEA": drop immediate repeats.
    deduped = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]
    return " ".join(deduped) or text.strip()
