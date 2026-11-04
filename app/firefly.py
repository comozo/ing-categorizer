"""Sends a finished review straight to Firefly III through its Data Importer.

Instead of downloading the CSV and uploading it again in the importer's web
UI, the review screen posts it to the importer's /autoupload endpoint with a
generated import configuration: the columns this app writes, mapped to
Firefly III fields, the account picked on the review screen, and duplicate
detection on the Transaction ID column (never on FITID: ING reuses it, see
ofx_io.transaction_ids()).

The importer answers /autoupload with a generic message whatever happened -
rows it skipped or failed only reach its own log - so a push is only
reported as done after checking Firefly III itself: every Transaction ID of
the statement must be present on that account, as a transaction's external
ID. That's what the review screen shows, row by row if anything is missing.

Configuration, all optional (the feature is hidden unless all four are set):
  FIREFLY_URL        Firefly III's API base, e.g. http://firefly-iii.firefly-iii.svc.cluster.local:8080
  FIREFLY_PUBLIC_URL what the browser opens to show an account (defaults to FIREFLY_URL)
  FIREFLY_TOKEN      a Firefly III personal access token
  IMPORTER_URL       the Data Importer, e.g. http://firefly-iii-data-importer.firefly-iii.svc.cluster.local:8080
  IMPORTER_SECRET    the importer's AUTO_IMPORT_SECRET (16+ characters)
The token travels to the importer as a Bearer header per request, so the
importer itself needs no stored token and its interactive OAuth login is
unaffected.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FIREFLY_URL = os.environ.get("FIREFLY_URL", "").rstrip("/")
FIREFLY_PUBLIC_URL = os.environ.get("FIREFLY_PUBLIC_URL", FIREFLY_URL).rstrip("/")
FIREFLY_TOKEN = os.environ.get("FIREFLY_TOKEN", "")
IMPORTER_URL = os.environ.get("IMPORTER_URL", "").rstrip("/")
IMPORTER_SECRET = os.environ.get("IMPORTER_SECRET", "")

# How each column write_categorized_csv() produces is imported. Type and FITID
# are informational only: the sign of Amount already says money in or out,
# and FITID isn't unique.
_ROLES = {
    "Date": "date_transaction",
    "Description": "description",
    "Amount": "amount",
    "Type": "_ignore",
    "FITID": "_ignore",
    "Transaction ID": "external-id",
    "Category": "category-name",
}

# The import itself can take minutes for a year of transactions.
_IMPORT_TIMEOUT = httpx.Timeout(1800.0, connect=10.0)
_API_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class FireflyError(Exception):
    """Firefly III or the importer refused or couldn't be reached. The
    message is what the review screen shows."""


def enabled() -> bool:
    return bool(FIREFLY_URL and FIREFLY_TOKEN and IMPORTER_URL and len(IMPORTER_SECRET) >= 16)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {FIREFLY_TOKEN}", "Accept": "application/vnd.api+json"}


async def _get(client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None) -> dict:
    try:
        resp = await client.get(f"{FIREFLY_URL}/api/v1{path}", params=params, headers=_headers(), timeout=_API_TIMEOUT)
    except httpx.HTTPError as exc:
        raise FireflyError(f"Can't reach Firefly III at {FIREFLY_URL} ({type(exc).__name__}).") from exc
    if resp.status_code in (401, 403):
        raise FireflyError("Firefly III refused the access token. It may have been revoked or expired.")
    if resp.status_code >= 400:
        raise FireflyError(f"Firefly III answered HTTP {resp.status_code} for {path}.")
    return resp.json()


async def asset_accounts() -> list[dict[str, Any]]:
    """Active asset accounts, for the account picker."""
    accounts, page = [], 1
    async with httpx.AsyncClient() as client:
        while True:
            body = await _get(client, "/accounts", {"type": "asset", "page": page, "limit": 100})
            for item in body.get("data", []):
                attrs = item["attributes"]
                if attrs.get("active", True):
                    accounts.append(
                        {
                            "id": item["id"],
                            "name": attrs["name"],
                            "balance": attrs.get("current_balance"),
                            "currency": attrs.get("currency_code") or "",
                        }
                    )
            pages = body.get("meta", {}).get("pagination", {}).get("total_pages", 1)
            if page >= pages:
                return accounts
            page += 1


def account_url(account_id: str) -> str:
    return f"{FIREFLY_PUBLIC_URL}/accounts/show/{account_id}"


def import_config(csv_text: str, account_id: str, filename: str) -> dict[str, Any]:
    header = next(csv.reader(io.StringIO(csv_text)))
    roles = [_ROLES.get(column, "_ignore") for column in header]
    return {
        "version": 3,
        "flow": "file",
        "content_type": "csv",
        "file_name": filename,
        "date": "Y-m-d",
        "delimiter": "comma",
        "headers": True,
        "default_account": int(account_id),
        "roles": roles,
        "do_mapping": [False] * len(roles),
        "mapping": {},
        "rules": True,
        "skip_form": True,
        # The tag costs a second API call per transaction, each of which makes
        # Firefly III recalculate running balances again.
        "add_import_tag": False,
        "duplicate_detection_method": "cell",
        "unique_column_index": header.index("Transaction ID"),
        "unique_column_type": "external-id",
        "ignore_duplicate_lines": False,
        "conversion": False,
    }


async def upload(csv_text: str, config: dict[str, Any], filename: str) -> str:
    """Posts the CSV and its configuration to the importer. Returns the
    importer's message; raises FireflyError if it refused the job."""
    files = {
        "importable": (filename, csv_text.encode("utf-8"), "text/csv"),
        "json": ("config.json", json.dumps(config).encode("utf-8"), "application/json"),
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{IMPORTER_URL}/autoupload",
                params={"secret": IMPORTER_SECRET},
                headers={"Authorization": f"Bearer {FIREFLY_TOKEN}", "Accept": "application/json"},
                files=files,
                timeout=_IMPORT_TIMEOUT,
            )
    except httpx.HTTPError as exc:
        raise FireflyError(f"Can't reach the Data Importer at {IMPORTER_URL} ({type(exc).__name__}).") from exc
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code >= 400 or "error" in body:
        detail = body.get("error") or body.get("message") or resp.text[:300]
        raise FireflyError(f"The Data Importer refused the import (HTTP {resp.status_code}): {detail}")
    return str(body.get("message", "accepted"))


async def external_ids_on_account(account_id: str, start: str, end: str) -> set[str]:
    """Every external ID on the account's transactions between two dates."""
    found: set[str] = set()
    page = 1
    async with httpx.AsyncClient() as client:
        while True:
            body = await _get(
                client,
                f"/accounts/{account_id}/transactions",
                {"start": start, "end": end, "page": page, "limit": 200},
            )
            for group in body.get("data", []):
                for split in group["attributes"].get("transactions", []):
                    if split.get("external_id"):
                        found.add(split["external_id"])
            pages = body.get("meta", {}).get("pagination", {}).get("total_pages", 1)
            if page >= pages:
                return found
            page += 1


async def account(account_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        attrs = (await _get(client, f"/accounts/{account_id}"))["data"]["attributes"]
    return {"name": attrs["name"], "balance": attrs.get("current_balance"), "currency": attrs.get("currency_code") or ""}
