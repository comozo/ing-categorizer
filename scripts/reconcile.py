"""Command-line version of the "Check against a statement" page.

    python -m scripts.reconcile --start 2025-07-01 --end 2026-06-30 \\
        --opening 4210.55 --closing 6873.10 \\
        --csv categorized.csv --ofx export.ofx
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app import reconcile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--opening", type=reconcile.parse_amount, required=True)
    parser.add_argument("--closing", type=reconcile.parse_amount, required=True)
    parser.add_argument("--total-credits", type=reconcile.parse_amount)
    parser.add_argument("--total-debits", type=reconcile.parse_amount)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--ofx", type=Path)
    args = parser.parse_args()

    ofx_rows = reconcile.rows_from_ofx(args.ofx.read_bytes()) if args.ofx else None
    csv_rows = reconcile.rows_from_csv(args.csv.read_bytes()) if args.csv else None
    rows = csv_rows if csv_rows is not None else ofx_rows
    if rows is None:
        parser.error("give --csv, --ofx or both")

    r = reconcile.reconcile(rows, args.start, args.end, args.opening, args.closing, args.total_credits, args.total_debits)
    print("MATCHES" if r["matches"] else f"OFF BY {r['difference']}")
    print(f"{r['count']} transactions, {r['first_date']} to {r['last_date']}, {r['inside_count']} inside the period")
    print(f"  opening {r['opening']} + in {r['money_in']} - out {r['money_out']} = {r['computed']} (statement {r['closing']})")
    for finding in r["findings"]:
        print(f"  - {finding}")
    for title, key in (("gap-sized", "suspects"), ("outside the period", "outside"), ("repeated IDs", "repeated_ids"), ("look-alikes", "look_alikes")):
        for row in r[key]:
            print(f"  [{title}] {row['date']} {row['amount']:>12} {row['fitid']:>10}  {row['description'][:60]}")
    print("\nmonth        n          in         out     closing")
    for m in r["months"]:
        print(f"{m['month']:9} {m['count']:4} {m['money_in']:>11} {m['money_out']:>11} {m['closing']:>11}")
    if ofx_rows is not None and csv_rows is not None:
        c = reconcile.compare(ofx_rows, csv_rows)
        print("\nCSV vs export:", "identical" if c["identical"] else
              f"{len(c['missing'])} missing, {len(c['extra'])} extra, {len(c['changed'])} changed "
              f"(export {c['ofx_count']} / {c['ofx_total']}, CSV {c['csv_count']} / {c['csv_total']})")


if __name__ == "__main__":
    main()
