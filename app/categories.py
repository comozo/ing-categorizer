"""The fixed category taxonomy every transaction is classified into.

Fixed rather than pulled from any one finance app's API - the output is a
plain CSV, reusable regardless of which app (Securo / Actual Budget /
Firefly III) it eventually gets imported into.
"""

CATEGORIES = [
    "Groceries",
    "Dining & Takeaway",
    "Transport",
    "Utilities",
    "Rent & Mortgage",
    "Subscriptions",
    "Shopping",
    "Health & Fitness",
    "Entertainment",
    "Travel",
    "Fees & Charges",
    "Salary & Income",
    "Interest",
    "Transfer",
    "Uncategorized",
]
