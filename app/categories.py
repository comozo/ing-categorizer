"""The fixed category taxonomy every transaction is classified into.

Fixed rather than pulled from any one finance app's API - the output is a
plain CSV, reusable regardless of which app (Securo / Actual Budget /
Firefly III) it eventually gets imported into.

Grouped (not just a flat list) because the review screen renders every
category as a colored, icon-labeled chip - the grouping and color live here,
one place, rather than being duplicated into the template or the frontend JS.
"""

CATEGORY_GROUPS = [
    {
        "name": "Housing",
        "color": "#8b5cf6",
        "items": [
            {"name": "Housing", "icon": "🏠"},
            {"name": "Utilities", "icon": "💡"},
        ],
    },
    {
        "name": "Food & Dining",
        "color": "#f5a524",
        "items": [
            {"name": "Food & Dining", "icon": "🍴"},
            {"name": "Groceries", "icon": "🛒"},
        ],
    },
    {
        "name": "Transport",
        "color": "#3ea6ff",
        "items": [{"name": "Transport", "icon": "🚗"}],
    },
    {
        "name": "Lifestyle",
        "color": "#ec4899",
        "items": [
            {"name": "Education", "icon": "📚"},
            {"name": "Health", "icon": "💊"},
            {"name": "Leisure", "icon": "🎮"},
            {"name": "Personal Care", "icon": "✂️"},
        ],
    },
    {
        "name": "Other",
        "color": "#8a93a7",
        "items": [
            {"name": "Donations", "icon": "🎁"},
            {"name": "Investments", "icon": "📈"},
            {"name": "Other", "icon": "❓"},
            {"name": "Shopping", "icon": "🛍️"},
            {"name": "Subscriptions", "icon": "📱"},
            {"name": "Taxes & Fees", "icon": "🏛️"},
            {"name": "Transfers", "icon": "🔁"},
        ],
    },
    {
        "name": "Income",
        "color": "#22c55e",
        "items": [
            {"name": "Salary & Income", "icon": "💰"},
            {"name": "Interest", "icon": "🪙"},
        ],
    },
]

# Flattened for everywhere that just needs the plain list: the <select> fallback,
# the local classifier's label set, and Ollama's structured-output schema/prompt.
# "Uncategorized" isn't a real spending category - it's the fallback/skip value -
# so it's appended here rather than living in a group of its own.
CATEGORIES = [item["name"] for group in CATEGORY_GROUPS for item in group["items"]] + [
    "Uncategorized"
]
