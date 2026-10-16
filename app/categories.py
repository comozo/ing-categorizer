"""The fixed category taxonomy every transaction is classified into.

Fixed rather than pulled from any one finance app's API - the output is a
plain CSV, reusable regardless of which app (Securo / Actual Budget /
Firefly III) it eventually gets imported into.

Grouped (not just a flat list) because the review screen renders every
category as a colored, icon-labeled chip - the grouping and color live here,
one place, rather than being duplicated into the template or the frontend JS.

Each item's hint is what Ollama is told the category means (app/classifier.py):
a small model given bare names can't tell Leisure from Shopping from
Subscriptions, and knows few Australian merchants by name.
"""

CATEGORY_GROUPS = [
    {
        "name": "Housing",
        "color": "#8b5cf6",
        "items": [
            {"name": "Housing", "icon": "🏠", "hint": "rent, mortgage repayments, strata, council rates, home and contents insurance"},
            {"name": "Utilities", "icon": "💡", "hint": "electricity, gas, water, internet, mobile and phone plans (AGL, Origin, EnergyAustralia, Telstra, Optus, Aussie Broadband)"},
        ],
    },
    {
        "name": "Food & Dining",
        "color": "#f5a524",
        "items": [
            {"name": "Food & Dining", "icon": "🍴", "hint": "restaurants, cafes, takeaway, bars, food delivery (Uber Eats, DoorDash, Menulog, McDonald's)"},
            {"name": "Groceries", "icon": "🛒", "hint": "supermarkets and food shops (Woolworths, Coles, Aldi, IGA, Harris Farm, butchers, bakeries)"},
        ],
    },
    {
        "name": "Transport",
        "color": "#3ea6ff",
        "items": [{"name": "Transport", "icon": "🚗", "hint": "fuel, public transport, rideshare, tolls, parking, car registration and servicing (Opal, Myki, Uber trips, Linkt, Ampol, BP, Shell)"}],
    },
    {
        "name": "Lifestyle",
        "color": "#ec4899",
        "items": [
            {"name": "Education", "icon": "📚", "hint": "courses, tuition, school fees, study materials (Udemy, Coursera, universities)"},
            {"name": "Health", "icon": "💊", "hint": "doctors, dentists, pharmacies, health insurance, gyms (Chemist Warehouse, Priceline, Bupa, Medibank)"},
            {"name": "Leisure", "icon": "🎮", "hint": "entertainment, events, cinemas, hobbies, sport, travel, flights and hotels (Ticketek, Event Cinemas, Qantas, Airbnb)"},
            {"name": "Personal Care", "icon": "✂️", "hint": "hairdressers, barbers, beauty, cosmetics (Mecca, Sephora)"},
        ],
    },
    {
        "name": "Other",
        "color": "#8a93a7",
        "items": [
            {"name": "Donations", "icon": "🎁", "hint": "charities and fundraising (Red Cross, GoFundMe)"},
            {"name": "Investments", "icon": "📈", "hint": "share trading, super contributions, crypto (CommSec, Stake, Pearler, Vanguard, Raiz, Spaceship)"},
            {"name": "Other", "icon": "❓", "hint": "anything that genuinely fits no other category"},
            {"name": "Shopping", "icon": "🛍️", "hint": "retail goods: clothing, electronics, homeware, hardware, online marketplaces (Amazon purchases, Kmart, Target, Big W, JB Hi-Fi, Bunnings, eBay)"},
            {"name": "Subscriptions", "icon": "📱", "hint": "recurring digital services and memberships (Netflix, Spotify, Disney+, Stan, Amazon Prime, Apple, Google, iCloud)"},
            {"name": "Taxes & Fees", "icon": "🏛️", "hint": "tax payments, the ATO, bank fees, fines, government charges"},
            {"name": "Transfers", "icon": "🔁", "hint": "money moving between your own accounts, or to and from friends and family"},
        ],
    },
    {
        "name": "Income",
        "color": "#22c55e",
        "items": [
            {"name": "Salary & Income", "icon": "💰", "hint": "wages, salary, government payments, tax refunds, reimbursements received"},
            {"name": "Interest", "icon": "🪙", "hint": "interest credited to an account"},
        ],
    },
]

CATEGORY_HINTS = {item["name"]: item["hint"] for group in CATEGORY_GROUPS for item in group["items"]}

# Flattened for everywhere that just needs the plain list: the <select> fallback,
# the local classifier's label set, and Ollama's structured-output schema/prompt.
# "Uncategorized" isn't a real spending category - it's the fallback/skip value -
# so it's appended here rather than living in a group of its own.
CATEGORIES = [item["name"] for group in CATEGORY_GROUPS for item in group["items"]] + [
    "Uncategorized"
]
