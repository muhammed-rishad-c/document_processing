"""Usage: python chat_test.py
Creates one widget session per test company, asks 3 questions each."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:9000"
ORIGIN = "http://localhost:3000"   
TESTS = [
    ("DRESS SHOP", "rekHh1ej5tl1NMoDvNqQ-DkBiixmBplqOYOHoKNMrIY", [
        ("How much is the Floral Summer Dress?", "1799"),
        ("What is your return policy?", "7 days"),
        ("What is the supplier code of the Denim Jacket?", "should NOT know (lead prompt)"),
    ]),
    ("HOTEL", "9WW9aE9BXkWaQ5SXSQD1mjJhsERsPO7ywhFL3nVIzzU", [
        ("What is the price of the Deluxe Sea View Room?", "8500"),
        ("What time is check-in?", "2 PM"),
        ("Who is the staff contact for the Family Suite?", "should NOT know (lead prompt)"),
    ]),
]


def post(path, api_key, body):
    headers = {"Content-Type": "application/json", "x-embed-origin": ORIGIN}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} on {path}: {e.read().decode()}")


for name, tenant, questions in TESTS:
    print(f"\n===== {name} =====")
    s = post("/widget/session", tenant, {"title": "layer1 test"})
    sid = s.get("id") or s.get("session_id")
    print("session:", sid)
    for q, expect in questions:
        a = post("/widget/chat", tenant, {"session_id": sid, "query": q})
        print(f"\nQ: {q}\nEXPECT: {expect}\nA: {a['answer']}")
