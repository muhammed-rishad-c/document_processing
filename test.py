# test_phase_a.py
import requests

BASE = "http://localhost:9000"

def show(label, resp):
    print(f"\n--- {label} ---")
    print("status:", resp.status_code)
    print("body:", resp.text)

# Test 1: create session
r = show("1. create session", requests.post(f"{BASE}/widget/session", json={}))
resp = requests.post(f"{BASE}/widget/session", json={})
show("1. create session", resp)
session_id = resp.json()["id"]

# Test 2: real question
resp = requests.post(f"{BASE}/widget/chat", json={
    "session_id": session_id,
    "query": "what services does liquidlab offer"
})
show("2. real question", resp)

# Test 3: bypass attempt (expect 422)
resp = requests.post(f"{BASE}/widget/chat", json={
    "session_id": session_id,
    "query": "test",
    "document_id": "00000000-0000-0000-0000-000000000000"
})
show("3. bypass attempt (expect 422)", resp)

# Test 4: invalid session (expect 404)
resp = requests.post(f"{BASE}/widget/chat", json={
    "session_id": "00000000-0000-0000-0000-000000000000",
    "query": "test"
})
show("4. invalid session (expect 404)", resp)

# Test 5: oversized query (expect 422)
resp = requests.post(f"{BASE}/widget/chat", json={
    "session_id": session_id,
    "query": "a" * 1500
})
show("5. oversized query (expect 422)", resp)