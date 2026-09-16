"""
grid_experiment.py — combined chunk_size x chunk_overlap x top_k sweep

Runs a full parameter grid in ONE script instead of separately running
chunk_size_experiment.py and topk_experiment.py. For each unique
(chunk_size, chunk_overlap) pair it re-indexes the documents ONCE, then
loops over every top_k value that pairs with it — so you don't pay for a
re-upload per grid point, only per distinct chunk_size/overlap.

Each grid point reports the full metric set from both source scripts:
  - avg_latency_ms
  - top1_accuracy_pct / top3_accuracy_pct  (document-level retrieval)
  - avg_cosine_similarity_pct              (answer quality vs expected_answer)
  - avg_context_tokens_est                 (0 if /documents/chat doesn't
    return chunk text under any of CHUNK_TEXT_KEYS — see NOTE below)

DAILY BUDGET
------------
Same OpenRouter free-tier constraint as chunk_size_experiment.py: 50
requests/day. Each grid point costs ~25 calls (one per eval question).
DAILY_CALL_BUDGET below stops the run cleanly once hit, mid-grid-point if
necessary, and discards that partial point so it's retried in full next
time. Already-completed grid points are skipped automatically on rerun
(results accumulate in RESULTS_PATH) unless FORCE_RERUN=True.

Just edit PARAM_GRID below, run it, and paste back the printed table /
the saved JSON file — no need to run multiple scripts or merge results
yourself.

NOTE ON CONTEXT TOKENS
-----------------------
avg_context_tokens_est will show 0 unless /documents/chat's response
includes the actual chunk text somewhere in each `sources[i]` entry. This
script checks a few likely key names (see CHUNK_TEXT_KEYS) — if your API
uses a different key, add it to that list, or print one raw response
(see the commented-out debug line in run_for_grid_point) to find it.

USAGE
-----
Run from the project root:
    python eval/grid_experiment.py
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import requests
import tiktoken
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — EDIT THESE
# ---------------------------------------------------------------------------

# Path to the source document(s) your eval_dataset.json questions reference.
# Re-uploaded fresh every time chunk_size/chunk_overlap changes.
SOURCE_FILES: list[str] = [
    "eval/liquidlab_content.txt",
]

# Every (chunk_size, chunk_overlap, top_k) combo you want tested.
# Order doesn't matter for correctness — the script internally groups and
# sorts by (chunk_size, chunk_overlap) so uploads only happen when that
# pair changes, regardless of the order you list them in here.
PARAM_GRID: list[dict] = [
    # --- sanity re-checks (see previous results) ---
    {"chunk_size": 500, "chunk_overlap": 50, "top_k": 6},   # re-check the cosine dip / latency outlier
    # EDIT: set chunk_size below to whatever was actually indexed when you
    # ran the original topk_experiment.py sweep, so this re-checks the
    # real top_k=4 cliff. Left at 500 as a placeholder.
    {"chunk_size": 500, "chunk_overlap": 50, "top_k": 4},   # re-check the 13.64% retrieval cliff

    # --- fills in the missing corners of the chunk_size x top_k grid ---
    {"chunk_size": 450, "chunk_overlap": 50, "top_k": 7},
    {"chunk_size": 600, "chunk_overlap": 50, "top_k": 7},

    # --- optional overlap test — only meaningful if chunk_size=600 wins above ---
    {"chunk_size": 600, "chunk_overlap": 75, "top_k": 7},
    {"chunk_size": 600, "chunk_overlap": 100, "top_k": 7},
]

# Hard stop once this many LLM calls have been made in this invocation,
# leaving headroom under the 50/day cap for retries. A grid point
# interrupted partway is discarded and retried in full next time.
DAILY_CALL_BUDGET = 1000

FORCE_RERUN = False  # set True to re-run grid points already in the results file

# Possible key names for chunk text in each `sources[i]` entry of the
# /documents/chat response. Add more here if yours uses a different name.
CHUNK_TEXT_KEYS = ["chunk_text", "text", "content"]

# ---------------------------------------------------------------------------
# Config — shouldn't need to change
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIR / "grid_experiment_results.json"

BASE_URL = "http://localhost:9000"
DOCUMENTS_ENDPOINT = f"{BASE_URL}/documents"
UPLOAD_ENDPOINT = f"{BASE_URL}/documents/upload"
CHAT_ENDPOINT = f"{BASE_URL}/documents/chat"

MAX_RETRIES = 3

tokenizer = tiktoken.get_encoding("cl100k_base")

print("Loading sentence-transformer model for cosine similarity scoring...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


class BudgetExceeded(Exception):
    """Raised when DAILY_CALL_BUDGET is hit mid-run, so main() can stop
    cleanly and save whatever full grid points already completed."""
    pass


# ---------------------------------------------------------------------------
# Shared helpers (same logic as chunk_size_experiment.py / topk_experiment.py)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def get_filename_to_doc_id_map() -> dict[str, str]:
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()
    return {d["filename"]: str(d["id"]) for d in docs}


def resolve_expected_sources(expected_sources: set[str], filename_to_id: dict[str, str]) -> set[str]:
    resolved = set()
    for name in expected_sources:
        doc_id = filename_to_id.get(name)
        if doc_id:
            resolved.add(doc_id)
        else:
            print(f"    [WARN] expected_sources filename '{name}' not found among "
                  f"currently uploaded documents.")
    return resolved


def post_with_retry(url: str, payload: dict, timeout: int) -> requests.Response:
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_exception = e
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"    [WARN] Request error ({e}); retrying in {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code == 429:
            if attempt == MAX_RETRIES:
                return response
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2 ** attempt
            print(f"    [WARN] 429 rate-limited; retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        return response

    if last_exception:
        raise last_exception
    return response


def cosine_similarity(generated: str, expected: str) -> float:
    if not generated.strip() or not expected.strip():
        return 0.0
    embeddings = similarity_model.encode([generated, expected])
    vec1, vec2 = embeddings[0], embeddings[1]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), 0.0, 1.0))


def extract_chunk_text(source: dict) -> str:
    for key in CHUNK_TEXT_KEYS:
        if key in source and source[key]:
            return source[key]
    return ""


# ---------------------------------------------------------------------------
# Document reset / re-upload — only called when (chunk_size, overlap) changes
# ---------------------------------------------------------------------------

def clear_all_documents() -> int:
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()

    deleted = 0
    for doc in docs:
        doc_id = doc["id"]
        del_resp = requests.delete(f"{DOCUMENTS_ENDPOINT}/{doc_id}", timeout=30)
        if del_resp.status_code == 200:
            deleted += 1
        else:
            print(f"    [WARN] failed to delete document {doc_id}: "
                  f"HTTP {del_resp.status_code} {del_resp.text[:150]}")
    return deleted


def upload_documents(chunk_size: int, chunk_overlap: int) -> int:
    uploaded = 0
    for path_str in SOURCE_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"    [WARN] source file not found, skipping: {path}")
            continue

        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            resp = requests.post(UPLOAD_ENDPOINT, files=files, params=params, timeout=120)

        if resp.status_code == 201:
            uploaded += 1
            body = resp.json()
            print(f"    Uploaded {path.name}: {body.get('total_chunks', '?')} chunks")
        else:
            print(f"    [WARN] failed to upload {path.name}: "
                  f"HTTP {resp.status_code} {resp.text[:150]}")
    return uploaded


def reset_and_upload(chunk_size: int, chunk_overlap: int) -> None:
    print(f"  Clearing existing documents...")
    deleted = clear_all_documents()
    print(f"  Deleted {deleted} existing document(s).")

    print(f"  Re-uploading {len(SOURCE_FILES)} source file(s) at "
          f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    uploaded = upload_documents(chunk_size, chunk_overlap)
    print(f"  Uploaded {uploaded}/{len(SOURCE_FILES)} file(s).")

    if uploaded == 0:
        raise RuntimeError(
            "No documents were uploaded — check SOURCE_FILES paths before continuing."
        )


# ---------------------------------------------------------------------------
# Eval run for a single (chunk_size, chunk_overlap, top_k) grid point
# ---------------------------------------------------------------------------

def run_for_grid_point(questions: list[dict], chunk_size: int, chunk_overlap: int,
                        top_k: int, call_counter: list[int]) -> dict:
    latencies = []
    cosine_scores = []
    context_token_estimates = []
    top1_hits, top3_hits, retrieval_checked = 0, 0, 0

    filename_to_id = get_filename_to_doc_id_map()

    for item in questions:
        if call_counter[0] >= DAILY_CALL_BUDGET:
            raise BudgetExceeded(
                f"Hit DAILY_CALL_BUDGET ({DAILY_CALL_BUDGET}) partway through "
                f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k} "
                f"({len(latencies)}/{len(questions)} questions done)."
            )

        category = item.get("category", "uncategorized")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = resolve_expected_sources(
            set(item.get("expected_sources", [])), filename_to_id
        )
        is_unanswerable = item.get("is_unanswerable", category == "unanswerable")

        payload = {"query": question, "top_k": top_k}
        call_counter[0] += 1

        start = time.perf_counter()
        try:
            response = post_with_retry(CHAT_ENDPOINT, payload, timeout=90)
        except requests.RequestException as e:
            print(f"    [WARN] request failed for '{question[:40]}...': {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            print(f"    [WARN] HTTP {response.status_code} for '{question[:40]}...'")
            continue

        latencies.append(elapsed_ms)
        data = response.json()
        generated_answer = data.get("answer", "")
        sources = data.get("sources", [])
        retrieved_doc_ids = [s.get("document_id", "") for s in sources]

        # Uncomment to debug the context-token key name (run once, check stdout):
        # if sources:
        #     print("    [DEBUG] raw source keys:", list(sources[0].keys()))

        chunk_texts = [extract_chunk_text(s) for s in sources]
        chunk_texts = [t for t in chunk_texts if t]
        if chunk_texts:
            context_token_estimates.append(sum(len(tokenizer.encode(t)) for t in chunk_texts))

        if not is_unanswerable and expected_sources:
            retrieval_checked += 1
            if retrieved_doc_ids and retrieved_doc_ids[0] in expected_sources:
                top1_hits += 1
            if any(doc_id in expected_sources for doc_id in retrieved_doc_ids):
                top3_hits += 1

        if not is_unanswerable:
            cosine_scores.append(cosine_similarity(generated_answer, expected_answer))

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "questions_run": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "top1_accuracy_pct": round(100 * top1_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "top3_accuracy_pct": round(100 * top3_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "avg_cosine_similarity_pct": round(100 * sum(cosine_scores) / len(cosine_scores), 2) if cosine_scores else 0,
        "avg_context_tokens_est": round(sum(context_token_estimates) / len(context_token_estimates), 1)
        if context_token_estimates else 0,
    }


# ---------------------------------------------------------------------------
# Grid orchestration
# ---------------------------------------------------------------------------

def grid_point_key(g: dict) -> tuple:
    return (g["chunk_size"], g["chunk_overlap"], g["top_k"])


def load_existing_results(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    return {(r["chunk_size"], r["chunk_overlap"], r["top_k"]): r for r in existing}


def print_comparison(all_results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("GRID EXPERIMENT RESULTS")
    print("=" * 100)
    header = (f"{'Chunk':<8}{'Overlap':<9}{'top_k':<7}{'Top-1 %':<10}{'Top-3 %':<10}"
              f"{'AnswerSim %':<13}{'CtxTokens*':<12}{'Latency (ms)':<14}")
    print(header)
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['chunk_size']:<8}{r['chunk_overlap']:<9}{r['top_k']:<7}"
            f"{r['top1_accuracy_pct']:<10}{r['top3_accuracy_pct']:<10}"
            f"{r['avg_cosine_similarity_pct']:<13}{r['avg_context_tokens_est']:<12}"
            f"{r['avg_latency_ms']:<14}"
        )
    print("-" * 100)
    if all_results:
        best = max(all_results, key=lambda r: (r["top3_accuracy_pct"], r["avg_cosine_similarity_pct"]))
        print(f"Best by Top-3 accuracy / answer similarity: "
              f"chunk_size={best['chunk_size']}, overlap={best['chunk_overlap']}, top_k={best['top_k']}")
    print("=" * 100 + "\n")


def main():
    if not SOURCE_FILES:
        raise RuntimeError("SOURCE_FILES is empty — list the document path(s) to index.")

    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_by_key = load_existing_results(RESULTS_PATH)
    if results_by_key:
        print(f"Found {len(results_by_key)} previously-completed grid point(s) in {RESULTS_PATH}")

    to_run = [g for g in PARAM_GRID if FORCE_RERUN or grid_point_key(g) not in results_by_key]
    skipped = [g for g in PARAM_GRID if grid_point_key(g) in results_by_key and not FORCE_RERUN]
    if skipped:
        print(f"Skipping {len(skipped)} already-completed grid point(s) "
              f"(set FORCE_RERUN=True to redo).")
    if not to_run:
        print("Nothing new to run — all of PARAM_GRID is already in the results file.")
        print_comparison(sorted(results_by_key.values(), key=grid_point_key))
        return

    # Group remaining grid points by (chunk_size, chunk_overlap) so each
    # distinct pair is uploaded only once, then every matching top_k runs
    # against that same upload.
    to_run_sorted = sorted(to_run, key=lambda g: (g["chunk_size"], g["chunk_overlap"], g["top_k"]))

    call_counter = [0]
    budget_hit = False

    for (chunk_size, chunk_overlap), group in itertools.groupby(
        to_run_sorted, key=lambda g: (g["chunk_size"], g["chunk_overlap"])
    ):
        group = list(group)
        print(f"\n=== chunk_size={chunk_size}, chunk_overlap={chunk_overlap} "
              f"(calls used so far today: {call_counter[0]}/{DAILY_CALL_BUDGET}) ===")

        if call_counter[0] >= DAILY_CALL_BUDGET:
            print("  [STOP] Daily budget already exhausted — skipping remaining grid points.")
            budget_hit = True
            break

        try:
            reset_and_upload(chunk_size, chunk_overlap)
        except RuntimeError as e:
            print(f"  [WARN] {e} — skipping all top_k values for this chunk_size/overlap.")
            continue

        for g in group:
            top_k = g["top_k"]
            print(f"\n  --- top_k={top_k} ---")
            try:
                result = run_for_grid_point(questions, chunk_size, chunk_overlap, top_k, call_counter)
            except BudgetExceeded as e:
                print(f"\n  [STOP] {e}")
                print("  Saving completed grid points only; re-run to pick up the rest.")
                budget_hit = True
                break
            results_by_key[grid_point_key(g)] = result
            print(f"  -> Top-1={result['top1_accuracy_pct']}%  Top-3={result['top3_accuracy_pct']}%  "
                  f"AnswerSim={result['avg_cosine_similarity_pct']}%  "
                  f"CtxTokens~{result['avg_context_tokens_est']}  "
                  f"Latency={result['avg_latency_ms']}ms")

        if budget_hit:
            break

    all_results = sorted(results_by_key.values(), key=lambda r: (r["chunk_size"], r["chunk_overlap"], r["top_k"]))
    print_comparison(all_results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {RESULTS_PATH}\n")

    remaining = [g for g in PARAM_GRID if grid_point_key(g) not in results_by_key]
    if remaining or budget_hit:
        print(f"Still need to run: {[grid_point_key(g) for g in remaining] if remaining else '(re-run the point that hit budget)'}")
        print("Just run this script again — completed points are skipped automatically.")
        print("NOTE: after ALL grid points are done, manually re-upload liquidlab_content.txt "
              "at whichever config won, so your live index isn't left on the last-tested value.")


if __name__ == "__main__":
    main()
