"""
chunk_size_experiment.py — Task 5: Chunk-Size Experiment

For each chunk size in CHUNK_SIZES, this script:
  1. Deletes every currently indexed document (GET /documents, then
     DELETE /documents/{id} for each) so chunks from a previous chunk-size
     run don't leak into this one.
  2. Re-uploads every file listed in SOURCE_FILES via POST /documents/upload
     with that chunk_size (chunk_overlap held fixed — see below).
  3. Runs the full eval_dataset.json through /documents/chat at a FIXED
     top_k and measures Top-1 / Top-3 document-level retrieval accuracy
     and answer relevance (cosine similarity vs expected_answer), the same
     way topk_experiment.py measured the top_k sweep.

WHY top_k IS FIXED HERE
------------------------
Task 5 asks "how does chunk size affect quality," in isolation. If top_k
and chunk_size both moved at once, a change in the numbers couldn't be
attributed to either one specifically.

NOTE (company-data run): EVAL_TOP_K below is a placeholder carried over
from the old Metamorphosis-story experiment (top_k=5 scored best there).
That result doesn't transfer to the LiquidLab company-content dataset —
re-run topk_experiment.py FIRST against your currently-indexed
liquidlab_content.txt to find the best top_k for THIS dataset, then set
EVAL_TOP_K here to that value before running this script. Keep it fixed
across all three chunk-size runs below.

WHY chunk_overlap IS FIXED AT 50
----------------------------------
This is your existing /documents/upload default. Task 5's spec only asks
you to test chunk_size (300/500/800), not overlap — sweeping both at once
would confound the results the same way varying top_k would. If you want
to test overlap too, that's a separate follow-on experiment, not this one.

CONTEXT-TOKEN COLUMN
---------------------
Like topk_experiment.py, this script can't report real context-token
counts: /documents/chat's response (ChunkSource) doesn't return chunk_text,
so there's nothing here to count tokens from. That gap gets closed in
Task 7 by adding an optional field to the response — this script will
report 0 for avg_context_tokens_est until then, matching your existing
topk_experiment_results.json.

DESTRUCTIVE WARNING
--------------------
This script deletes and re-uploads every document in your Postgres +
Qdrant store, once per chunk size (3 times total). Only point it at a
local/dev database you're fine wiping and re-seeding. It does not touch
any FastAPI/service code — it only calls your existing HTTP endpoints.

SETUP
-----
SOURCE_FILES below points at eval/liquidlab_content.txt — put a copy of
that file in the eval/ folder (same folder as this script) before running.
eval_dataset.json's expected_sources must match the filename you upload
here EXACTLY (see get_filename_to_doc_id_map / resolve_expected_sources
above) — the current eval_dataset.json already expects "liquidlab_content.txt".

Run from the project root:
    python eval/chunk_size_experiment.py
"""

import json
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — EDIT THESE
# ---------------------------------------------------------------------------

# Paths to the source documents your eval_dataset.json questions reference.
# These get deleted and re-uploaded fresh for every chunk size below.
SOURCE_FILES: list[str] = [
    "eval/liquidlab_content.txt",
]

CHUNK_SIZES = [400, 500, 600]
CHUNK_OVERLAP = 50   # fixed across all runs — see docstring

# PLACEHOLDER — see "NOTE (company-data run)" in the module docstring above.
# Run topk_experiment.py against the currently-indexed liquidlab_content.txt
# first, then replace this with the top_k that scored best there.
EVAL_TOP_K = 6        # fixed across all runs — see docstring

# ---------------------------------------------------------------------------
# MULTI-DAY BUDGET (OpenRouter free tier: 50 requests/day)
# ---------------------------------------------------------------------------
# Each chunk size costs 25 LLM calls (one per question). EDIT THIS LIST
# before each run: pick 1-2 not-yet-run values from CHUNK_SIZES above
# (2 x 25 = 50 = the daily cap exactly; DAILY_CALL_BUDGET below trims a
# safety buffer off that). Results accumulate in
# results/chunk_size_experiment_results.json across runs/days —
# already-completed chunk sizes are skipped automatically unless
# FORCE_RERUN=True.
RUN_TODAY = [400, 500]

# Hard stop once this many LLM calls have been made in this invocation, even
# mid-value, leaving headroom under the 50/day cap for retries. A chunk size
# interrupted partway is discarded and retried in full next time it's in
# RUN_TODAY — its documents have already been re-uploaded by then anyway.
DAILY_CALL_BUDGET = 45

FORCE_RERUN = False  # set True to re-run a chunk size already in the results file

# ---------------------------------------------------------------------------
# Config — shouldn't need to change
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASE_URL = "http://localhost:9000"
DOCUMENTS_ENDPOINT = f"{BASE_URL}/documents"
UPLOAD_ENDPOINT = f"{BASE_URL}/documents/upload"
CHAT_ENDPOINT = f"{BASE_URL}/documents/chat"

print("Loading sentence-transformer model for cosine similarity scoring...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def get_filename_to_doc_id_map() -> dict[str, str]:
    """eval_dataset.json's expected_sources holds filenames (stable), not
    document UUIDs. This script assigns a brand-new UUID to each document
    on every re-upload, so expected_sources must be resolved against
    whatever is CURRENTLY indexed, not a value baked into the JSON."""
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()
    return {d["filename"]: str(d["id"]) for d in docs}


def resolve_expected_sources(expected_sources: set[str], filename_to_id: dict[str, str]) -> set[str]:
    """Converts a question's expected_sources (filenames) into the current
    document_ids those filenames resolve to. Unrecognized filenames are
    dropped with a warning rather than silently mismatching everything."""
    resolved = set()
    for name in expected_sources:
        doc_id = filename_to_id.get(name)
        if doc_id:
            resolved.add(doc_id)
        else:
            print(f"    [WARN] expected_sources filename '{name}' not found among "
                  f"currently uploaded documents — is it uploaded?")
    return resolved


class BudgetExceeded(Exception):
    """Raised when DAILY_CALL_BUDGET is hit mid-run, so main() can stop
    cleanly and save whatever full chunk sizes already completed."""
    pass


MAX_RETRIES = 3


def post_with_retry(url: str, payload: dict, timeout: int) -> requests.Response:
    """429/network-aware retry, matching topk_experiment.py. This script
    previously had no retry logic at all, so a single rate-limit response
    would kill the whole run instead of backing off."""
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


# ---------------------------------------------------------------------------
# Document reset / re-upload
# ---------------------------------------------------------------------------

def clear_all_documents() -> int:
    """Deletes every document currently in the store. Returns count deleted."""
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
    """Uploads every file in SOURCE_FILES at the given chunk_size/overlap.
    Returns count successfully uploaded."""
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
# Eval run for a single chunk size
# ---------------------------------------------------------------------------

def run_for_chunk_size(questions: list[dict], chunk_size: int, call_counter: list[int]) -> dict:
    latencies = []
    cosine_scores = []
    top1_hits, top3_hits, retrieval_checked = 0, 0, 0

    filename_to_id = get_filename_to_doc_id_map()

    for item in questions:
        if call_counter[0] >= DAILY_CALL_BUDGET:
            raise BudgetExceeded(
                f"Hit DAILY_CALL_BUDGET ({DAILY_CALL_BUDGET}) partway through chunk_size={chunk_size} "
                f"({len(latencies)}/{len(questions)} questions done for this value)."
            )

        category = item.get("category", "uncategorized")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = resolve_expected_sources(
            set(item.get("expected_sources", [])), filename_to_id
        )
        is_unanswerable = item.get("is_unanswerable", category == "unanswerable")

        payload = {"query": question, "top_k": EVAL_TOP_K}
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
        "chunk_overlap": CHUNK_OVERLAP,
        "eval_top_k": EVAL_TOP_K,
        "questions_run": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "top1_accuracy_pct": round(100 * top1_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "top3_accuracy_pct": round(100 * top3_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "avg_cosine_similarity_pct": round(100 * sum(cosine_scores) / len(cosine_scores), 2) if cosine_scores else 0,
        # See module docstring — real per-request context-token counts require
        # a Task 7 schema change; left here for a consistent table shape with
        # topk_experiment_results.json until that's in place.
        "avg_context_tokens_est": 0,
    }


def print_comparison(all_results: list[dict]) -> None:
    print("\n" + "=" * 95)
    print(f"CHUNK-SIZE EXPERIMENT RESULTS (Task 5)  [fixed top_k={EVAL_TOP_K}, overlap={CHUNK_OVERLAP}]")
    print("=" * 95)
    header = (f"{'Chunk Size':<12}{'Top-1 %':<10}{'Top-3 %':<10}"
              f"{'Answer Sim %':<15}{'Latency (ms)':<14}")
    print(header)
    print("-" * 95)
    for r in all_results:
        print(
            f"{r['chunk_size']:<12}{r['top1_accuracy_pct']:<10}{r['top3_accuracy_pct']:<10}"
            f"{r['avg_cosine_similarity_pct']:<15}{r['avg_latency_ms']:<14}"
        )
    print("-" * 95)
    best = max(all_results, key=lambda r: (r["top3_accuracy_pct"], r["avg_cosine_similarity_pct"]))
    print(f"Best by Top-3 accuracy / answer similarity: chunk_size={best['chunk_size']}")
    print("=" * 95 + "\n")


def load_existing_results(path: Path) -> dict[int, dict]:
    """Loads previously-saved results (from earlier days), keyed by
    chunk_size, so today's run only adds to them instead of overwriting."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    return {r["chunk_size"]: r for r in existing}


def main():
    if not SOURCE_FILES:
        raise RuntimeError(
            "SOURCE_FILES is empty — edit chunk_size_experiment.py and list the "
            "paths to the documents your eval_dataset.json was written against."
        )

    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "chunk_size_experiment_results.json"
    results_by_size = load_existing_results(out_path)
    if results_by_size:
        print(f"Found {len(results_by_size)} previously-completed chunk size(s) "
              f"in {out_path}: {sorted(results_by_size.keys())}")

    to_run = [s for s in RUN_TODAY if FORCE_RERUN or s not in results_by_size]
    skipped = [s for s in RUN_TODAY if s in results_by_size and not FORCE_RERUN]
    if skipped:
        print(f"Skipping already-completed values (set FORCE_RERUN=True to redo): {skipped}")
    if not to_run:
        print("Nothing new to run today — all of RUN_TODAY is already in the results file.")
        print_comparison(sorted(results_by_size.values(), key=lambda r: r["chunk_size"]))
        return

    call_counter = [0]
    budget_hit = False
    for chunk_size in to_run:
        print(f"\n--- chunk_size={chunk_size}  (calls used so far today: {call_counter[0]}/{DAILY_CALL_BUDGET}) ---")
        reset_and_upload(chunk_size, CHUNK_OVERLAP)

        print(f"  Running eval set at top_k={EVAL_TOP_K}...")
        try:
            result = run_for_chunk_size(questions, chunk_size, call_counter)
        except BudgetExceeded as e:
            print(f"\n[STOP] {e}")
            print("Saving completed values only; re-run tomorrow to pick up the rest.")
            budget_hit = True
            break
        results_by_size[chunk_size] = result
        print(f"  -> Top-1={result['top1_accuracy_pct']}%  Top-3={result['top3_accuracy_pct']}%  "
              f"AnswerSim={result['avg_cosine_similarity_pct']}%  "
              f"Latency={result['avg_latency_ms']}ms")

    all_results = sorted(results_by_size.values(), key=lambda r: r["chunk_size"])
    print_comparison(all_results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {out_path}\n")

    remaining = [s for s in CHUNK_SIZES if s not in results_by_size]
    if remaining or budget_hit:
        print(f"Still need to run: {remaining if remaining else '(re-run the value that hit budget)'}")
        print("Set RUN_TODAY to the next 1-2 values and run again tomorrow.")
        print("NOTE: after ALL chunk sizes are done, manually re-upload liquidlab_content.txt "
              "at whichever chunk_size won, so your live index isn't left on the last-tested value.")


if __name__ == "__main__":
    main()
