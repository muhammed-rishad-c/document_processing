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
attributed to either one specifically. EVAL_TOP_K below defaults to 5,
based on your saved topk_experiment_results.json: top_k=5 scored 59.93%
avg cosine similarity vs 55.57% at top_k=3, without the extra ~1000ms
latency top_k=10 added. Change EVAL_TOP_K if your app uses a different
top_k in production — just keep it fixed across all three chunk-size runs.

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
Fill in SOURCE_FILES below with the paths to the .txt/.pdf files your
eval_dataset.json questions were written against (the same documents you
originally uploaded to build the system).

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
    "eval/the-metamorphosis-franz-kafka-10258.pdf",
]

CHUNK_SIZES = [400]
CHUNK_OVERLAP = 50   # fixed across all runs — see docstring
EVAL_TOP_K = 6        # fixed across all runs — see docstring

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

def run_for_chunk_size(questions: list[dict], chunk_size: int) -> dict:
    latencies = []
    cosine_scores = []
    top1_hits, top3_hits, retrieval_checked = 0, 0, 0

    filename_to_id = get_filename_to_doc_id_map()

    for item in questions:
        category = item.get("category", "uncategorized")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = resolve_expected_sources(
            set(item.get("expected_sources", [])), filename_to_id
        )
        is_unanswerable = item.get("is_unanswerable", category == "unanswerable")

        payload = {"query": question, "top_k": EVAL_TOP_K}

        start = time.perf_counter()
        try:
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=90)
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


def main():
    if not SOURCE_FILES:
        raise RuntimeError(
            "SOURCE_FILES is empty — edit chunk_size_experiment.py and list the "
            "paths to the documents your eval_dataset.json was written against."
        )

    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")

    all_results = []
    for chunk_size in CHUNK_SIZES:
        print(f"\n--- chunk_size={chunk_size} ---")
        reset_and_upload(chunk_size, CHUNK_OVERLAP)

        print(f"  Running eval set at top_k={EVAL_TOP_K}...")
        result = run_for_chunk_size(questions, chunk_size)
        all_results.append(result)
        print(f"  -> Top-1={result['top1_accuracy_pct']}%  Top-3={result['top3_accuracy_pct']}%  "
              f"AnswerSim={result['avg_cosine_similarity_pct']}%  "
              f"Latency={result['avg_latency_ms']}ms")

    print_comparison(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "chunk_size_experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
