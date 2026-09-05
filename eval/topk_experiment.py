"""
topk_experiment.py — Task 6: Top-K Experiment

Runs the full eval_dataset.json through the stateless /documents/chat
endpoint four times, once for each top_k in TOP_K_VALUES, and reports:

  - Top-1 / Top-3 document-level retrieval accuracy (answerable questions)
  - Average answer relevance (cosine similarity vs expected_answer)
  - Average context token count returned per response*
  - Average end-to-end latency

*NOTE: your current RAGResponse/MemoryRAGResponse schema does not return
context token count directly. This script estimates it by counting tokens
in the concatenated chunk_text of the returned sources (using the same
cl100k_base tokenizer your project already uses), which approximates but
does not exactly reproduce build_safe_context()'s internal count (that
function also adds ~200 base tokens + query + chat history tokens on top
of chunk text, which this script does not have access to over the API).
Treat this as a directional estimate, not an exact figure — if you need
the exact number, it would require exposing it in the API response.

Run from the project root:
    python eval/topk_experiment.py
"""

import json
import time
from pathlib import Path

import numpy as np
import requests
import tiktoken
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASE_URL = "http://localhost:9000"
DOCUMENTS_ENDPOINT = f"{BASE_URL}/documents"
CHAT_ENDPOINT = f"{BASE_URL}/documents/chat"
TOP_K_VALUES = [4,5,6,7]

# openrouter/free auto-routes across community-hosted models that commonly
# rate-limit under rapid sequential requests. This script makes 25 x 4 = 100
# calls in one run, so pacing matters more here than in a single-pass script.
REQUEST_DELAY_SECONDS = 1.5
BETWEEN_PASS_DELAY_SECONDS = 5  # extra breathing room when switching top_k
MAX_RETRIES = 3

tokenizer = tiktoken.get_encoding("cl100k_base")

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
    document UUIDs (reassigned on every re-upload). Resolves the CURRENT
    document_id for each filename from what's actually indexed right now."""
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()
    return {d["filename"]: str(d["id"]) for d in docs}


def resolve_expected_sources(expected_sources: set[str], filename_to_id: dict[str, str]) -> set[str]:
    """Converts a question's expected_sources (filenames) into current
    document_ids. Unrecognized filenames are dropped with a warning."""
    resolved = set()
    for name in expected_sources:
        doc_id = filename_to_id.get(name)
        if doc_id:
            resolved.add(doc_id)
        else:
            print(f"    [WARN] expected_sources filename '{name}' not found among "
                  f"currently uploaded documents — is it uploaded?")
    return resolved


def post_with_retry(url: str, payload: dict, timeout: int) -> requests.Response:
    """POSTs with retry-with-backoff on HTTP 429 and transient network
    errors. Respects a Retry-After header if present; otherwise backs off
    exponentially. Raises on the final attempt so the caller's existing
    except/continue logic still applies."""
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_exception = e
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"    [WARN] request error ({e}); retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        if response.status_code == 429:
            if attempt == MAX_RETRIES:
                return response
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2 ** attempt
            print(f"    [WARN] 429 rate-limited; retrying in {wait:.1f}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
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


def run_for_topk(questions: list[dict], top_k: int) -> dict:
    """Runs the full eval set once at a given top_k and returns aggregate stats."""
    latencies = []
    cosine_scores = []
    context_token_estimates = []
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

        payload = {"query": question, "top_k": top_k}

        start = time.perf_counter()
        try:
            response = post_with_retry(CHAT_ENDPOINT, payload, timeout=90)
        except requests.RequestException as e:
            print(f"    [WARN] request failed after {MAX_RETRIES} attempts for "
                  f"'{question[:40]}...': {e}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            print(f"    [WARN] HTTP {response.status_code} for '{question[:40]}...'")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        latencies.append(elapsed_ms)
        data = response.json()
        generated_answer = data.get("answer", "")
        sources = data.get("sources", [])
        retrieved_doc_ids = [s.get("document_id", "") for s in sources]

        # Estimate context tokens from returned chunk text (see module docstring
        # for why this is an approximation, not an exact match to build_safe_context).
        chunk_texts = [s.get("chunk_text", "") for s in sources if "chunk_text" in s]
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

        time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "top_k": top_k,
        "questions_run": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "top1_accuracy_pct": round(100 * top1_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "top3_accuracy_pct": round(100 * top3_hits / retrieval_checked, 2) if retrieval_checked else 0,
        "avg_cosine_similarity_pct": round(100 * sum(cosine_scores) / len(cosine_scores), 2) if cosine_scores else 0,
        "avg_context_tokens_est": round(sum(context_token_estimates) / len(context_token_estimates), 1)
        if context_token_estimates else 0,
    }


def print_comparison(all_results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("TOP-K EXPERIMENT RESULTS (Task 6)")
    print("=" * 90)
    header = f"{'top_k':<8}{'Top-1 %':<10}{'Top-3 %':<10}{'Answer Sim %':<15}{'Ctx Tokens*':<14}{'Latency (ms)':<14}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['top_k']:<8}{r['top1_accuracy_pct']:<10}{r['top3_accuracy_pct']:<10}"
            f"{r['avg_cosine_similarity_pct']:<15}{r['avg_context_tokens_est']:<14}{r['avg_latency_ms']:<14}"
        )
    print("-" * 90)
    print("*Ctx Tokens is an estimate from returned chunk_text only — see script")
    print(" docstring for why it won't exactly match build_safe_context()'s count.")
    print("=" * 90 + "\n")


def main():
    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")

    all_results = []
    for top_k in TOP_K_VALUES:
        print(f"\nRunning eval set at top_k={top_k}...")
        result = run_for_topk(questions, top_k)
        all_results.append(result)
        print(f"  -> Top-1={result['top1_accuracy_pct']}%  Top-3={result['top3_accuracy_pct']}%  "
              f"AnswerSim={result['avg_cosine_similarity_pct']}%  "
              f"CtxTokens~{result['avg_context_tokens_est']}  "
              f"Latency={result['avg_latency_ms']}ms")
        print(f"  Pausing {BETWEEN_PASS_DELAY_SECONDS}s before next top_k pass...")
        time.sleep(BETWEEN_PASS_DELAY_SECONDS)

    print_comparison(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "topk_experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
