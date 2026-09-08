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
TOP_K_VALUES = [4, 5, 6, 7, 8]

REQUEST_DELAY_SECONDS = 1.5
BETWEEN_PASS_DELAY_SECONDS = 5
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
    """Resolves current document_id for each indexed filename."""
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()
    return {d["filename"]: str(d["id"]) for d in docs}


def resolve_expected_sources(expected_sources: set[str], filename_to_id: dict[str, str]) -> set[str]:
    """Converts expected filenames to active document UUIDs."""
    resolved = set()
    for name in expected_sources:
        doc_id = filename_to_id.get(name)
        if doc_id:
            resolved.add(doc_id)
        else:
            print(f"    [WARN] expected source '{name}' not found in index.")
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


def run_for_topk(questions: list[dict], top_k: int) -> dict:
    latencies = []
    cosine_scores = []
    context_token_estimates = []
    doc_hit_count, retrieval_checked = 0, 0

    filename_to_id = get_filename_to_doc_id_map()

    for item in questions:
        category = item.get("category", "uncategorized")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = resolve_expected_sources(
            set(item.get("expected_sources", [])), filename_to_id
        )
        is_unanswerable = category == "unanswerable" or not expected_sources

        payload = {"query": question, "top_k": top_k}

        start = time.perf_counter()
        try:
            response = post_with_retry(CHAT_ENDPOINT, payload, timeout=90)
        except requests.RequestException as e:
            print(f"    [WARN] Request failed for '{question[:30]}...': {e}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code != 200:
            print(f"    [WARN] HTTP {response.status_code} for '{question[:30]}...'")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        latencies.append(elapsed_ms)
        data = response.json()
        generated_answer = data.get("answer", "")
        sources = data.get("sources", [])
        retrieved_doc_ids = [s.get("document_id", "") for s in sources]

        # Estimate context tokens from returned chunk text
        chunk_texts = [s.get("chunk_text", "") for s in sources if "chunk_text" in s]
        if chunk_texts:
            context_token_estimates.append(sum(len(tokenizer.encode(t)) for t in chunk_texts))

        # Track document hit rate (Will be ~100% for single-file dataset)
        if not is_unanswerable and expected_sources:
            retrieval_checked += 1
            if any(doc_id in expected_sources for doc_id in retrieved_doc_ids):
                doc_hit_count += 1

        # Cosine similarity evaluation for ALL questions (including unanswerables)
        cosine_scores.append(cosine_similarity(generated_answer, expected_answer))

        time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "top_k": top_k,
        "questions_run": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "doc_retrieval_rate_pct": round(100 * doc_hit_count / retrieval_checked, 2) if retrieval_checked else 0,
        "avg_cosine_similarity_pct": round(100 * sum(cosine_scores) / len(cosine_scores), 2) if cosine_scores else 0,
        "avg_context_tokens_est": round(sum(context_token_estimates) / len(context_token_estimates), 1)
        if context_token_estimates else 0,
    }


def print_comparison(all_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("TOP-K EXPERIMENT RESULTS — COMPANY DATASET")
    print("=" * 80)
    header = f"{'top_k':<8}{'Doc Hit %':<12}{'Answer Sim %':<16}{'Ctx Tokens*':<15}{'Latency (ms)':<14}"
    print(header)
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['top_k']:<8}{r['doc_retrieval_rate_pct']:<12}"
            f"{r['avg_cosine_similarity_pct']:<16}{r['avg_context_tokens_est']:<15}{r['avg_latency_ms']:<14}"
        )
    print("-" * 80)


def main():
    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")

    all_results = []
    for top_k in TOP_K_VALUES:
        print(f"\nRunning eval set at top_k={top_k}...")
        result = run_for_topk(questions, top_k)
        all_results.append(result)
        print(f"  -> AnswerSim={result['avg_cosine_similarity_pct']}%  "
              f"CtxTokens~{result['avg_context_tokens_est']}  "
              f"Latency={result['avg_latency_ms']}ms")
        time.sleep(BETWEEN_PASS_DELAY_SECONDS)

    print_comparison(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "topk_experiment_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()