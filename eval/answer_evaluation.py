"""
answer_evaluation.py — Task 3 (Answer Evaluation) + Task 4 (Hallucination Testing)

Hits the stateless /documents/chat endpoint for every question in
eval_dataset.json, then scores each answer two ways depending on category:

  - Answerable questions: cosine similarity between the generated answer
    and the expected_answer (using the same all-MiniLM-L6-v2 model your
    project already uses for embeddings), plus a document-level retrieval
    check against expected_sources.
  - Unanswerable questions (category == "unanswerable"): checked for the
    exact refusal string your system prompt enforces
    ("I cannot find the answer in the provided document context.").

Run from the project root:
    python eval/answer_evaluation.py

NOTE on scope: cosine similarity is a good proxy for topical *relevance*
(is the answer in the right ballpark), but it can score a fluent, on-topic,
factually wrong answer highly. This script prints the full generated vs.
expected text for every question so you can spot-check correctness by eye
alongside the score — treat the printed cosine number as one signal, not
a verdict.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASE_URL = "http://localhost:9000"
DOCUMENTS_ENDPOINT = f"{BASE_URL}/documents"
CHAT_ENDPOINT = f"{BASE_URL}/documents/chat"  # stateless — no session/history bleed
TOP_K = 3

# openrouter/free auto-routes across community-hosted models that commonly
# rate-limit under rapid sequential requests. A fixed pause between calls
# avoids tripping 429s mid-run and keeps latency numbers uncontended by
# request queuing on the server side.
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3

REFUSAL_STRING = "i cannot find the answer in the provided document context."

# Your system prompt tries to force the LLM to reply with REFUSAL_STRING
# verbatim when it can't answer, but LLMs don't always obey "reply EXACTLY
# with X" to the letter — especially when partial context exists and the
# model explains *why* the specific detail is missing instead of reciting
# the canned line. A correct refusal like "the document does not specify
# a particular species" is still a correct refusal; it just isn't the
# exact string. Checking only the exact string undercounts genuine
# hallucination-avoidance. These additional patterns catch equivalent
# phrasing without requiring an exact match.
REFUSAL_PATTERNS = [
    REFUSAL_STRING,
    "does not specify",
    "does not contain",
    "does not include",
    "does not provide",
    "does not mention",
    "does not name",
    "not present in the document",
    "not specified",
    "not provided",
    "not given",
    "not included",
    "no information",
]


def is_refusal(generated_answer: str) -> bool:
    """True if the answer indicates the info isn't in the document, whether
    via the exact canned string or equivalent phrasing. Only meaningful for
    questions the dataset marks as unanswerable — a match here elsewhere
    would just mean the model was hedging, not necessarily refusing."""
    text = generated_answer.lower()
    return any(pattern in text for pattern in REFUSAL_PATTERNS)

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
    document UUIDs (which are reassigned every time a file is re-uploaded).
    This resolves the CURRENT document_id for each filename by asking the
    server what's actually indexed right now, so scoring stays correct no
    matter how many times documents have been re-uploaded since the
    dataset was written."""
    resp = requests.get(DOCUMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    docs = resp.json()
    return {d["filename"]: str(d["id"]) for d in docs}


def resolve_expected_sources(expected_sources: set[str], filename_to_id: dict[str, str]) -> set[str]:
    """Converts a question's expected_sources (filenames) into the set of
    current document_ids those filenames resolve to. Unrecognized filenames
    are dropped with a warning rather than silently mismatching everything."""
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
    """POSTs with retry-with-backoff on HTTP 429 (rate limited) and on
    transient network errors. Respects a Retry-After header if the server
    sends one; otherwise backs off exponentially. Raises on the final
    attempt so the caller's existing except/continue logic still applies."""
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
                return response  # let caller's status_code check handle it
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
    score = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return float(np.clip(score, 0.0, 1.0))


def run_evaluation(questions: list[dict]) -> dict:
    results = []
    latencies = []
    correctness_scores = []
    retrieval_hits = 0
    retrieval_checked = 0
    hallucination_correct = 0
    hallucination_total = 0

    filename_to_id = get_filename_to_doc_id_map()

    print(f"\nRunning answer evaluation on {len(questions)} questions...")
    print("=" * 78)

    for idx, item in enumerate(questions, 1):
        qid = item["id"]
        category = item.get("category", "uncategorized")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = resolve_expected_sources(
            set(item.get("expected_sources", [])), filename_to_id
        )
        is_unanswerable = item.get("is_unanswerable", category == "unanswerable")

        payload = {"query": question, "top_k": TOP_K}

        start = time.perf_counter()
        try:
            response = post_with_retry(CHAT_ENDPOINT, payload, timeout=60)
        except requests.RequestException as e:
            print(f"[{qid}] REQUEST FAILED after {MAX_RETRIES} attempts: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        if response.status_code != 200:
            print(f"[{qid}] HTTP {response.status_code}: {response.text[:200]}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        data = response.json()
        generated_answer = data.get("answer", "")
        sources = data.get("sources", [])
        retrieved_doc_ids = [s.get("document_id", "") for s in sources]

        result = {
            "id": qid,
            "category": category,
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "latency_ms": round(elapsed_ms, 2),
            "is_unanswerable": is_unanswerable,
        }

        if is_unanswerable:
            hallucination_total += 1
            refused = is_refusal(generated_answer)
            if refused:
                hallucination_correct += 1
            result["correctly_refused"] = refused
            print(f"[{qid}] ({category}) latency={elapsed_ms:.0f}ms  refused={'YES' if refused else 'NO'}")
            print(f"    Q: {question}")
            print(f"    A: {generated_answer[:150]}")
        else:
            score = cosine_similarity(generated_answer, expected_answer)
            correctness_scores.append(score)
            result["cosine_similarity"] = round(score, 4)

            if expected_sources:
                retrieval_checked += 1
                if any(doc_id in expected_sources for doc_id in retrieved_doc_ids):
                    retrieval_hits += 1
                    result["retrieval_correct"] = True
                else:
                    result["retrieval_correct"] = False

            print(f"[{qid}] ({category}) latency={elapsed_ms:.0f}ms  cosine_similarity={score*100:.1f}%")
            print(f"    Q: {question}")
            print(f"    Expected: {expected_answer[:150]}")
            print(f"    Got:      {generated_answer[:150]}")

        print("-" * 78)
        results.append(result)
        time.sleep(REQUEST_DELAY_SECONDS)

    summary = {
        "total_questions": len(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "answerable_count": len(correctness_scores),
        "avg_cosine_similarity_pct": round(100 * sum(correctness_scores) / len(correctness_scores), 2)
        if correctness_scores else 0,
        "retrieval_accuracy_pct": round(100 * retrieval_hits / retrieval_checked, 2)
        if retrieval_checked else 0,
        "hallucination_avoidance_pct": round(100 * hallucination_correct / hallucination_total, 2)
        if hallucination_total else 100,
        "hallucination_total": hallucination_total,
        "hallucination_correct": hallucination_correct,
    }

    return {"results": results, "summary": summary}


def print_summary(evaluation: dict) -> None:
    s = evaluation["summary"]
    print("\n" + "=" * 78)
    print("ANSWER EVALUATION SUMMARY (Task 3 + Task 4)")
    print("=" * 78)
    print(f"Total questions run:              {s['total_questions']}")
    print(f"Average response latency:         {s['avg_latency_ms']} ms")
    print(f"Answerable questions scored:      {s['answerable_count']}")
    print(f"Average answer relevance (cosine):{s['avg_cosine_similarity_pct']}%")
    print(f"Retrieval accuracy (doc-level):    {s['retrieval_accuracy_pct']}%")
    print(f"Hallucination tests run:          {s['hallucination_total']}")
    print(f"Hallucination avoidance rate:      {s['hallucination_avoidance_pct']}% "
          f"({s['hallucination_correct']}/{s['hallucination_total']} correctly refused)")
    print("=" * 78)
    print("NOTE: cosine similarity measures topical relevance, not verified factual")
    print("correctness. Spot-check the printed Q/Expected/Got triples above for any")
    print("question that looks off, especially scores in the 60-85% band.")
    print("=" * 78 + "\n")


def save_results(evaluation: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"answer_eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    return out_path


def main():
    questions = load_dataset(DATASET_PATH)
    print(f"Loaded {len(questions)} questions from {DATASET_PATH}")
    evaluation = run_evaluation(questions)
    print_summary(evaluation)
    saved_path = save_results(evaluation)
    print(f"Full results saved to: {saved_path}\n")


if __name__ == "__main__":
    main()
