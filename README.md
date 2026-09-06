# Mini RAG + Chatbot System — LiquidLab

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot: upload a document, ask questions about it, get answers grounded in retrieved chunks. This README covers the system architecture plus the Day 3 evaluation & optimization work (eval dataset, retrieval/answer scoring, hallucination testing, chunking/top-k experiments, token/cost tracking, performance profiling, and the improvements applied as a result).

---

## 1. Tech Stack

| Layer | Tool |
|---|---|
| API framework | FastAPI + Starlette |
| Relational DB | PostgreSQL (SQLAlchemy ORM) |
| Vector DB | Qdrant (`document_chunks` collection, 384-dim, cosine distance) |
| Embedding model | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Tokenizer | `tiktoken`, encoding `cl100k_base` (BPE) — used consistently for chunking, token counting, and context-token estimation |
| PDF/text extraction | `pymupdf` (PDFs), raw UTF-8 decode (`.txt`) |
| LLM | OpenRouter, model `openrouter/free` (auto-routed across community-hosted models), temperature=0.3 |
| Async task queue (optional) | RQ + Redis, with automatic fallback to FastAPI `BackgroundTasks` if Redis is unavailable |

---

## 2. Project Structure

```
doc_processor/
├── app/
│   ├── main.py                  # FastAPI app, routes, analytics middleware
│   ├── service.py                # text extraction, doc stats, tiktoken-based chunking
│   ├── vector_store.py            # Qdrant init/upsert/search, embedding calls
│   ├── llm_service.py             # RAG answer generation, context building, token usage
│   ├── analytics.py               # request logging, /analytics aggregation
│   ├── models.py / schemas.py / database.py
│   ├── analytics_log.jsonl        # append-only request/token/timing log
│   └── token_vectors/             # per-document token+vector CSV exports
│
└── eval/
    ├── eval_dataset.json          # 25-question evaluation set (see §3)
    ├── answer_evaluation.py       # Task 3+4: answer quality + hallucination scoring
    ├── chunk_size_experiment.py   # Task 5: chunk-size sweep
    ├── topk_experiment.py         # Task 6: top-k sweep
    └── results/                   # timestamped JSON output from each script above
```

---

## 3. Evaluation Methodology

**Corpus:** `the-metamorphosis-franz-kafka-10258.pdf` (single document).

**Dataset (`eval_dataset.json`):** 25 hand-written questions across five categories:

| Category | Count | Purpose |
|---|---|---|
| Easy | 8 | Single-fact lookup |
| Multi-chunk | 6 | Answer spans content spread across multiple chunks |
| Cross-section | 4 | Requires comparing Part I vs Part II/III content |
| Ambiguous pairs | 4 | Near-duplicate questions testing retrieval precision |
| Unanswerable | 3 | Answer is not in the document — tests hallucination avoidance |

Each entry has a `question`, `expected_answer`, and `expected_sources` (resolved by **filename**, not document UUID — UUIDs are reassigned on every re-upload during experiments, so resolving by filename via `GET /documents` at run time keeps scoring valid across re-uploads).

**Retrieval scoring:** does the retrieved chunk's `document_id` match the expected source, checked at Top-1 and Top-3.

**Answer scoring:** cosine similarity between the generated answer and `expected_answer`, using the same `all-MiniLM-L6-v2` model already used for retrieval embeddings. This measures topical relevance, not verified factual correctness — a fluent, on-topic, wrong answer can still score high, and a correct answer with different phrasing can score low. Every eval run prints the full Q/expected/generated triple so results can be spot-checked by eye, not trusted to the score alone.

**Hallucination scoring:** for the 3 unanswerable questions, the generated answer is checked against a set of refusal phrases (broadened beyond a single exact canned string, since the LLM phrases correct refusals inconsistently — e.g. *"does not specify a particular species"* is a correct refusal even though it isn't the literal canned sentence).

---

## 4. Retrieval Results (Task 2)

**Top-1 / Top-3 document-level retrieval accuracy: 100% in every configuration ever tested** — all three chunk sizes, all four top-k values, across multiple independent runs.

> ⚠️ **Read this as a ceiling effect, not a tuned achievement.** With a single-document corpus, any retrieved chunk is trivially "the right document" — there's no distractor document to filter out. This metric has been maxed out since the first run and does not discriminate between configurations. It would only become meaningful with a multi-document corpus.

---

## 5. Answer Evaluation & Hallucination Testing (Tasks 3–4)

Latest full run (25 questions):

| Metric | Result |
|---|---|
| Answerable questions scored | 22 |
| Average answer relevance (cosine similarity) | 50.82% |
| Retrieval accuracy (doc-level) | 100.0% |
| Hallucination tests run | 3 |
| **Hallucination avoidance rate** | **100.0% (3/3 correctly refused)** |
| Average latency | 7019.0 ms |

**Hallucination avoidance is the standout, unambiguous result** — all 3 unanswerable questions (unnamed insect species, Gregor's exact age, exact wording of the apology letters) were correctly refused. This directly validates the mid-project fix that broadened the refusal-detection patterns beyond a single exact string.

---

## 6. Chunking Experiment (Task 5)

Fixed `top_k=5`, `chunk_overlap=50`. Tested across three independent runs over the project's lifetime:

| Chunk Size | Run 1 | Run 2 | Run 3 | Avg Answer Sim |
|---|---|---|---|---|
| 300 | 56.19% | 51.68% | 54.05%¹ | ~54% |
| **500** | **59.74%** | **62.23%** | **60.44%** | **~60.8%** |
| 800 | 51.98% | 55.29% | 55.43% | ~54.2% |

¹ This run only completed 23/25 questions (2 requests failed and were dropped), so it isn't perfectly comparable to the 25/25 runs — noted here rather than silently averaged in as equivalent.

**500 tokens wins consistently across every run, with the ranking never flipping.** Why:
- **300 tokens underperforms:** too short to hold a complete scene for multi-chunk/cross-section questions — the model gets fragments and has to guess at connective narrative.
- **800 tokens underperforms:** a longer chunk often blends multiple unrelated topics into one embedding, diluting semantic focus even though document-level retrieval still succeeds trivially (see §4's ceiling-effect caveat).

**Decision: `chunk_size = 500`, `chunk_overlap = 50`.** Confidence: high (3 independent confirming runs).

---

## 7. Top-K Experiment (Task 6)

Fixed `chunk_size`, swept `top_k`:

| top_k | Answer Sim % | Latency (ms) |
|---|---|---|
| 1 | 26.56% – 40.42%² | 5612 – 6756 |
| 3 | 55.53% – 55.57% | 6005 – 7898 |
| **5** | **59.93% – 62.57%** | **5749 – 7326** |
| 6³ | 62.57% | 6849 |
| 7³ | 57.74% | 7540 |
| 10 | 62.08% – 63.47% | 6982 – 9349 |

² The wide spread on `top_k=1` across repeated runs of the *same* configuration is LLM-side sampling/routing variance (`openrouter/free` auto-routes across different community models per call, at temperature=0.3), not a retrieval or chunking effect — retrieval accuracy stayed at 100% in both runs.
³ A narrower follow-up sweep (top_k 4/5/6/7) around the original winner. `top_k=6` scored highest in that single run, but on only one run versus `top_k=5`'s multiple confirming runs — see §11 for why `top_k=5` was still chosen.

**top_k=1 is sharply worse than every other value** — with only one retrieved chunk, multi-chunk questions are structurally unanswerable regardless of retrieval quality. Quality rises through k=3 and k=5, then **flattens**: k=10 buys only ~0.4–2 points over k=5 for roughly +25–30% latency and double the context tokens — a poor trade.

**Decision: `top_k = 5`.** Confidence: high — best-replicated config (multiple confirming runs in the same band), and the marginal quality gain from higher k values doesn't clear the noise floor observed in repeated runs of identical configs.

---

## 8. Token & Cost Tracking (Task 7)

`GET /analytics` aggregates a per-request JSONL log (`analytics_log.jsonl`) capturing method, path, status code, latency, and — for the two chat endpoints — input/context/output token counts sourced from the LLM provider's own usage accounting (not estimated).

- **Cost is pinned at $0.00** — the deployed model is OpenRouter's free tier. Per-1K-token rate constants exist in `analytics.py` and are ready to activate with a one-line change if a paid model is used later.
- **Known limitation:** `/analytics` currently aggregates the entire log file for the server's lifetime, with no distinction between real usage and eval-script traffic (each eval run sends dozens to hundreds of requests). A `?since=<timestamp>` filter or log-reset mechanism is recommended before quoting these numbers as representing genuine user traffic.
- **Known limitation:** failed requests (HTTP 500) are logged with a status code but no exception detail — currently a dead end for debugging *why* a request failed. Recommended fix: add an `error_detail` field to `log_request()` on the exception path.

---

## 9. Performance Analysis (Task 8)

Per-stage timing, now confirmed across ~300+ real requests (upgraded from the original n=1 preliminary measurement):

| Pipeline | Stage | Typical share of total time |
|---|---|---|
| Query (`/documents/chat`) | `llm_generation_ms` | **85–98%** |
| Query | `query_embedding_ms` + `vector_search_ms` + `context_prep_ms` | 2–15% combined |
| Upload (`/documents/upload`) | `chunk_embedding_ms` | **~96%** (unbatched, one `encoder.encode()` call per chunk) |
| Upload | `document_processing_ms` | ~4% |

**Bottlenecks identified:**
- **Query pipeline:** LLM generation time dominates overwhelmingly — a network round-trip to an external API will always dominate local embedding/search computation. **Tail latency is a separate concern from the mean:** several requests measured 13,000–20,000 ms in `llm_generation_ms` alone against a typical range of 3,000–8,000 ms, consistent with free-tier model-routing variability. Report median/p90 alongside the mean, not the mean alone.
- **Upload pipeline:** embedding chunks one at a time with no batching is the clear cost center. `sentence-transformers` supports batch encoding (`encoder.encode(list_of_texts)`) — this has already been applied in the new token-vector export path (§10) but **is not yet applied to the main chunk-embedding loop in `/documents/upload`** — flagged as a pending optimization, not yet done.

---

## 10. Improvements Implemented (Task 9)

| # | Problem identified | Change implemented | Why selected | Result |
|---|---|---|---|---|
| 1 | Discarded context-token count in `llm_service.py` (`context_str, _ = build_safe_context(...)`) meant real context-token usage was never visible in `/analytics`. | Captured the value instead of discarding it (`context_str, context_tokens = ...`), returned it in the response dict. | Minimal, additive change; unblocks accurate token/cost reporting (Task 7) and removes the need for `topk_experiment.py`'s separate token *estimate*. | `/analytics` now reports real context-token totals, not estimates. |
| 2 | Overly strict hallucination-refusal detection matched only one exact canned sentence, undercounting genuine refusals phrased differently. | Broadened `REFUSAL_PATTERNS` to a set of equivalent refusal phrases (`"does not specify"`, `"not present in the document"`, etc.). | LLMs don't reliably reproduce an exact canned string even when correctly refusing. | Hallucination avoidance now measured at 100% (3/3) instead of undercounting correct refusals as failures. |
| 3 | Chunking/retrieval configuration was unvalidated defaults, not evidence-based. | Ran repeated chunk-size (300/500/800) and top-k (1/3/5/6/7/10) sweeps across multiple independent runs; selected `chunk_size=500`, `top_k=5` based on consistent winning performance across replications, not a single run. | Single-run comparisons on this pipeline are noisy (LLM-side sampling + free-tier model auto-routing) — replication was necessary before trusting a config change. | Config finalized with documented confidence levels (see §6–7) rather than an unverified guess. |
| 4 | New per-token CSV export feature (word-level tokenization + embedding audit trail) would require hundreds of embedding calls per chunk if run synchronously inline in `/documents/upload`, risking upload-latency regression. | Implemented as an out-of-band task: batched per-chunk token embedding (`get_embeddings_batch`) instead of one-call-per-token, dispatched via an RQ/Redis queue with automatic fallback to FastAPI `BackgroundTasks` if Redis is unavailable. | Matches the Day 3 spec's own suggested improvement category ("Asynchronous processing"); keeps the export from ever blocking or slowing the main upload response path. | Upload response time is unaffected by the new export feature; export runs in a separate worker process when Redis is available, or as a background task otherwise. |

**Pending, not yet implemented** (documented honestly rather than claimed done): batching the *main* chunk-embedding loop in `/documents/upload` (§9); adding `error_detail` to the analytics log for 500s (§8); a `?since=` filter on `/analytics` to separate eval traffic from real usage (§8).

---

## 11. Before vs. After Comparison (Task 10)

Measured directly from the sweeps in §6–7 (isolated single-variable comparisons, not a combined grid search — see caveat below):

**Chunk size (at fixed top_k=5):**

| Metric | Before (naive default: 300) | After (evidence-based: 500) |
|---|---|---|
| Answer relevance (cosine sim) | ~54% | ~60.8% |
| Retrieval accuracy | 100%* | 100%* |
| Latency | ~8567 ms | ~7000–8500 ms |

**Top-k (at fixed chunk size):**

| Metric | Before (naive default: k=1) | After (evidence-based: k=5) |
|---|---|---|
| Answer relevance (cosine sim) | 26.56–40.42% | 59.93–62.57% |
| Retrieval accuracy | 100%* | 100%* |
| Latency | 5612–6756 ms | 5749–7326 ms |

\* See §4 — retrieval accuracy is unchanged because it's ceiling-effected by the single-document corpus, not because the tuning had no retrieval impact.

**Caveat:** chunk_size and top_k were swept independently, one variable at a time, per the Day 3 spec's own methodology (to avoid confounding which variable caused a change). A single confirming run of the *combined* recommended-adjacent config (`chunk_size=400, top_k=6`) was also tested and landed within the same noise band as the chosen `500/5` config (~61.6% vs. ~60.8% average) — not a clear win, and under-replicated relative to `500/5`'s three confirming runs, so `500/5` was kept as the shipped configuration. A full joint grid search across both variables together was not performed and is listed as a next step below.

---

## 12. Known Limitations & Next Steps

- **Retrieval accuracy is not a discriminating metric on this corpus.** A multi-document corpus would be needed to meaningfully evaluate retrieval quality independent of answer quality.
- **LLM-side non-determinism** (`openrouter/free`'s multi-model auto-routing + temperature=0.3) means identical configurations produce different absolute scores run-to-run (observed spread: ~±3–7 points on cosine similarity). Rankings between configurations have held stable across repeated runs; absolute percentages should not be over-trusted from a single run.
- **A joint chunk_size × top_k grid search** has not been performed — only one-at-a-time sweeps, plus a single spot-check of one promising combination.
- **~1–2% request failure rate (HTTP 500)** observed on `/documents/chat` under sustained sweep load, with no captured error detail (see §8, §10 pending items).
- **Main upload chunk-embedding loop remains unbatched** — the batching pattern proven out in the new token-vector export path has not yet been backported to the primary upload flow.
