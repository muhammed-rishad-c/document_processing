# Mini RAG + Chatbot System — LiquidLab

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot: upload a document, ask questions about it, get answers grounded in retrieved chunks. This README covers the system architecture, the exact end-to-end pipeline (tokenization, chunking, embedding, retrieval, generation), the summary-query classification fix, the widget lead-capture/department-routing feature, and the Day 3 evaluation & optimization work.

---

## 1. Tech Stack

| Layer                       | Tool                                                                                                                                                                      | Notes                                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| API framework               | FastAPI + Starlette                                                                                                                                                       | —                                                                                                                     |
| Relational DB               | PostgreSQL (SQLAlchemy ORM)                                                                                                                                               | Documents, chunks, chat sessions/messages, companies, departments, leads                                               |
| Vector DB                   | Qdrant, collection`document_chunks`                                                                                                                                     | 384-dim vectors, cosine distance                                                                                       |
| Embedding model             | `sentence-transformers/all-MiniLM-L6-v2`, loaded via `langchain_huggingface.HuggingFaceEmbeddings` from a local model directory                                       | Same model used for query embedding, chunk embedding, and eval answer-similarity scoring                               |
| Vector search wrapper       | `langchain_qdrant.QdrantVectorStore` (`similarity_search_with_score_by_vector`)                                                                                       | Lazy singleton, created on first search call                                                                           |
| Tokenizer                   | `tiktoken`, encoding `cl100k_base` (BPE)                                                                                                                              | Used consistently for chunk splitting, token counting, history-budget checks, and context-budget checks                |
| Text splitter               | `langchain_text_splitters.RecursiveCharacterTextSplitter.from_tiktoken_encoder`                                                                                         | Token-aware splitting (length measured in tokens, not characters), separator priority`["\n\n", "\n", ". ", " ", ""]` |
| PDF/text extraction         | `pymupdf` (`fitz`) for `.pdf`, raw UTF-8 decode for `.txt`                                                                                                        | Null bytes stripped from extracted text                                                                                |
| LLM (primary)               | Local LM Studio server, model`gemma-4-e4b-it` (configurable via `LLM_MODEL_NAME`), via `langchain_openai.ChatOpenAI` pointed at an OpenAI-compatible local endpoint | temperature=0.3, 30s timeout, no retries                                                                               |
| LLM (fallback)              | OpenRouter, model`openrouter/free` (auto-routed across community-hosted models)                                                                                         | temperature=0.3, 20s timeout, 1 retry — wired via LangChain's`.with_fallbacks([...])`                               |
| Async task queue (optional) | RQ + Redis, with automatic fallback to FastAPI`BackgroundTasks` if Redis is unavailable                                                                                 | Used for the per-chunk token-sequence CSV export                                                                       |
| Email delivery              | `smtplib` + `email.mime` (stdlib, no new dependency)                                                                                                                  | STARTTLS by default; dev/test uses a Mailtrap sandbox inbox                                                            |

---

## 2. Project Structure

```
doc_processor/
├── app/
│   ├── main.py                  # FastAPI app, routes, analytics middleware
│   ├── service.py               # text extraction, doc stats, tiktoken-based chunking, token counting
│   ├── vector_store.py          # Qdrant init/upsert/search, embedding calls (single + batch)
│   ├── llm_service.py           # RAG answer generation, summary-query classification, lead extraction,
│   │                             #   department classification, context building, token usage
│   ├── analytics.py             # request logging, /analytics aggregation
│   ├── widget.py                # public widget API: /widget/session, /widget/chat — RAG flow +
│   │                             #   lead-capture + department classification
│   ├── internal.py              # admin-only API (shared-secret auth): create/list companies,
│   │                             #   add departments, download leads
│   ├── email_service.py         # SMTP lead-notification email, sent to the resolved department address
│   ├── lead_export.py           # per-company Excel lead export (append_lead())
│   ├── rate_limit.py            # slowapi limiter, keyed by API key / session id / IP
│   ├── models.py / schemas.py / database.py
│   ├── analytics_log.jsonl      # append-only request/token/timing log
│   └── token_vectors/           # per-document token+vector CSV exports
│
└── eval/
    ├── eval_dataset.json          # 25-question evaluation set (see §6)
    ├── answer_evaluation.py       # answer quality + hallucination scoring
    ├── chunk_size_experiment.py   # chunk-size sweep (single-variable, §9)
    ├── topk_experiment.py         # top-k sweep (single-variable, §10)
    ├── grid_experiment.py         # joint chunk_size x chunk_overlap x top_k grid (§11); re-indexes
    │                               #   once per (chunk_size, chunk_overlap) pair, resumable across
    │                               #   runs via results/grid_experiment_results.json
    └── results/                   # timestamped JSON output from each script above
```

---

## 3. End-to-End Pipeline: How an Answer Is Generated

This section walks through exactly what happens, function by function, with the actual parameters used in the code — split into the **upload/ingestion** path (runs once per document) and the **chat/answer** path (runs once per message).

### 3.1 Upload & Ingestion (`POST /documents/upload`)

| Step                                     | Function                                                                                                                                           | Library                                                                               | Parameters / detail                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Extract text                          | `extract_text_from_file(file_bytes, filename)` (`service.py`)                                                                                  | `pymupdf` (`fitz`) for `.pdf`; raw decode for `.txt`                          | PDFs:`fitz.open(stream=..., filetype="pdf")`, text pulled page-by-page via `page.get_text("text")`. `.txt`: `file_bytes.decode("utf-8", errors="ignore")`. Null bytes stripped from output either way. Raises `ValueError` if the result is empty/unreadable or the extension isn't `.txt`/`.pdf`.                                                                 |
| 2. Compute document stats                | `calculate_document_stats(text)` (`service.py`)                                                                                                | stdlib`re`, `collections.Counter`                                                 | Sentence split on`[.!?]+`, paragraph split on `\n`, words via `\b\w+\b` regex (a lightweight regex tokenizer — **not** the tiktoken tokenizer, used only for display stats). Stopword-filtered top-10 word frequency.                                                                                                                                               |
| 3. Chunk the text                        | `chunk_text(text, max_chunk_size=600, chunk_overlap=50)` (`service.py`)                                                                        | `langchain_text_splitters.RecursiveCharacterTextSplitter.from_tiktoken_encoder`     | `encoding_name="cl100k_base"`, `chunk_size=600`, `chunk_overlap=50`, `separators=["\n\n", "\n", ". ", " ", ""]`. Splitting is **token-aware** (length measured in tiktoken tokens, not raw characters), tried in separator-priority order. Values chosen from the evidence-based joint grid search in §11 (superseding the earlier single-variable sweep in §9). |
| 4. Count tokens per chunk                | `count_token(text)` (`service.py`)                                                                                                             | `tiktoken.get_encoding("cl100k_base")`                                              | `len(tokenizer.encode(text))`. This exact tokenizer/encoding is reused everywhere token budgets matter (chunking, history reduction, context assembly) — one source of truth, no drift between subsystems.                                                                                                                                                                    |
| 5. Embed chunks (batch)                  | `get_embeddings_batch(chunk_texts)` (`vector_store.py`)                                                                                        | `langchain_huggingface.HuggingFaceEmbeddings` → `encoder.embed_documents(texts)` | Model:`all-MiniLM-L6-v2`, loaded from a local model directory (`MODEL_PATH`). Output: 384-dim vectors, one per chunk, **order-preserving** (safe to `zip()` against the input chunk list).                                                                                                                                                                           |
| 6. Persist chunks (Postgres)             | ORM insert of`DocumentChunk` rows                                                                                                                | SQLAlchemy                                                                            | `id`, `document_id`, `chunk_index`, `chunk_text`, `token_count` per row.                                                                                                                                                                                                                                                                                               |
| 7. Persist vectors (Qdrant)              | `store_chunk_vector(vector_data)` (`vector_store.py`)                                                                                          | `qdrant_client`                                                                     | One`PointStruct` per chunk: `vector=embedding`, `payload={page_content, metadata: {document_id, chunk_index, token_count, chunk_id}}`. Written via `qdrant.upsert(collection_name="document_chunks", points=[...])`.                                                                                                                                                     |
| 8. (Background) token-sequence audit CSV | `generate_chunk_token_sequence_csv(chunks, output_path)` (`service.py`), dispatched via FastAPI `BackgroundTasks` (or RQ/Redis if available) | `tiktoken`, stdlib `csv`                                                          | Per chunk: encodes every token individually (`tokenizer.encode` → per-token `tokenizer.decode([token_id])`), deduplicates by `token_id`, writes `chunk_index, position, token_id, token_text` rows to CSV. Debugging/audit artifact only — never blocks the upload response.                                                                                           |

**Qdrant collection setup** (`vector_store.py`, runs once at import time via `init_qdrant()`): creates the `document_chunks` collection if it doesn't already exist, with `VectorParams(size=384, distance=Distance.COSINE)`. A `QdrantVectorStore` (LangChain wrapper around the same client/collection/embedding function) is created lazily on first search, not at import time, specifically so it never races `init_qdrant()`'s collection creation during FastAPI startup.

### 3.2 Chat / Answer Generation (`POST /documents/chat-memory`, `POST /widget/chat`)

| Step                                           | Function                                                                                    | Library                                                                         | Parameters / detail                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Resolve the retrieval query                 | `classify_summary_query(query)` (`llm_service.py`)                                      | stdlib`re`                                                                    | Decides whether the query is summary-flavored, and if so generic vs. targeted — see §5. Non-summary queries pass through unchanged.                                                                                                                                                                                                                       |
| 2. Embed the query                             | `get_embedding(text)` (`vector_store.py`) → `encoder.embed_query(text)`              | `HuggingFaceEmbeddings`, same `all-MiniLM-L6-v2` model                      | Single 384-dim vector.                                                                                                                                                                                                                                                                                                                                      |
| 3. Vector search                               | `search_similar_chunks(query_text, top_k, document_id, timing_out)` (`vector_store.py`) | `QdrantVectorStore.similarity_search_with_score_by_vector`                    | `k=top_k` (default 7, see §11, superseding the earlier single-variable sweep in §10), optional `Filter(FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id)))` to scope search to one document/tenant. Per-stage timings (`query_embedding_ms`, `vector_search_ms`) captured for analytics.                             |
| 4. Reduce chat history                         | `reduce_chat_history(chat_history, max_history_tokens=1200)` (`llm_service.py`)         | `count_token` (tiktoken)                                                      | If total history tokens ≤ 1200, passed through unchanged. Otherwise: last 4 turns kept verbatim, everything older condensed into one`[Prior Conversation Summary Block]` (first 120 chars per turn).                                                                                                                                                     |
| 5. Assemble context                            | `build_safe_context(retrieved_chunks, query_text, chat_history)` (`llm_service.py`)     | `count_token` (tiktoken)                                                      | Greedily packs`--- Chunk N (Doc ID: ...) ---`-formatted chunks into a budget of `MAX_CONTEXT_TOKENS = 4000` (query + history tokens counted first), stopping before the first chunk that would overflow the budget.                                                                                                                                     |
| 6. Select system prompt                        | `classify_summary_query(query)["is_summary"]` (`llm_service.py`)                        | —                                                                              | Chooses between the normal formatting prompt and the summary-flavored formatting prompt (both cap bulleted answers at 6 items; the summary prompt is tuned for broader, less follow-up-anchored answers). See §5.                                                                                                                                          |
| 7. Generate the answer                         | `rag_chain.invoke({...})` (`llm_service.py`)                                            | `langchain_core.prompts.ChatPromptTemplate` + `langchain_openai.ChatOpenAI` | Prompt:`system_prompt` + `MessagesPlaceholder("chat_history")` (real `HumanMessage`/`AIMessage` objects, never string-templated — avoids brace-escaping issues) + `user_query`. Model: `primary_llm.with_fallbacks([fallback_llm])` — local `gemma-4-e4b-it` first, OpenRouter `openrouter/free` on failure. Both at `temperature=0.3`. |
| 8. Post-process the raw output                 | inline in`generate_rag_answer_with_memory` (`llm_service.py`)                           | stdlib`re`                                                                    | Strips stray`"User Safety:"` / `"Response Safety:"` lines; strips a leading "based on the document..."-style lead-in via `LEAD_IN_PATTERN`; caps bulleted answers at 6 items with a truncation note; falls back to the exact string `NO_ANSWER_TEXT` if nothing usable remains.                                                                     |
| 9. Fallback → lead capture (widget path only) | `resolve_standalone_query` → `classify_query` → lead-capture state machine            | LLM-backed,`llm_service.py` / `widget.py`                                   | Only triggers on an**exact match** to `NO_ANSWER_TEXT`. Documented in full in §4.                                                                                                                                                                                                                                                                  |

Both the visitor's message and the generated answer are persisted as `ChatMessage` rows regardless of which branch was taken, and per-stage timings are logged for `/analytics` either way.

---

## 4. Widget Lead Capture & Dynamic Department Email Routing

When the chatbot can't answer a visitor's question (RAG returns `NO_ANSWER_TEXT`), it captures the visitor as a lead and routes a notification email to the correct company department automatically — instead of every unanswered question going to one inbox.

**Company-defined departments, not a fixed category list.** Each `Company` owns a set of `CompanyDepartment` rows (name, email, `is_default`, `is_active` — max 10 per company, admin-created). Exactly one department per company must be marked default, enforced by a partial unique DB index (`uq_company_departments_one_default`) — not just application code — so a company can never end up with zero or multiple defaults. This guarantees a captured lead always has somewhere to go, even if classification fails or is ambiguous.

**Flow, end to end:**

1. RAG fails → exact-match on `NO_ANSWER_TEXT` (the only detection point anywhere in the app) triggers lead capture.
2. At that exact moment, `classify_query()` (`llm_service.py`) asks the LLM to match the visitor's original question against this company's own active department names — returning one of those exact names, or `None` if nothing clearly fits. The raw LLM output is never trusted directly: it's matched case-insensitively against the real department list before being accepted.
3. The classified name rides along in the same pending-lead JSON blob (`session.pending_lead_query`) that already tracks the visitor's in-progress name/email/phone, so it survives across multiple back-and-forth turns without reclassifying.
4. Once name + email are both captured, `_resolve_department()` (`widget.py`) does the final lookup: exact department match if one exists and is still active, otherwise the company's default department. Both the resolved `department_id` (FK) and `category_name` (plain-text snapshot) are stored on the `Lead` row — the snapshot keeps historical leads readable even if a department is later renamed or removed.
5. `send_lead_notification()` (`email_service.py`, stdlib `smtplib`) emails the lead's details to the resolved department's address. Runs synchronously right after the `Lead` row is committed, wrapped so an SMTP failure never breaks the visitor's chat response. Success/failure tracked via `Lead.email_sent`.

**Admin endpoints** (`internal.py`, `X-Internal-Secret` shared-secret auth):

| Route                                                 | Purpose                                                                                            |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `POST /internal/companies`                          | Create a company + its initial departments atomically (at least 1, exactly one default, required)  |
| `GET /internal/companies`                           | List all companies with their nested departments                                                   |
| `POST /internal/companies/{company_id}/departments` | Add departments to a company (refuses to add a second default onto a company that already has one) |
| `GET /internal/leads/{company_id}/download`         | Download captured leads as an Excel file                                                           |

**Known gaps:** no standalone department edit/deactivate endpoint yet; no automatic retry for a failed lead-notification email (`Lead.email_sent = False` is the signal, nothing acts on it yet); the Excel lead export doesn't yet include a `category` column even though the DB does.

---

## 5. Summary-Query Classification: Generic vs. Targeted

**The problem this replaced:** the retrieval query used to be overridden by a single keyword check — *any* message containing a summary-ish word (`summarize`, `summary`, `overview`, `recap`, `main points`) had its entire search query replaced with a fixed generic string before vector search ran. This meant a targeted request like *"summarize chapter abc"* retrieved the exact same chunks as a bare *"summarize"* — the actual subject named in the message was silently discarded before it ever reached retrieval. The keyword list also existed **twice**, independently, in two different files (once gating retrieval in `main.py`, once gating prompt selection in `llm_service.py`), which is how the mismatch went unnoticed.

**The fix — one classifier, two consumers.** `classify_summary_query(query)` (`llm_service.py`) is the single source of truth both call sites now use:

```python
SUMMARY_TRIGGER_RE = re.compile(
    r"\b(summari[sz]e|summary|summaries|recap|overview|tl;?dr|"
    r"main\s+points|key\s+points|highlights|gist|brief\s+me(\s+on)?)\b",
    re.IGNORECASE,
)

SUMMARY_OVERVIEW_SEARCH_QUERY = (
    "overview summary main background introduction key takeaways"
)

_SUMMARY_FILLER_WORDS = {
    "give", "me", "a", "an", "the", "of", "on", "for", "about", "regarding",
    "this", "that", "our", "my", "your", "please", "can", "you", "could",
    "i", "want", "need", "get", "provide", "short", "quick", "brief",
    "document", "doc", "chat", "conversation", "session", "everything",
    "all", "it", "thread", "file", "up", "with",
}

def classify_summary_query(query: str) -> dict:
    """Returns {"is_summary": bool, "mode": "generic"|"targeted"|None, "search_query": str}."""
```

**Logic:**

1. Match against `SUMMARY_TRIGGER_RE`. No match → not a summary query at all; `search_query` passes through unchanged.
2. On a match, strip the trigger phrase out of the query, tokenize what remains (`[a-zA-Z0-9']+`), and drop anything in `_SUMMARY_FILLER_WORDS`.
3. **Nothing meaningful left** (e.g. *"summarize"*, *"give me a summary"*, *"recap this chat"*) → **generic**: `search_query` is set to the fixed overview-boost string, same as the original (pre-fix) behavior.
4. **Something meaningful left** (e.g. *"summarize chapter abc"* → `"chapter"`, `"abc"` survive; *"overview of onboarding"* → `"onboarding"` survives) → **targeted**: `search_query` is set to the **original user query**, so retrieval searches for the actual named subject instead of the generic string.

**Where it's consumed:**

- `main.py` → `chat_with_memory`: uses `classify_summary_query(payload.query)["search_query"]` as the retrieval query passed to `search_similar_chunks`. This is the actual bug fix — non-summary and generic-summary retrieval is byte-for-byte identical to the old behavior; targeted-summary retrieval now finds chunks about the real subject.
- `llm_service.py` → `generate_rag_answer_with_memory`: uses `classify_summary_query(user_query)["is_summary"]` to pick between the normal and summary-flavored system prompt (formatting rules only — both generic and targeted summaries get the same prompt treatment, since prompt formatting doesn't depend on retrieval scope).
- `widget.py` is intentionally **not** wired to this classifier — it never had the override bug (it always searched on the raw query), so it's unaffected either way. It also currently has no generic-summary overview boost at all; that's a separate, pre-existing gap tracked independently, not part of this fix.

**Verified against, among others:** `"summarize"`, `"give me a summary"`, `"recap our conversation"`, `"tl;dr"` → generic; `"summarize chapter abc"`, `"tldr the pricing section"`, `"highlights of the Q3 report"`, `"summarize my account"` → targeted; ordinary non-summary questions → untouched.

---

## 6. Evaluation Methodology

**Corpus:** `the-metamorphosis-franz-kafka-10258.pdf` (single document).

**Dataset (`eval_dataset.json`):** 25 hand-written questions across five categories:

| Category        | Count | Purpose                                                        |
| --------------- | ----- | -------------------------------------------------------------- |
| Easy            | 8     | Single-fact lookup                                             |
| Multi-chunk     | 6     | Answer spans content spread across multiple chunks             |
| Cross-section   | 4     | Requires comparing Part I vs Part II/III content               |
| Ambiguous pairs | 4     | Near-duplicate questions testing retrieval precision           |
| Unanswerable    | 3     | Answer is not in the document — tests hallucination avoidance |

Each entry has a `question`, `expected_answer`, and `expected_sources` (resolved by **filename**, not document UUID — UUIDs are reassigned on every re-upload during experiments, so resolving by filename via `GET /documents` at run time keeps scoring valid across re-uploads).

**Retrieval scoring:** does the retrieved chunk's `document_id` match the expected source, checked at Top-1 and Top-3.

**Answer scoring:** cosine similarity between the generated answer and `expected_answer`, using the same `all-MiniLM-L6-v2` model already used for retrieval embeddings. This measures topical relevance, not verified factual correctness. Every eval run prints the full Q/expected/generated triple so results can be spot-checked by eye.

**Hallucination scoring:** for the 3 unanswerable questions, the generated answer is checked against a broadened set of refusal phrases (not a single exact canned string), since the LLM phrases correct refusals inconsistently.

---

## 7. Retrieval Results

**Top-1 / Top-3 document-level retrieval accuracy: 100% in every configuration ever tested** — all three chunk sizes, all four top-k values, across multiple independent runs.

> ⚠️ **Read this as a ceiling effect, not a tuned achievement.** With a single-document corpus, any retrieved chunk is trivially "the right document" — there's no distractor document to filter out. It would only become meaningful with a multi-document corpus.

---

## 8. Answer Evaluation & Hallucination Testing

Latest full run (25 questions):

| Metric                                       | Result                                   |
| -------------------------------------------- | ---------------------------------------- |
| Answerable questions scored                  | 22                                       |
| Average answer relevance (cosine similarity) | 50.82%                                   |
| Retrieval accuracy (doc-level)               | 100.0%                                   |
| Hallucination tests run                      | 3                                        |
| **Hallucination avoidance rate**       | **100.0% (3/3 correctly refused)** |
| Average latency                              | 7019.0 ms                                |

All 3 unanswerable questions (unnamed insect species, Gregor's exact age, exact wording of the apology letters) were correctly refused — this validates the broadened refusal-detection patterns.

---

## 9. Chunking Experiment

Fixed `top_k=5`, `chunk_overlap=50`. Tested across three independent runs:

| Chunk Size    | Run 1            | Run 2            | Run 3            | Avg Answer Sim   |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 300           | 56.19%           | 51.68%           | 54.05%¹         | ~54%             |
| **500** | **59.74%** | **62.23%** | **60.44%** | **~60.8%** |
| 800           | 51.98%           | 55.29%           | 55.43%           | ~54.2%           |

¹ Only 23/25 questions completed (2 requests failed) — noted rather than silently averaged in as equivalent.

**500 tokens wins consistently across every run.** 300 is too short to hold a complete scene for multi-chunk/cross-section questions; 800 often blends multiple unrelated topics into one embedding, diluting semantic focus.

**Decision: `chunk_size = 500`, `chunk_overlap = 50`.** Confidence: high (3 independent confirming runs).

> **Superseded by §11.** This sweep never tested `chunk_size` together with a `top_k` other than 5. A joint grid (§11) found `chunk_size=600` scores higher when paired with `top_k=7`.

---

## 10. Top-K Experiment

Fixed chunk size, swept `top_k`:

| top_k       | Answer Sim %               | Latency (ms)           |
| ----------- | -------------------------- | ---------------------- |
| 1           | 26.56% – 40.42%²         | 5612 – 6756           |
| 3           | 55.53% – 55.57%           | 6005 – 7898           |
| **5** | **59.93% – 62.57%** | **5749 – 7326** |
| 6³         | 62.57%                     | 6849                   |
| 7³         | 57.74%                     | 7540                   |
| 10          | 62.08% – 63.47%           | 6982 – 9349           |

² Wide spread on `top_k=1` is LLM-side sampling/routing variance, not a retrieval effect — retrieval accuracy stayed at 100%.
³ A narrower follow-up sweep around the original winner; `top_k=6` scored highest on a single run vs. `top_k=5`'s multiple confirming runs.

**top_k=1 is sharply worse** — multi-chunk questions are structurally unanswerable with one chunk. Quality rises through k=3 and k=5, then flattens: k=10 buys only ~0.4–2 points over k=5 for ~25–30% more latency and double the context tokens.

**Decision: `top_k = 5`.** Confidence: high.

> **Superseded by §11.** §9 and §10 varied `chunk_size` and `top_k` independently, never together, and both used a smaller/older version of `eval_dataset.json`. A full joint grid search (below) replaces `500/5` with a new evidence-based default.

---

## 11. Joint Grid Search (`chunk_size` × `chunk_overlap` × `top_k`)

§9 and §10 each varied one parameter at a time, so it was never confirmed that their individual winners (`chunk_size=500`, `top_k=5`) were actually the best combination when run together. This section closes that gap with a proper joint grid, run via `eval/grid_experiment.py` — a single script that re-indexes the document once per distinct `(chunk_size, chunk_overlap)` pair, then sweeps every paired `top_k` against that same index, so results are never contaminated by a stale/leftover index from a previous grid point. Each grid point is the full 25-question `eval_dataset.json` set.

| chunk_size    | chunk_overlap | top_k       | Top-1 %         | Top-3 %         | Answer Sim %    | Latency (ms) |
| ------------- | ------------- | ----------- | --------------- | --------------- | --------------- | ------------ |
| 450           | 50            | 7           | 100.0           | 100.0           | 70.33           | 11,255.5     |
| 500           | 50            | 4           | 100.0           | 100.0           | 60.67           | 9,982.3      |
| 500           | 50            | 6           | 100.0           | 100.0           | 66.60           | 11,366.4     |
| **600** | **50**  | **7** | **100.0** | **100.0** | **72.61** | 11,978.4     |
| 600           | 75            | 7           | 100.0           | 100.0           | 67.62           | 12,228.7     |
| 600           | 100           | 7           | 100.0           | 100.0           | 69.04           | 12,457.2     |

**Findings:**

- **`chunk_size=600, chunk_overlap=50, top_k=7` is the new best config** — highest answer similarity (72.61%) of any run, joint or single-variable, with 100% Top-1/Top-3 retrieval.
- **`top_k=4`'s earlier 13.64% doc-retrieval rate (§10-era result) was a bug, not a real effect.** Re-run here against a freshly-indexed corpus, `top_k=4` retrieves correctly 100% of the time — in line with every other `top_k` value. The likely cause was a stale `document_id` left over from a prior sweep's index; `get_filename_to_doc_id_map()` is now re-resolved at the start of every grid point specifically to prevent this.
- **More overlap does not help at `chunk_size=600` — it hurts.** Overlap 50 → 72.61%, overlap 75 → 67.62%, overlap 100 → 69.04%. Larger overlap shifts chunk boundaries in ways that diluted retrieval precision here rather than improving it; `chunk_overlap=50` is kept.
- **`chunk_size=450, top_k=7` (70.33%) is a reasonable fallback** if latency matters more than the last ~2 points of answer similarity — it's ~700ms faster than the winning config.

**Decision: `chunk_size = 600`, `chunk_overlap = 50`, `top_k = 7`.** This supersedes the `500/5` decision from §9–§10. Confidence: medium — this is a single joint-grid run (not yet independently repeated the way §9's chunk-size sweep was), and retrieval accuracy is still ceiling-effected by the single-document corpus (§7), so only answer similarity and latency are true discriminators here.

**Known gap carried over from this grid:** `avg_context_tokens_est` is 0 in every row above. `topk_experiment.py` / `grid_experiment.py` compute it from a `chunk_text` field on each returned source, but `main.py`'s `/documents/chat` endpoint never includes `chunk_text` when building `ChunkSource` objects — it's available in the underlying `chunks` list but not copied onto the response schema. Real context-token and cost figures for this grid are not yet available; see §16.

---

## 12. Token & Cost Tracking

`GET /analytics` aggregates a per-request JSONL log (`analytics_log.jsonl`) capturing method, path, status code, latency, and — for the two chat endpoints — input/context/output token counts sourced from the LLM provider's own usage accounting (not estimated).

- **Cost is pinned at $0.00** — deployed model is OpenRouter's free tier. Per-1K-token rate constants exist in `analytics.py`, ready to activate for a paid model.
- **Known limitation:** `/analytics` aggregates the entire log file for the server's lifetime, with no distinction between real usage and eval-script traffic. A `?since=<timestamp>` filter is recommended before quoting these numbers as real user traffic.
- **Known limitation:** failed requests (HTTP 500) are logged with a status code but no exception detail.

---

## 13. Performance Analysis

Per-stage timing, confirmed across ~300+ real requests:

| Pipeline                       | Stage                                                               | Typical share of total time                           |
| ------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------- |
| Query (`/documents/chat`)    | `llm_generation_ms`                                               | **85–98%**                                     |
| Query                          | `query_embedding_ms` + `vector_search_ms` + `context_prep_ms` | 2–15% combined                                       |
| Upload (`/documents/upload`) | `chunk_embedding_ms`                                              | **~96%** (batched via `get_embeddings_batch`) |
| Upload                         | `document_processing_ms`                                          | ~4%                                                   |

**Bottlenecks:** LLM generation time dominates the query pipeline — a network round-trip to an external/local API always dominates local embedding/search computation. Tail latency matters separately from the mean: several requests measured 13,000–20,000 ms against a typical 3,000–8,000 ms range, consistent with model-routing variability.

---

## 14. Improvements Implemented

| # | Problem identified                                                                                                                                                                               | Change implemented                                                                                                                                                                                                                                      | Result                                                                                                                                                         |
| - | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Discarded context-token count meant real usage was never visible in`/analytics`.                                                                                                               | Captured and returned`context_tokens` instead of discarding it.                                                                                                                                                                                       | `/analytics` now reports real context-token totals, not estimates.                                                                                           |
| 2 | Overly strict hallucination-refusal detection matched only one exact canned sentence.                                                                                                            | Broadened`REFUSAL_PATTERNS` to a set of equivalent refusal phrases.                                                                                                                                                                                   | Hallucination avoidance measured at 100% (3/3) instead of undercounting.                                                                                       |
| 3 | Chunking/retrieval configuration was unvalidated defaults.                                                                                                                                       | Ran repeated chunk-size and top-k sweeps across multiple independent runs, then a joint grid search across`chunk_size × chunk_overlap × top_k` to confirm the combination (§9–11); selected`chunk_size=600`, `chunk_overlap=50`, `top_k=7`. | Config finalized with documented confidence levels; joint grid confirmed the single-variable sweeps' winners did not hold once tested together (§11).         |
| 4 | New per-token CSV export would require synchronous per-chunk embedding calls inline in upload, risking latency regression.                                                                       | Implemented as an out-of-band task: batched embedding (`get_embeddings_batch`), dispatched via RQ/Redis with `BackgroundTasks` fallback.                                                                                                            | Upload response time unaffected by the export feature.                                                                                                         |
| 5 | Summary-flavored queries ("summarize chapter abc") had their real subject discarded and replaced with a generic search string, and the keyword list gating this was duplicated across two files. | Added`classify_summary_query()` (generic vs. targeted classification) as a single shared helper in `llm_service.py`; wired into `main.py`'s retrieval query and `llm_service.py`'s prompt selection.                                            | Targeted summary queries now retrieve chunks about the actual named subject; generic summaries and non-summary queries are unaffected; duplication eliminated. |

**Pending, not yet implemented:** `error_detail` on the analytics log for 500s; a `?since=` filter on `/analytics`; extending generic-summary overview-boost behavior to `widget.py` (currently has no override at all, targeted or generic); adding `chunk_text` to the `ChunkSource` response schema so eval scripts can compute real context-token counts instead of `0` (§11, §16).

---

## 15. Before vs. After Comparison

**Chunk size (at fixed top_k=5, single-variable sweep, §9):**

| Metric                        | Before (naive default: 300) | After (evidence-based: 500) |
| ----------------------------- | --------------------------- | --------------------------- |
| Answer relevance (cosine sim) | ~54%                        | ~60.8%                      |
| Retrieval accuracy            | 100%*                       | 100%*                       |
| Latency                       | ~8567 ms                    | ~7000–8500 ms              |

**Top-k (at fixed chunk size, single-variable sweep, §10):**

| Metric                        | Before (naive default: k=1) | After (evidence-based: k=5) |
| ----------------------------- | --------------------------- | --------------------------- |
| Answer relevance (cosine sim) | 26.56–40.42%               | 59.93–62.57%               |
| Retrieval accuracy            | 100%*                       | 100%*                       |
| Latency                       | 5612–6756 ms               | 5749–7326 ms               |

**Final joint-optimized config (§11) vs. the naive default:**

| Metric                        | Before (naive default:`chunk_size=300, top_k=1`) | After (joint grid winner:`chunk_size=600, chunk_overlap=50, top_k=7`) |
| ----------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------- |
| Answer relevance (cosine sim) | ~26.56–40.42% (k=1 range)                         | **72.61%**                                                        |
| Retrieval accuracy            | 100%*                                              | 100%*                                                                   |
| Latency                       | ~5612–6756 ms                                     | ~11,978 ms                                                              |

\* Retrieval accuracy is unchanged because it's ceiling-effected by the single-document corpus (§7), not because tuning had no retrieval impact.

**Caveat, resolved:** the two single-variable sweeps above (§9, §10) never confirmed their winners held when combined — an earlier single confirming run of `chunk_size=400, top_k=6` had landed within noise of the chosen `500/5` and was left under-replicated. §11's joint grid search closes this gap: it found `500/5` was **not** the joint optimum — `600/50/7` scores meaningfully higher on answer relevance, at the cost of noticeably higher latency (~12s vs. the ~7-8.5s and ~5.6-7.3s ranges above). Whether that latency cost is acceptable depends on the deployment's tolerance; `chunk_size=450, top_k=7` (70.33%, ~11.3s) is documented in §11 as a faster fallback.

---

## 16. Known Limitations & Next Steps

- **Retrieval accuracy is not a discriminating metric on this corpus** — a multi-document corpus would be needed to evaluate retrieval quality independent of answer quality.
- **LLM-side non-determinism** (`openrouter/free`'s multi-model auto-routing + temperature=0.3) means identical configurations produce different absolute scores run-to-run (~±3–7 points on cosine similarity). Rankings between configurations have held stable.
- ~~A joint chunk_size × top_k grid search has not been performed.~~ **Done (§11).** One caveat carried forward: it's currently a single run per grid point, not independently repeated the way §9's chunk-size sweep was (3 confirming runs) — worth a repeat pass if higher confidence is needed before a production change.
- **`avg_context_tokens_est` is always 0 in every eval script's output**, including §11's joint grid. Root cause found: `main.py`'s `/documents/chat` endpoint builds `ChunkSource` objects without a `chunk_text` field, even though `chunk_text` is available on the underlying `chunks` list — it's just never copied onto the response. Real context-token and cost figures require adding that field to the schema (see §14, pending).
- **~1–2% request failure rate (HTTP 500)** observed under sustained sweep load, with no captured error detail.
- **`widget.py` has no generic-summary overview boost** — bare "summarize" queries there still search on the literal word, unlike the fixed `main.py` path (§5).
