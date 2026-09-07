"""
tests/test_main.py — Task 11: Automated Testing

TEST STRATEGY
-------------
Per Task 11's own instruction ("mock external APIs where appropriate"):

- Postgres and Qdrant are REAL — these are your own infrastructure, not an
  external API, and integration-testing against them is what actually
  proves retrieval/upload work end to end. Tests that need this are
  marked "INTEGRATION" below and will write real rows/vectors.
- The OpenRouter LLM call (doc_processor.app.llm_service.client) is
  MOCKED everywhere — this is the one genuinely external, token-costing
  API call, and Task 11 explicitly calls out mocking external APIs.
- search_similar_chunks is mocked in the chat-endpoint unit tests (to
  isolate response-shaping logic from retrieval quality, matching your
  original test file's existing pattern) but NOT mocked in the dedicated
  retrieval integration tests further down, which need it to hit real
  Qdrant data.

KNOWN GAPS SURFACED WHILE WRITING THESE TESTS (not fixed here, flagging
for awareness):
  1. DELETE /documents/{id} does not remove the {doc_id}_chunk_tokens.csv
     background-task file — only the original uploaded source file. Test
     fixtures clean this up manually; the route itself does not.
  2. There is no DELETE endpoint for chat sessions, so tests that create
     one via POST /chats cannot clean it up through the API. Real Postgres
     will accumulate chat_sessions/chat_messages rows across test runs.
  3. No similarity-threshold filtering exists in search_similar_chunks yet
     (it's a possible future Task 9 improvement, not implemented). The
     "similarity threshold" test below is a baseline of CURRENT behavior
     only, not a test of threshold filtering that doesn't exist yet.

Run from the project root:
    pytest tests/test_main.py -v
"""

import io
import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor.app.main import app, UPLOAD_DIR

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_llm_response(text: str, prompt_tokens: int = 140, completion_tokens: int = 12):
    """Builds a MagicMock shaped like the OpenAI/OpenRouter chat completion
    response object generate_rag_answer_with_memory expects, so the mocked
    LLM client returns something structurally real."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))],
        usage=MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _cleanup_document_artifacts(document_id: str) -> None:
    """Removes the token-frequency CSV the background task writes, since
    DELETE /documents/{id} does not (see module docstring, gap #1)."""
    token_csv = os.path.join(UPLOAD_DIR, f"{document_id}_chunk_tokens.csv")
    if os.path.exists(token_csv):
        os.remove(token_csv)


@pytest.fixture
def uploaded_document():
    """INTEGRATION fixture: uploads a real document through the real
    pipeline (real embedding model, real Postgres, real Qdrant). Yields
    the parsed upload response, then deletes the document afterward so
    repeated test runs don't pile up documents in the real database."""
    content = (
        b"Retrieval-Augmented Generation combines a vector database with an "
        b"LLM. FastAPI powers the backend API. Qdrant stores the embeddings."
    )
    file = ("pytest_integration_doc.txt", io.BytesIO(content), "text/plain")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()

    yield data

    client.delete(f"/documents/{data['document_id']}")
    _cleanup_document_artifacts(data["document_id"])


# ---------------------------------------------------------------------------
# Upload — fixed from the original suite
# ---------------------------------------------------------------------------

def test_upload_valid_txt():
    """INTEGRATION: real embedding model, real Postgres, real Qdrant.
    Previously mocked get_embedding, which no longer exists on this code
    path — the upload route now calls get_embeddings_batch()."""
    content = b"FastAPI is a modern web framework. It is fast and easy to build APIs with Python."
    file = ("test_upload_valid.txt", io.BytesIO(content), "text/plain")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_upload_valid.txt"
    assert data["total_chunks"] > 0

    # cleanup — not using the fixture here since we want the raw response
    client.delete(f"/documents/{data['document_id']}")
    _cleanup_document_artifacts(data["document_id"])


def test_upload_invalid_file_type():
    content = b"Image binary data..."
    file = ("test.png", io.BytesIO(content), "image/png")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file():
    """Invalid-request case: an empty file body should be rejected before
    any processing happens."""
    file = ("empty.txt", io.BytesIO(b""), "text/plain")
    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_generates_token_csv(uploaded_document):
    """Verifies the background token-sequence report actually gets written.
    TestClient runs BackgroundTasks synchronously as part of the request
    lifecycle, so by the time this response has returned, the file should
    already exist on disk."""
    doc_id = uploaded_document["document_id"]
    expected_path = os.path.join(UPLOAD_DIR, f"{doc_id}_chunk_tokens.csv")
    assert os.path.exists(expected_path), (
        "Expected the background chunk-token CSV to exist immediately after "
        "the upload response returns."
    )

    with open(expected_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    assert header == "chunk_index,position,token_id,token_text"


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

def test_list_and_get_document():
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)


def test_get_invalid_document():
    fake_uuid = str(uuid.uuid4())
    response = client.get(f"/documents/{fake_uuid}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_delete_nonexistent_document():
    """A document that was never uploaded should 404, not silently pass."""
    fake_uuid = str(uuid.uuid4())
    response = client.delete(f"/documents/{fake_uuid}")
    assert response.status_code == 404


def test_delete_real_document_removes_it():
    """INTEGRATION: uploads a real document, deletes it via the real route
    (hitting real Qdrant's delete_vector, not mocked), then confirms a
    follow-up GET correctly 404s."""
    content = b"This document exists only to be deleted by this test."
    file = ("pytest_delete_me.txt", io.BytesIO(content), "text/plain")
    upload_res = client.post("/documents/upload", files={"file": file})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_id"] == doc_id

    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 404

    _cleanup_document_artifacts(doc_id)


# ---------------------------------------------------------------------------
# Semantic search (Task 2 — retrieval)
# ---------------------------------------------------------------------------

def test_semantic_search_endpoint_contract():
    """Unit test of the endpoint's response shaping — mocks
    search_similar_chunks so this doesn't depend on any real data being
    present in Qdrant."""
    with patch("doc_processor.app.main.search_similar_chunks") as mock_search:
        mock_search.return_value = [
            {
                "chunk_id": str(uuid.uuid4()),
                "document_id": str(uuid.uuid4()),
                "chunk_index": 0,
                "chunk_text": "Vector databases index embeddings for rapid cosine search.",
                "token_count": 9,
                "similarity_score": 0.91,
            }
        ]

        payload = {"query": "vector databases", "top_k": 3}
        response = client.post("/documents/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "vector databases"
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity_score"] == 0.91


def test_semantic_search_empty_query():
    response = client.post("/documents/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 400
    assert response.json()["detail"] == "Search query cannot be empty."


def test_semantic_search_correct_document_retrieved(uploaded_document):
    """INTEGRATION: real Qdrant search — Task 11's 'correct document
    retrieval' case. Uploads a real document, searches for content that
    only exists in it, and confirms the returned document_id matches."""
    doc_id = uploaded_document["document_id"]
    payload = {"query": "Qdrant stores the embeddings", "top_k": 3, "document_id": doc_id}

    response = client.post("/documents/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert all(r["document_id"] == doc_id for r in data["results"])


def test_semantic_search_no_relevant_document():
    """INTEGRATION: 'no relevant document' case — filtering by a
    document_id that was never uploaded should return an empty result set
    from real Qdrant, not an error."""
    never_uploaded_id = str(uuid.uuid4())
    payload = {"query": "anything at all", "top_k": 3, "document_id": never_uploaded_id}

    response = client.post("/documents/search", json=payload)
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_similarity_score_baseline(uploaded_document):
    """BASELINE, not a real threshold test — see module docstring gap #3.
    No similarity-threshold filtering exists yet, so this only documents
    CURRENT behavior: results come back regardless of how low the score
    is, in descending similarity order. If threshold filtering is added
    later, this test's expectations will need to change, which is the
    point — it becomes the regression signal for that change.
    """
    doc_id = uploaded_document["document_id"]
    payload = {"query": "completely unrelated gibberish xyzzy", "top_k": 3, "document_id": doc_id}

    response = client.post("/documents/search", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Expected results ordered by descending similarity_score"
    )


# ---------------------------------------------------------------------------
# Token budget / context building (Task 6/7 — pure local logic, no mocking
# needed since count_token/build_safe_context never touch the network)
# ---------------------------------------------------------------------------

def test_token_budget_context_building():
    """Rewritten from scratch — the original test mocked a Gemini-style
    count_tokens() call that build_safe_context() never actually uses; it
    computes tokens itself via the project's own tiktoken-based
    count_token(). MAX_CONTEXT_TOKENS is patched down to a small value so
    the truncation boundary is deterministic and doesn't depend on the
    real production value (4000), which would need enormous test text to
    exercise."""
    from doc_processor.app import llm_service

    original_max = llm_service.MAX_CONTEXT_TOKENS
    llm_service.MAX_CONTEXT_TOKENS = 260  # room for ~2 short chunks past the ~200-token base overhead
    try:
        retrieved_chunks = [
            {"document_id": "doc-1", "chunk_text": "First short relevant chunk."},
            {"document_id": "doc-1", "chunk_text": "Second short relevant chunk here."},
            {
                "document_id": "doc-1",
                "chunk_text": (
                    "This third chunk is deliberately long and should be excluded "
                    "because by this point the token budget has already been mostly "
                    "consumed by the base overhead and the first two chunks, so this "
                    "much larger block of text pushes the running total past the "
                    "patched MAX_CONTEXT_TOKENS limit and must be dropped entirely."
                ),
            },
        ]

        context_str, total_tokens = llm_service.build_safe_context(retrieved_chunks, "test query")

        assert "First short relevant chunk" in context_str
        assert "Second short relevant chunk" in context_str
        assert "deliberately long and should be excluded" not in context_str
        assert total_tokens <= llm_service.MAX_CONTEXT_TOKENS
    finally:
        llm_service.MAX_CONTEXT_TOKENS = original_max


def test_chunking_and_token_count():
    from doc_processor.app.service import chunk_text

    sample_text = (
        "Retrieval-Augmented Generation enhances LLM capabilities by querying external vector databases. "
        "It prevents hallucination and provides up-to-date context groundings."
    )
    chunks = chunk_text(text=sample_text, max_chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 0
    assert "token_count" in chunks[0]
    assert "chunk_text" in chunks[0]


# ---------------------------------------------------------------------------
# Chat endpoints (Task 3/4 — LLM mocked, retrieval mocked for isolation)
# ---------------------------------------------------------------------------

@patch("doc_processor.app.main.search_similar_chunks")
@patch("doc_processor.app.llm_service.client")
def test_rag_chat_success(mock_llm_client, mock_search):
    mock_chunk_id = str(uuid.uuid4())
    mock_doc_id = str(uuid.uuid4())

    mock_search.return_value = [
        {
            "chunk_id": mock_chunk_id,
            "document_id": mock_doc_id,
            "chunk_index": 0,
            "chunk_text": "FastAPI enables rapid backend development.",
            "token_count": 6,
            "similarity_score": 0.88,
        }
    ]
    mock_llm_client.chat.completions.create.return_value = _fake_llm_response(
        "FastAPI is designed for fast API development."
    )

    payload = {"query": "What is FastAPI used for?", "top_k": 3}
    response = client.post("/documents/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "FastAPI is designed for fast API development."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == mock_doc_id
    assert data["sources"][0]["similarity_score"] == 0.88


@patch("doc_processor.app.main.search_similar_chunks")
def test_rag_chat_no_relevant_results(mock_search):
    mock_search.return_value = []

    payload = {"query": "Unknown topic", "top_k": 3}
    response = client.post("/documents/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "there is no content found in vector database"
    assert len(data["sources"]) == 0


def test_rag_chat_invalid_payload():
    res_invalid = client.post("/documents/chat", json={"top_k": "not_an_int"})
    assert res_invalid.status_code == 422


@patch("doc_processor.app.main.search_similar_chunks")
@patch("doc_processor.app.llm_service.client")
def test_chat_memory_success(mock_llm_client, mock_search):
    """INTEGRATION for the session (real Postgres via POST /chats), MOCKED
    for retrieval and the LLM call — see module docstring gap #2: there's
    no delete endpoint for chat sessions, so this session is not cleaned
    up afterward."""
    session_res = client.post("/chats", json={"title": "pytest session"})
    assert session_res.status_code == 201
    session_id = session_res.json()["id"]

    mock_search.return_value = [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "chunk_index": 0,
            "chunk_text": "Relevant chunk text for memory chat.",
            "token_count": 6,
            "similarity_score": 0.75,
        }
    ]
    mock_llm_client.chat.completions.create.return_value = _fake_llm_response(
        "Here is the answer using chat memory."
    )

    payload = {"session_id": session_id, "query": "Tell me more", "top_k": 3}
    response = client.post("/documents/chat-memory", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["answer"] == "Here is the answer using chat memory."
    assert len(data["sources"]) == 1

    # Confirm the turn was actually persisted
    messages_res = client.get(f"/chats/{session_id}/messages")
    assert messages_res.status_code == 200
    messages = messages_res.json()
    assert any(m["role"] == "user" and m["content"] == "Tell me more" for m in messages)
    assert any(m["role"] == "assistant" for m in messages)


def test_chat_memory_invalid_session():
    """Invalid-request case: a session_id that was never created should
    404, not 500 or silently proceed."""
    fake_session_id = str(uuid.uuid4())
    payload = {"session_id": fake_session_id, "query": "hello", "top_k": 3}
    response = client.post("/documents/chat-memory", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Chat session not found"


# ---------------------------------------------------------------------------
# Analytics (Task 7)
# ---------------------------------------------------------------------------

def test_analytics_endpoint_shape():
    """Reads the REAL analytics_log.jsonl, which accumulates entries from
    every test that ran before this one in the same session (via the
    app's own middleware) plus any prior real usage. Deliberately asserts
    structure and types only, never exact counts, since the log is not
    reset between runs."""
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()

    for key in (
        "total_requests", "requests_by_endpoint", "avg_response_time_ms",
        "token_usage", "stage_performance", "slowest_query_stage", "slowest_upload_stage",
    ):
        assert key in data

    assert isinstance(data["total_requests"], int)
    assert data["total_requests"] >= 1  # this very request was just logged
    assert isinstance(data["requests_by_endpoint"], dict)

    token_usage = data["token_usage"]
    for key in (
        "total_input_tokens", "total_context_tokens", "total_output_tokens",
        "total_tokens", "total_estimated_cost_usd", "requests_with_token_data",
    ):
        assert key in token_usage
