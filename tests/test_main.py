import io
import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor.app.main import app

client = TestClient(app)




@patch("doc_processor.app.main.store_chunk_vector")
@patch("doc_processor.app.main.get_embedding")
def test_upload_valid_txt(mock_get_embedding, mock_store_vector):
    mock_get_embedding.return_value = [0.1] * 384
    mock_store_vector.return_value = True

    content = b"FastAPI is a modern web framework. It is fast and easy to build APIs with Python."
    file = ("test.txt", io.BytesIO(content), "text/plain")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test.txt"
    assert "total_chunks" in data


def test_upload_invalid_file_type():
    content = b"Image binary data..."
    file = ("test.png", io.BytesIO(content), "image/png")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_list_and_get_document():
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)


def test_get_invalid_document():
    fake_uuid = str(uuid.uuid4())
    response = client.get(f"/documents/{fake_uuid}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


@patch("doc_processor.app.main.delete_vector")
def test_delete_document(mock_delete_vector):
    mock_delete_vector.return_value = True
    fake_uuid = str(uuid.uuid4())

    del_res = client.delete(f"/documents/{fake_uuid}")
    assert del_res.status_code in [200, 404]



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


@patch("doc_processor.app.llm_service.client")
def test_token_budget_context_building(mock_genai_client):
    from doc_processor.app.llm_service import build_safe_context

    mock_genai_client.models.count_tokens.return_value = MagicMock(total_tokens=1500)

    retrieved_chunks = [
        {"document_id": "doc-1", "chunk_text": "First highly relevant context chunk."},
        {"document_id": "doc-1", "chunk_text": "Second relevant context chunk."},
        {"document_id": "doc-1", "chunk_text": "Third chunk that exceeds limit."},
    ]

    context_str, total_tokens = build_safe_context(retrieved_chunks, "User Query")

    assert total_tokens == 3000
    assert "First highly relevant" in context_str
    assert "Second relevant" in context_str
    assert "Third chunk" not in context_str




@patch("doc_processor.app.main.search_similar_chunks")
def test_semantic_search_endpoint(mock_search):
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


@patch("doc_processor.app.main.search_similar_chunks")
@patch("doc_processor.app.main.generate_rag_answer")
def test_rag_chat_success(mock_generate, mock_search):
    mock_chunk_id = str(uuid.uuid4())
    mock_doc_id = str(uuid.uuid4())

    mock_search.return_value = [
        {
            "chunk_id": mock_chunk_id,
            "document_id": mock_doc_id,
            "chunk_index": 0,
            "chunk_text": "FastAPI enables rapid backend development.",
            "similarity_score": 0.88,
        }
    ]
    mock_generate.return_value = {
        "text": "FastAPI is designed for fast API development.",
        "input_tokens": 140,
        "output_tokens": 12,
    }

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



def test_semantic_search_empty_query():
    
    response = client.post("/documents/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 400
    assert response.json()["detail"] == "Search query cannot be empty."


def test_rag_chat_invalid_payload():
    
    res_invalid = client.post("/documents/chat", json={"top_k": "not_an_int"})
    assert res_invalid.status_code == 422