import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor.app.main import app

client = TestClient(app)


def test_upload_valid_txt():
    content = b"FastAPI is a modern web framework. It is fast and easy to build APIs with Python."
    file = ("test.txt", io.BytesIO(content), "text/plain")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.txt"
    assert data["file_type"] == ".txt"
    assert data["stats"]["total_words"] > 0


def test_upload_invalid_file_type():
    content = b"Image binary data..."
    file = ("test.png", io.BytesIO(content), "image/png")

    response = client.post("/documents/upload", files={"file": file})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_list_and_get_document():
    content = b"Document retrieval test content."
    file = ("retrieval.txt", io.BytesIO(content), "text/plain")
    upload_res = client.post("/documents/upload", files={"file": file})
    doc_id = upload_res.json()["id"]

    list_res = client.get("/documents")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id
    assert get_res.json()["extracted_text"] == "Document retrieval test content."


def test_get_invalid_document():
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/documents/{fake_uuid}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_search_document():
    content = b"Python is great for machine learning. Python is also great for backend development."
    file = ("search.txt", io.BytesIO(content), "text/plain")
    upload_res = client.post("/documents/upload", files={"file": file})
    doc_id = upload_res.json()["id"]

    search_res = client.get(f"/documents/{doc_id}/search?query=Python")
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["query"] == "Python"
    assert data["occurrences"] == 2
    assert len(data["matching_sentences"]) > 0


def test_delete_document():
    content = b"Temporary content to be deleted."
    file = ("delete.txt", io.BytesIO(content), "text/plain")
    upload_res = client.post("/documents/upload", files={"file": file})
    doc_id = upload_res.json()["id"]

    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200

    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 404