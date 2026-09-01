# 📄 Intelligent Document Processor API

A high-performance, asynchronous RESTful API built with **FastAPI**, **SQLAlchemy**, and **PostgreSQL** designed to ingest, process, extract text, and compute real-time analytical metadata from uploaded documents.

---

## ✨ Features

* **File Upload & Validation**: Secure ingestion for document formats (e.g., `.txt`, `.pdf`).
* **Automated Text Analytics**: Instant computation of text metrics including word count, character count, sentence count, paragraph count, and top keyword frequency distributions.
* **Structured PostgreSQL Storage**: Strongly typed schema modeling utilizing **UUIDs** and native **JSONB** columns for optimized document metadata indexing.
* **RESTful Endpoints**: Full CRUD operations for document uploading, listing, fetching, text search, and deletion.
* **Automated Testing Suite**: End-to-end unit and integration testing built with **Pytest** and **HTTPX**.

---

## 🏗️ Project Architecture & Tech Stack

* **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
* **Database**: [PostgreSQL](https://www.postgresql.org/)
* **ORM**: [SQLAlchemy 2.0](https://www.sqlalchemy.org/)
* **Data Validation**: [Pydantic V2](https://docs.pydantic.dev/)
* **Testing**: [Pytest](https://docs.pytest.org/) & `TestClient` / `httpx`

```text
liquidlab/
├── doc_processor/
│   ├── app/
│   │   ├── database.py   # Database connection & session setup
│   │   ├── main.py       # FastAPI routing & application entrypoint
│   │   ├── models.py     # SQLAlchemy ORM models (PostgreSQL)
│   │   ├── schemas.py    # Pydantic schemas for request/response validation
│   │   └── utils.py      # Text processing & analytics utilities
│   └── __init__.py
├── tests/
│   └── test_main.py      # Integration and unit tests
├── requirements.txt      # Project dependencies
└── README.md