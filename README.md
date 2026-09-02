# 📄 Intelligent Document Processor & Vector API

A high-performance, local-first RESTful API built with **FastAPI**, **SQLAlchemy**, **PostgreSQL**, and **Qdrant** designed to ingest, sanitize, extract text, compute real-time analytical metadata, and perform vector similarity matching on uploaded documents.

---

## ✨ Features

* **Multi-Format Ingestion & Cleaning**: Support for `.txt` and `.pdf` files with automatic NUL byte (`\x00`) sanitization to prevent database encoding errors.
* **Automated Text Analytics**: Instant computation of text metrics including total word counts, character counts, average word length, and language statistics.
* **Vector Similarity Matching (Qdrant)**: Automated dense vector embedding generation using `sentence-transformers` (`all-MiniLM-L6-v2`) with instant semantic similarity searching across stored documents upon upload.
* **Structured PostgreSQL Storage**: Strongly typed schema modeling utilizing **UUIDs** and native **JSONB** columns for optimized document metadata indexing.
* **RESTful Endpoints & Cascading Deletion**: Full CRUD operations for document uploading, listing, fetching, text search, and deletion across PostgreSQL, Qdrant, and local file storage.
* **Interactive Web Workspace**: Built-in tabbed dark-mode dashboard (`index.html`) for uploading files, inspecting vector scores, searching by UUID, and managing records.

---

## 🏗️ Project Architecture & Tech Stack

* **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
* **Relational Database**: [PostgreSQL](https://www.postgresql.org/) (via SQLAlchemy 2.0 ORM)
* **Vector Store**: [Qdrant](https://qdrant.tech/) (`qdrant-client`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Data Validation**: [Pydantic V2](https://docs.pydantic.dev/)
* **Document Extraction**: `pypdf` & native Python text decoding

```text
doc_processor/
├── app/
│   ├── database.py       # PostgreSQL database connection & session setup
│   ├── main.py           # FastAPI routing, middleware, & application entrypoint
│   ├── models.py         # SQLAlchemy ORM models (PostgreSQL)
│   ├── schemas.py        # Pydantic schemas for request/response validation
│   ├── service.py        # Text extraction, NUL-byte sanitization, & analytics utilities
│   ├── vector_store.py   # Qdrant client connection & embedding generation
│   └── uploads/          # Physical disk storage for raw file uploads
├── index.html            # Web workspace frontend dashboard
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation