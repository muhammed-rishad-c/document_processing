import uuid
import time
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
encoder = HuggingFaceEmbeddings(model_name=MODEL_PATH)

qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "document_chunks"


def init_qdrant():
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
init_qdrant()


def get_embedding(text: str) -> list[float]:
    return encoder.embed_query(text)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Encodes multiple texts in a single batched call via LangChain's
    HuggingFaceEmbeddings wrapper (still backed by SentenceTransformer
    internally, so batching behavior is unchanged). Does not change
    get_embedding() or any of its existing callers (e.g. search_similar_chunks)
    --- this is purely additive for the upload path. Preserves input order,
    so zip(chunks, embeddings) stays correctly aligned."""
    return encoder.embed_documents(texts)


# Lazy singleton: created on first use, not at import time, so it doesn't
# race against init_qdrant() (which runs in FastAPI's startup event and
# must create the collection before this wraps it).
_vectorstore = None


def _get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = QdrantVectorStore(
            client=qdrant,
            collection_name=COLLECTION_NAME,
            embedding=encoder,
        )
    return _vectorstore


def store_chunk_vector(chunks_data: list[dict]):
    points = []
    for chunk in chunks_data:
        try:
            point_id = str(uuid.UUID(str(chunk["point_id"])))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk["point_id"])))
        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "page_content": str(chunk["chunk_text"]),
                    "metadata": {
                        "document_id": str(chunk["document_id"]),
                        "chunk_index": int(chunk["chunk_index"]),
                        "token_count": int(chunk["token_count"]),
                        "chunk_id": point_id,
                    },
                },
            )
        )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def delete_vector(doc_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
    )


def search_similar_chunks(query_text: str,
                           top_k: int = 5,
                           document_id: str = None,
                           timing_out: dict | None = None) -> list[dict]:
    t_embed_start = time.perf_counter()
    query_vector = get_embedding(query_text)
    t_embed_end = time.perf_counter()

    query_filter = None
    if document_id and str(document_id).strip().lower() not in ["", "null", "undefined", "none"]:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=str(document_id).strip())
                )
            ]
        )

    t_search_start = time.perf_counter()
    try:
        scored_docs = _get_vectorstore().similarity_search_with_score_by_vector(
            embedding=query_vector,
            k=top_k,
            filter=query_filter,
        )
    except Exception as e:
        print(f"Qdrant query execution error: {str(e)}")
        raise e
    t_search_end = time.perf_counter()

    if timing_out is not None:
        timing_out["query_embedding_ms"] = round((t_embed_end - t_embed_start) * 1000, 2)
        timing_out["vector_search_ms"] = round((t_search_end - t_search_start) * 1000, 2)

    results = []
    for doc, score in scored_docs:
        metadata = doc.metadata or {}
        results.append({
            "chunk_id": str(metadata.get("chunk_id", "")),
            "document_id": str(metadata.get("document_id", "")),
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "chunk_text": doc.page_content,
            "token_count": int(metadata.get("token_count", 0)),
            "similarity_score": round(float(score), 4),
        })
    return results