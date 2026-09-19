import uuid
import time
import os
import json as _json_dbg
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from .models import DocumentChunk        
from .service import count_token  

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
encoder = HuggingFaceEmbeddings(model_name=MODEL_PATH)

qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "document_chunks"

DEBUG_CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "debug_last_chunks.json")


 
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
    
    return encoder.embed_documents(texts)

def dump_chunks_for_debug(query_text: str, results: list[dict]) -> None:
    try:
        with open(DEBUG_CHUNKS_PATH, "w", encoding="utf-8") as f:
            _json_dbg.dump({"query": query_text, "results": results}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[dump_chunks_for_debug] failed: {e}")



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
        if chunk.get("is_parent"):
            continue  
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
                        "parent_index": int(chunk["parent_index"]),
                        "token_count": int(chunk["token_count"]),
                        "chunk_id": point_id,
                    },
                },
            )
        )
    if points:
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


PARENT_CONTEXT_BUDGET = 2500  

def search_similar_chunks(query_text: str,
                           top_k: int = 10,
                           document_id: str = None,
                           db_session=None,
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

    seen_parents = set()
    results = []
    total_tokens = 0
    for doc, score in scored_docs:
        metadata = doc.metadata or {}
        parent_idx = metadata.get("parent_index")
        if parent_idx is None or parent_idx in seen_parents:
            continue
        if db_session is None:
            print("[search_similar_chunks] no db_session provided, cannot resolve parent — skipping hit")
            continue
        parent_text = fetch_parents_by_index(
            db_session, str(metadata.get("document_id", "")), [parent_idx]
        ).get(parent_idx)
        if not parent_text:
            continue
        parent_tokens = count_token(parent_text)
        if total_tokens + parent_tokens > PARENT_CONTEXT_BUDGET:
            break
        seen_parents.add(parent_idx)
        total_tokens += parent_tokens
        results.append({
            "chunk_id": str(metadata.get("chunk_id", "")),
            "document_id": str(metadata.get("document_id", "")),
            "chunk_index": int(parent_idx),
            "chunk_text": parent_text,
            "token_count": parent_tokens,
            "similarity_score": round(float(score), 4),
        })

    dump_chunks_for_debug(query_text, results)
    return results


def fetch_parents_by_index(db_session, document_id: str, parent_indices: list[int]) -> dict[int, str]:
    
    if not parent_indices:
        return {}
    rows = (
        db_session.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_parent == True,
            DocumentChunk.chunk_index.in_(parent_indices),
        )
        .all()
    )
    return {row.chunk_index: row.chunk_text for row in rows}