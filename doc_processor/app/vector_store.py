import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct ,Filter, FieldCondition, MatchValue

encoder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "document_chunks"

def init_qdrant():
    
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
def get_embedding(text: str) -> list[float]:
    return encoder.encode(text).tolist()

    
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
                    "document_id": str(chunk["document_id"]),
                    "chunk_index": int(chunk["chunk_index"]),
                    "chunk_text": str(chunk["chunk_text"]),
                    "token_count": int(chunk["token_count"])
                }
            )
        )
        
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
def delete_vector(doc_id: str):

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
    )
    
def search_similar_chunks(query_text: str,
                        top_k: int = 5,
                        document_id: str = None,
                        timing_out:dict | None=None) -> list[dict]:
    t_embed_start=time.perf_counter()
    query_vector = get_embedding(query_text)
    t_embed_end=time.perf_counter()
    query_filter = None

    if document_id and str(document_id).strip().lower() not in ["", "null", "undefined", "none"]:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id).strip())
                )
            ]
        )
        
    t_search_start=time.perf_counter()

    try:
        if hasattr(qdrant, "query_points"):
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            points = response.points
        else:
            points = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
    except Exception as e:
        print(f"Qdrant query execution error: {str(e)}")
        raise e
    t_search_end=time.perf_counter()
    
    if timing_out is not None:
        timing_out["query_embedding_ms"] = round((t_embed_end - t_embed_start) * 1000, 2)
        timing_out["vector_search_ms"] = round((t_search_end - t_search_start) * 1000, 2)

    results = []
    for point in points:
        payload = point.payload or {}
        results.append({
            "chunk_id": str(point.id),
            "document_id": str(payload.get("document_id", "")),
            "chunk_index": int(payload.get("chunk_index", 0)),
            "chunk_text": str(payload.get("chunk_text", "")),
            "token_count": int(payload.get("token_count", 0)),
            "similarity_score": round(float(point.score), 4)
        })

    return results
