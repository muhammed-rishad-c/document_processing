from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

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

    
def store_chunk_vector(chunks_data:list[dict]):
    points = [
        PointStruct(
            id=str(chunk["point_id"]),
            vector=chunk["embedding"],
            payload={
                "document_id": str(chunk["document_id"]),
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "token_count": chunk["token_count"]
            }
        )
        for chunk in chunks_data
    ]
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
    
def search_similar_chunks(query_text: str, top_k: int = 3, document_id: str = None) -> list[dict]:
    query_vector = get_embedding(query_text)
    
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

    results = []
    for point in points:
        results.append({
            "chunk_id": str(point.id),
            "document_id": str(point.payload.get("document_id")),
            "chunk_index": int(point.payload.get("chunk_index", 0)),
            "chunk_text": str(point.payload.get("chunk_text", "")),
            "token_count": int(point.payload.get("token_count", 0)),
            "similarity_score": round(float(point.score), 4)
        })

    return results