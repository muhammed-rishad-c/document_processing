from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "documents"

def init_qdrant():
    
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
def get_embedding(text: str) -> list[float]:
    return encoder.encode(text).tolist()

def check_similarity(vector: list[float], limit: int = 3):
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit
    )
    return [
        {
            "document_id": point.payload.get("document_id"),
            "filename": point.payload.get("filename"),
            "score": round(point.score, 4)
        }
        for point in results.points
    ]
    
def store_vector(doc_id: str, vector: list[float], filename: str):
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=doc_id,
                vector=vector,
                payload={"document_id": doc_id, "filename": filename}
            )
        ]
    )