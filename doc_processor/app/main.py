import os
from uuid import UUID
import uuid
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import engine, Base, get_db
from .models import Document,DocumentChunk
from .schemas import (
    DocumentResponse,
    DocumentDetailResponse,
    DocumentUploadResponse,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
    SemanticSearchResponse,
    SemanticSearchRequest
)
from .service import (
    extract_text_from_file,
    calculate_document_stats,
    search_text_in_document,
    count_token,
    chunk_text

)
from .vector_store import (
    init_qdrant,
    get_embedding,
    delete_vector,
    store_chunk_vector,
    search_similar_chunks
    
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Mini Document Processing System",
    description="API for uploading, analyzing, searching, and comparing documents.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("index.html")

@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...),
        db: Session = Depends(get_db),
        chunk_size:int=300,
        chunk_overlap:int=50
    ):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    try:
        text, file_type = extract_text_from_file(file_bytes, file.filename)
        stats = calculate_document_stats(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    doc = Document(
            filename=file.filename,
            file_type=file_type,
            extracted_text=text,
            stats=stats
        ) 
    
    try:
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500,detail=f"Database error: {str(e)}")
    
    
    try:
        raw_chunks=chunk_text(text=text,max_chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e))
    
    db_chunks=[]
    vector_data=[]
    
    for c in raw_chunks:
        chunk_uuid=uuid.uuid4()
        chunk_embedding=get_embedding(c["chunk_text"])
        
        db_chunk=DocumentChunk(
            id=chunk_uuid,
            document_id=doc.id,
            chunk_index=c['chunk_index'],
            chunk_text=c["chunk_text"],
            token_count=c["token_count"]
        )
        
        db_chunks.append(db_chunk)
        
        vector_data.append({
            "point_id": chunk_uuid,
            "document_id": doc.id,
            "chunk_index": c["chunk_index"],
            "chunk_text": c["chunk_text"],
            "token_count": c["token_count"],
            "embedding": chunk_embedding
        })
        
    try:
        db.add_all(db_chunks)
        db.commit()
        store_chunk_vector(vector_data)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Chunk processing error: {str(e)}")
    
    
    saved_path = os.path.join(UPLOAD_DIR, f"{doc.id}_{file.filename}")
    with open(saved_path, "wb") as f:
        f.write(file_bytes)
        
    
        
    return {
        "document_id":doc.id,
        "filename":doc.filename,
        "total_chunks": len(db_chunks),
        "stats": doc.stats,
        "message": "Document successfully processed, chunked, embedded, and stored in PostgreSQL & Qdrant."
    }

@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(doc_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@app.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(doc_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc_id_str = str(doc.id)

    try:
        delete_vector(doc_id_str)
    except Exception as e:
        print(f"Warning: Failed to delete Qdrant vectors for doc {doc_id_str}: {str(e)}")
        

    saved_path = os.path.join(UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if os.path.exists(saved_path):
        os.remove(saved_path)

    db.delete(doc)
    db.commit()

    return {
        "message": "Document successfully deleted from PostgreSQL, Qdrant, and local storage.",
        "deleted_id": doc_id_str
    }

@app.post("/documents/search", response_model=SemanticSearchResponse)
def semantic_search(request: SemanticSearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        doc_id_str = str(request.document_id) if request.document_id else None
        
        results = search_similar_chunks(
            query_text=request.query,
            top_k=request.top_k,
            document_id=doc_id_str
        )
        
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
    
