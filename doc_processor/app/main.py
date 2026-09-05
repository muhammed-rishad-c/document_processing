import uuid
import os
from uuid import UUID
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi import APIRouter,Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import engine, Base, get_db
from .models import (
    Document,
    DocumentChunk,
    ChatSession,
    ChatMessage
)
from .schemas import (
    DocumentResponse,
    DocumentDetailResponse,
    DocumentUploadResponse,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
    SemanticSearchResponse,
    SemanticSearchRequest,
    RAGRequest,
    RAGResponse,
    ChunkSource,
    ChatSessionCreate, 
    ChatSessionResponse, 
    ChatMessageResponse, 
    MemoryRAGRequest, 
    MemoryRAGResponse
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

from .llm_service import(
    generate_rag_answer_with_memory,
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
    

@app.post("/documents/chat",response_model=RAGResponse)
def chat_with_document(payload:RAGRequest):
    try:
        chunks=search_similar_chunks(
            query_text=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id
            
        )
        
        if not chunks:
            return RAGResponse(
                query=payload.query,
                answer="there is no content found in vector database",
                sources=[]
            )
        answer_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=chunks
        )
        sources = [
        ChunkSource(
            chunk_id=str(c.get("chunk_id", "")),
            document_id=c.get("document_id", 0),
            chunk_index=c.get("chunk_index", 0),
            similarity_score=float(c.get("similarity_score", 0.0)),
        )
        for c in chunks
        ]

        return RAGResponse(
            query=payload.query,
            answer=answer_result["text"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/chats", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_chat_session(payload: ChatSessionCreate, db: Session = Depends(get_db)):
    doc_uuid = UUID(payload.document_id) if payload.document_id else None
    
    session = ChatSession(
        title=payload.title,
        document_id=doc_uuid
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/chats/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_chat_messages(session_id: UUID, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return messages

@app.post("/documents/chat-memory", response_model=MemoryRAGResponse)
def chat_with_memory(payload: MemoryRAGRequest, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    all_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == payload.session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    history_payload = [{"role": msg.role, "content": msg.content} for msg in all_messages]

    try:
        target_doc_id = payload.document_id or (str(session.document_id) if session.document_id else None)
        
        search_query = payload.query
        summary_terms = ["summarize", "summary", "overview", "recap", "main points"]
        if any(term in payload.query.lower() for term in summary_terms):
            search_query = "overview summary main background introduction key takeaways"

        retrieved_chunks = search_similar_chunks(
            query_text=search_query,
            top_k=payload.top_k,
            document_id=target_doc_id
        )

        llm_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=retrieved_chunks,
            chat_history=history_payload
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

    user_msg = ChatMessage(
        session_id=payload.session_id,
        role="user",
        content=payload.query
    )
    assistant_msg = ChatMessage(
        session_id=payload.session_id,
        role="assistant",
        content=llm_result["text"]
    )
    
    db.add_all([user_msg, assistant_msg])
    db.commit()

    formatted_sources = [
        ChunkSource(
            chunk_id=str(c.get("chunk_id", "")),
            document_id=str(c.get("document_id", "")),
            chunk_index=c.get("chunk_index", 0),
            similarity_score=float(c.get("similarity_score", 0.0))
        )
        for c in retrieved_chunks
    ]

    return MemoryRAGResponse(
        session_id=payload.session_id,
        query=payload.query,
        answer=llm_result["text"],
        sources=formatted_sources
    )
    
    
    
