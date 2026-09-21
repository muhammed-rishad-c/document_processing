import uuid
import os
import time
from uuid import UUID
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Request, BackgroundTasks
from fastapi import APIRouter,Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse,Response
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from . import analytics
from . import feedback
from .database import engine, Base, get_db, Sessionlocal
from .models import (  
    Document,
    DocumentChunk,
    ChatSession,
    ChatMessage,
    Company
)
from .schemas import (
    DocumentResponse,
    DocumentDetailResponse,
    DocumentUploadResponse,
    SemanticSearchResponse,
    SemanticSearchRequest, 
    RAGRequest,
    RAGResponse,
    ChunkSource,
    ChatSessionCreate, 
    ChatSessionResponse, 
    ChatMessageResponse, 
    MemoryRAGRequest, 
    MemoryRAGResponse,
    FeedbackRequest
)
from .service import (
    extract_text_from_file, 
    calculate_document_stats,
    chunk_text_parent_child,
    generate_chunk_token_sequence_csv,
    extract_document_structure,
    needs_tier3_llm_fallback,
    _extract_tier3_llm_structure,
)  
from .vector_store import (
    init_qdrant,
    get_embeddings_batch,
    delete_vector,
    store_chunk_vector,
    search_similar_chunks
    
) 
  
from .llm_service import(
    generate_rag_answer_with_memory,
    classify_summary_query,
    classify_summary_target,
    generate_chat_summary,
    classify_structural_query,   
    answer_structural_query,
    classify_conversational_intent_dynamic
)


from .rate_limit import limiter

from .widget import router as widget_router
from .internal import router as internal_router
from .widget_cors import WidgetCorsMiddleware,ConditionalCORSMiddleware

Base.metadata.create_all(bind=engine)
  
app = FastAPI(
    title="Mini Document Processing System",
    description="API for uploading, analyzing, searching, and comparing documents.",
    version="2.0.0"
) 
  
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


load_dotenv()

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
] 

app.add_middleware(
    ConditionalCORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
app.add_middleware(WidgetCorsMiddleware)
 


app.include_router(internal_router)
app.include_router(widget_router)
WIDGET_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "widget")
app.mount("/widget-ui", StaticFiles(directory=WIDGET_STATIC_DIR, html=True), name="widget-ui")

GREETING_TEXT = (
    "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, "
    "solutions, or company — happy to help."
)      
 
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_qdrant()
    
     
@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        route = request.scope.get("route")
        path_template = route.path if route else request.url.path
        token_data = getattr(request.state, "token_usage", None) or {}
        stage_timings = getattr(request.state, "stage_timings", None)
        analytics.log_request(
            method=request.method,
            path=path_template,
            status_code=status_code,
            response_time_ms=elapsed_ms,
            input_tokens=token_data.get("input_tokens"),
            context_tokens=token_data.get("context_tokens"),
            output_tokens=token_data.get("output_tokens"),
            stage_timings=stage_timings,
        )
        
def _run_chunk_token_sequence_report(chunks: list[dict], output_path: str, doc_id) -> None:
    try:
        report = generate_chunk_token_sequence_csv(chunks, output_path)
        print(f"[background] Chunk token sequence report saved for {doc_id}: {report}")
    except Exception as e:
        print(f"[background] WARNING: failed to generate chunk token sequence CSV "
              f"for doc {doc_id}: {str(e)}")


def _run_tier3_structure_background(doc_id, extracted_text: str, page_count) -> None:
    """Runs Tier 3 (LLM chapter inference) after the upload response has
    already been sent, then persists the result onto Document.structure.

    Uses its own DB session (Sessionlocal) rather than the request-scoped
    `db` from get_db(), because that session is closed as soon as the
    request finishes — long before this background task runs.

    Never raises: a failed/slow LLM must not affect document ingestion,
    which has already succeeded by the time this runs.
    """
    try:
        structure = _extract_tier3_llm_structure(extracted_text, page_count)
    except Exception as e:
        print(f"[background] Tier 3 structure extraction crashed for doc {doc_id}: {e}")
        return

    db = Sessionlocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            print(f"[background] Tier 3 finished but doc {doc_id} no longer exists, discarding result")
            return
        doc.structure = structure
        db.add(doc)
        db.commit()
        print(f"[background] Tier 3 structure stored for doc {doc_id}: source={structure.get('source')}")
    except Exception as e:
        db.rollback()
        print(f"[background] Failed to persist Tier 3 structure for doc {doc_id}: {e}")
    finally:
        db.close()
         
        
@app.get("/analytics")
def get_analytics():
    return analytics.build_summary()

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("index.html")

@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all() 
 
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    t_proc_start = time.perf_counter()
    try:
        text, file_type = extract_text_from_file(file_bytes, file.filename)
        stats = calculate_document_stats(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Tier 1 (embedded TOC) only — fast, local, no LLM calls, safe to run
    # inline in the request. Tier 2 (font heuristics) was removed: it was
    # unreliable on real documents (miscounted "(Continued)" markers and
    # front/back matter as new chapters). Any document without an embedded
    # TOC now falls straight through to "none" and picks up Tier 3 below.
    structure = extract_document_structure(file_bytes, file.filename)
       
    doc = Document(
            filename=file.filename,
            file_type=file_type,
            extracted_text=text,
            stats=stats,
            structure=structure 
        ) 
    
    try:
        db.add(doc)
        db.commit()
        db.refresh(doc)
        print(f"[upload] document_id: {doc.id}")
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500,detail=f"Database error: {str(e)}")

    # Tier 3 (LLM inference) only runs when Tier 1 found nothing (no
    # embedded TOC), or the document is a non-PDF that skips Tier 1 entirely.
    # It's scheduled
    # as a background task — it needs 2-9 LLM calls, and the upload response
    # shouldn't wait on that. Document.structure gets updated in place once
    # it finishes; until then it stays "none".
    if needs_tier3_llm_fallback(structure):
        background_tasks.add_task(
            _run_tier3_structure_background,
            doc.id,
            text,
            structure.get("page_count"),
        )
    
       
        
    try:
        raw_chunks = chunk_text_parent_child(text=text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t_proc_end = time.perf_counter()
    
    token_csv_path = os.path.join(UPLOAD_DIR, f"{doc.id}_chunk_tokens.csv")
    background_tasks.add_task(_run_chunk_token_sequence_report, raw_chunks, token_csv_path, doc.id)
    
    db_chunks = []
    vector_data = []
    
    t_embed_start = time.perf_counter()
    
    child_chunks = [c for c in raw_chunks if not c["is_parent"]]
    chunk_texts = [c["chunk_text"] for c in child_chunks]
    chunk_embeddings = get_embeddings_batch(chunk_texts) if chunk_texts else []
    embeddings_by_index = {c["chunk_index"]: emb for c, emb in zip(child_chunks, chunk_embeddings)}

    for c in raw_chunks:
        chunk_uuid = uuid.uuid4()

        db_chunks.append(DocumentChunk(
            id=chunk_uuid,
            document_id=doc.id,
            chunk_index=c["chunk_index"],
            chunk_text=c["chunk_text"],
            token_count=c["token_count"],
            is_parent=c["is_parent"],
            parent_index=c["parent_index"],
        ))

        if not c["is_parent"]:
            vector_data.append({
                "point_id": chunk_uuid,
                "document_id": doc.id,
                "chunk_index": c["chunk_index"],
                "chunk_text": c["chunk_text"],
                "token_count": c["token_count"],
                "parent_index": c["parent_index"],
                "is_parent": False,
                "embedding": embeddings_by_index[c["chunk_index"]],
            })
    t_embed_end = time.perf_counter()

    request.state.stage_timings = {
        "document_processing_ms": round((t_proc_end - t_proc_start) * 1000, 2),
        "chunk_embedding_ms": round((t_embed_end - t_embed_start) * 1000, 2),
    } 
         
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

    affected_companies = db.query(Company).filter(Company.document_id == doc_id).all()
    
    # Extract IDs *before* deletion so we can safely return them later
    cascade_company_ids = [str(c.id) for c in affected_companies]

    if affected_companies:
        print(f"[delete_document] WARNING: deleting doc {doc_id} will cascade-delete "
              f"{len(affected_companies)} company(ies): "
              f"{[(c.id, c.name) for c in affected_companies]}")

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
        "cascade_deleted_companies": cascade_company_ids, # Use the pre-saved list here
    } 

@app.post("/documents/search", response_model=SemanticSearchResponse)
def semantic_search(request: SemanticSearchRequest, db: Session = Depends(get_db)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        doc_id_str = str(request.document_id) if request.document_id else None
        
        results = search_similar_chunks(
            query_text=request.query,
            top_k=request.top_k,
            document_id=doc_id_str,
            db_session=db
        )
        
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
    

@app.post("/documents/chat",response_model=RAGResponse)
def chat_with_document(payload:RAGRequest,request:Request,db: Session = Depends(get_db)):
    try:
        # No session here, so there's nothing to persist (no visitor_name
        # memory, no "last assistant message" for context) — but the
        # classifier still works fine without it, it just leans more on the
        # message itself. Real questions are unaffected; this only ever
        # returns early for greetings/thanks/farewells/etc.
        smalltalk = classify_conversational_intent_dynamic(payload.query)
        if smalltalk["is_smalltalk"]:
            return RAGResponse(query=payload.query, answer=smalltalk["reply"], sources=[])

        structural = classify_structural_query(payload.query)
        if structural["is_structural"]:
            doc = db.query(Document).filter(Document.id == payload.document_id).first() if payload.document_id else None
            answer_text = answer_structural_query(structural["kind"], doc.structure if doc else None)
            return RAGResponse(query=payload.query, answer=answer_text, sources=[])

        stage_timings: dict = {}
        chunks=search_similar_chunks(
            query_text=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            db_session=db,
            timing_out=stage_timings
        )
        
        
        if not chunks:
            request.state.stage_timings = stage_timings
            return RAGResponse(
                query=payload.query,
                answer="there is no content found in vector database",
                sources=[]
            )
        answer_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=chunks
        )
        stage_timings["context_prep_ms"] = answer_result.get("context_prep_ms", 0)
        stage_timings["llm_generation_ms"] = answer_result.get("llm_generation_ms", 0)
        request.state.stage_timings = stage_timings

        request.state.token_usage = {
            "input_tokens": answer_result.get("input_tokens", 0),
            "context_tokens": answer_result.get("context_tokens", 0),
            "output_tokens": answer_result.get("output_tokens", 0),
        }
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

    greeting_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=GREETING_TEXT
    )
    db.add(greeting_msg)
    db.commit()

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

@app.post("/chats/{session_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def submit_feedback(session_id: UUID, payload: FeedbackRequest, db: Session = Depends(get_db)):
    session_exists = db.query(ChatSession.id).filter(ChatSession.id == session_id).first()
    if not session_exists:
        raise HTTPException(status_code=404, detail="Chat session not found")

    feedback.log_feedback(
        session_id=str(session_id),
        rating=payload.rating,
        comment=payload.comment
    )

@app.post("/documents/chat-memory", response_model=MemoryRAGResponse)
def chat_with_memory(payload: MemoryRAGRequest, db: Session = Depends(get_db),request:Request=None):
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

    # Reuses the history we already loaded above for the "last assistant
    # message" context — no extra DB query needed here, unlike widget.py.
    existing_memory = session.session_memory or {}
    smalltalk = classify_conversational_intent_dynamic(
        payload.query,
        chat_history=history_payload,
        visitor_name=existing_memory.get("visitor_name"),
        is_first_turn=False,
    )
    if smalltalk["is_smalltalk"]:
        if smalltalk["memory_update"]:
            memory = dict(existing_memory)
            memory.update(smalltalk["memory_update"])
            session.session_memory = memory
            db.add(session)

        user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
        assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=smalltalk["reply"])
        db.add_all([user_msg, assistant_msg])
        db.commit()

        return MemoryRAGResponse(
            session_id=payload.session_id,
            query=payload.query,
            answer=smalltalk["reply"],
            sources=[]
        )

    if classify_summary_query(payload.query)["is_summary"] and classify_summary_target(payload.query) == "chat":
        unsummarized = history_payload[session.summarized_count:]
        answer_text = generate_chat_summary(unsummarized, session.running_summary or "")

        user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
        assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
        db.add_all([user_msg, assistant_msg])
        db.commit()

        return MemoryRAGResponse(
            session_id=payload.session_id,
            query=payload.query,
            answer=answer_text,
            sources=[]
        )
        
    target_doc_id = payload.document_id or (str(session.document_id) if session.document_id else None)
    structural = classify_structural_query(payload.query)
    if structural["is_structural"]: 
        if not target_doc_id:
            answer_text = "I'm not sure which document you mean — this chat isn't tied to one document. Could you specify which one you're asking about?"
        else:
            doc = db.query(Document).filter(Document.id == target_doc_id).first()
            answer_text = answer_structural_query(structural["kind"], doc.structure if doc else None)

        user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
        assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
        db.add_all([user_msg, assistant_msg])
        db.commit()

        return MemoryRAGResponse(
            session_id=payload.session_id,
            query=payload.query,
            answer=answer_text,
            sources=[]
        )

    try:
        stage_timings: dict = {}
        target_doc_id = payload.document_id or (str(session.document_id) if session.document_id else None)
        
        summary_info = classify_summary_query(payload.query)
        search_query = summary_info["search_query"]

        retrieved_chunks = search_similar_chunks(
            query_text=search_query,
            top_k=payload.top_k,
            document_id=target_doc_id,
            db_session=db,
            timing_out=stage_timings
        ) 

        llm_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=retrieved_chunks,
            chat_history=history_payload,
            session_summary=session.running_summary,
            session_summary_count=session.summarized_count,
        )
        stage_timings["context_prep_ms"] = llm_result.get("context_prep_ms", 0)
        stage_timings["llm_generation_ms"] = llm_result.get("llm_generation_ms", 0)
        request.state.stage_timings = stage_timings

        request.state.token_usage = {
            "input_tokens": llm_result.get("input_tokens", 0),
            "context_tokens": llm_result.get("context_tokens", 0),
            "output_tokens": llm_result.get("output_tokens", 0),
        }
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

    session.running_summary = llm_result.get("updated_summary", session.running_summary)
    session.summarized_count = llm_result.get("summarized_count", session.summarized_count)
    db.add_all([user_msg, assistant_msg, session])
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