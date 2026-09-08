import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import get_db
from .models import ChatSession, ChatMessage
from .schemas import (
    WidgetSessionCreate,
    ChatSessionResponse,
    WidgetChatRequest,
    WidgetChatResponse,
)
from .llm_service import generate_rag_answer_with_memory
from .vector_store import search_similar_chunks

LIQUIDLAB_DOCUMENT_ID = os.getenv("LIQUIDLAB_DOCUMENT_ID")
if not LIQUIDLAB_DOCUMENT_ID:
    raise RuntimeError(
        "LIQUIDLAB_DOCUMENT_ID environment variable is not set. "
        "Set it in .env before starting the server."
    )

GREETING_TEXT = (
    "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, "
    "solutions, or company -- happy to help."
)

router = APIRouter(prefix="/widget", tags=["public-widget"])


@router.post("/session", response_model=ChatSessionResponse, status_code=201)
def create_widget_session(payload: WidgetSessionCreate, db: Session = Depends(get_db)):
    session = ChatSession(
        title=payload.title,
        document_id=LIQUIDLAB_DOCUMENT_ID,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    greeting_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=GREETING_TEXT,
    )
    db.add(greeting_msg)
    db.commit()
    return session


@router.post("/chat", response_model=WidgetChatResponse)
def widget_chat(payload: WidgetChatRequest, db: Session = Depends(get_db)):
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

    retrieved_chunks = search_similar_chunks(
        query_text=payload.query,
        top_k=5,
        document_id=LIQUIDLAB_DOCUMENT_ID,
    )

    llm_result = generate_rag_answer_with_memory(
        user_query=payload.query,
        retrieved_chunks=retrieved_chunks,
        chat_history=history_payload,
    )

    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
    assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=llm_result["text"])
    db.add_all([user_msg, assistant_msg])
    db.commit()

    return WidgetChatResponse(session_id=payload.session_id, answer=llm_result["text"])