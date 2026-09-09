from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from fastapi import Request

from .database import get_db
from .models import ChatSession, ChatMessage, Company
from .schemas import (
    WidgetSessionCreate,
    ChatSessionResponse,
    WidgetChatRequest,
    WidgetChatResponse,
)
from .llm_service import generate_rag_answer_with_memory
from .vector_store import search_similar_chunks
from .rate_limit import limiter

GREETING_TEXT = (
    "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, "
    "solutions, or company -- happy to help."
)

router = APIRouter(prefix="/widget", tags=["public-widget"])


def get_company_from_api_key(
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
) -> Company:
    company = db.query(Company).filter(Company.api_key == x_api_key).first()
    if not company or not company.is_active:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return company


@router.post("/session", response_model=ChatSessionResponse, status_code=201)
@limiter.limit("20/minute")
def create_widget_session(
    request: Request,
    payload: WidgetSessionCreate,
    db: Session = Depends(get_db),
    company: Company = Depends(get_company_from_api_key),
):
    session = ChatSession(
        title=payload.title,
        document_id=company.document_id,
        company_id=company.id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    greeting_msg = ChatMessage(session_id=session.id, role="assistant", content=GREETING_TEXT)
    db.add(greeting_msg)
    db.commit()
    return session


@router.post("/chat", response_model=WidgetChatResponse)
@limiter.limit("10/minute")
def widget_chat(request: Request, payload: WidgetChatRequest, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    company = db.query(Company).filter(Company.id == session.company_id).first()
    if not company or not company.is_active:
        raise HTTPException(status_code=401, detail="This session is no longer active.")

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
        document_id=str(company.document_id),
    )

    try:
        llm_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=retrieved_chunks,
            chat_history=history_payload,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        )

    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
    assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=llm_result["text"])
    db.add_all([user_msg, assistant_msg])
    db.commit()

    return WidgetChatResponse(session_id=payload.session_id, answer=llm_result["text"])