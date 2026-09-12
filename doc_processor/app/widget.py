import json

from fastapi import APIRouter, Depends, HTTPException, Header, Response
from sqlalchemy.orm import Session
from fastapi import Request
from slowapi.util import get_remote_address

from .database import get_db
from .models import ChatSession, ChatMessage, Company, Lead, CompanyDepartment
from .schemas import (
    WidgetSessionCreate,
    ChatSessionResponse,
    WidgetChatRequest,
    WidgetChatResponse,
)
from .llm_service import generate_rag_answer_with_memory, extract_lead_info, classify_query, NO_ANSWER_TEXT
from .lead_export import append_lead
from .email_service import send_lead_notification
from .vector_store import search_similar_chunks
from .rate_limit import limiter, key_func_by_api_key, key_func_by_session_id


GREETING_TEXT = (
    "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, "
    "solutions, or company -- happy to help."
)

MAX_LEAD_CAPTURE_ATTEMPTS = 3

LEAD_CAPTURE_PROMPT = (
    "I couldn't find that in our documentation, but our support team can help directly. "
    "Could you share your name and email (and phone, if you'd like) so they can reach out?"
)
LEAD_CAPTURE_REPROMPT_MISSING_NAME = "Thanks! Could you also share your name?"
LEAD_CAPTURE_REPROMPT_MISSING_EMAIL = "Thanks! Could you also share your email address?"
LEAD_CAPTURE_REPROMPT_MISSING_BOTH = "Could you share your name and email so support can reach out?"
LEAD_CAPTURE_THANK_YOU = (
    "Thanks — I've passed this along to our support team, they'll be in touch shortly!"
)
LEAD_CAPTURE_GIVE_UP = (
    "No problem — feel free to ask me anything else in the meantime!"
)


def _session_ip_backstop(request: Request):
    pass

_session_ip_backstop = limiter.limit("60/minute", key_func=get_remote_address)(_session_ip_backstop)

router = APIRouter(prefix="/widget", tags=["public-widget"])


def _check_origin_and_allow(request: Request, response: Response, company: Company):
    
    embed_origin = request.headers.get("x-embed-origin")
    allowed = company.allowed_origins or []
    if not embed_origin or embed_origin not in allowed:
        raise HTTPException(status_code=403, detail="Origin not allowed for this company.")

    browser_origin = request.headers.get("origin", "*")
    response.headers["Access-Control-Allow-Origin"] = browser_origin
    response.headers["Vary"] = "Origin"


def get_company_from_api_key(
    request: Request,
    response: Response,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
) -> Company:
    company = db.query(Company).filter(Company.tenant_id == x_api_key).first()
    if not company or not company.is_active:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    _check_origin_and_allow(request, response, company)
    return company


def _load_pending_lead(session: ChatSession) -> dict:
    """Parses the JSON blob stored in pending_lead_query into
    {"question", "category_name", "name", "email", "phone"}. Never raises —
    falls back to an empty shell if the field is missing or somehow
    malformed, so a bad/old value can't crash the request. category_name
    defaults to None for old-shape blobs saved before this field existed;
    it gets resolved to the company's default department at Lead-creation
    time, not here."""
    empty = {"question": "", "category_name": None, "name": None, "email": None, "phone": None}
    if not session.pending_lead_query:
        return empty
    try:
        data = json.loads(session.pending_lead_query)
        if not isinstance(data, dict):
            return empty
        return {
            "question": data.get("question", ""),
            "category_name": data.get("category_name"),
            "name": data.get("name"),
            "email": data.get("email"),
            "phone": data.get("phone"),
        }
    except Exception:
        return empty


def _save_pending_lead(session: ChatSession, question: str, category_name, name, email, phone) -> None:
    session.pending_lead_query = json.dumps(
        {"question": question, "category_name": category_name, "name": name, "email": email, "phone": phone}
    )
    
def _get_active_departments(db: Session, company_id) -> list[CompanyDepartment]:
    return (
        db.query(CompanyDepartment)
        .filter(CompanyDepartment.company_id == company_id, CompanyDepartment.is_active.is_(True))
        .all()
    )


def _resolve_department(db: Session, company: Company, category_name: str | None):
    """Resolves a classified category name to a real, active department for
    this company. Falls back to the company's default department if
    category_name is None or doesn't match any active department —
    guarantees a lead is never left without a deliverable destination.
    Note: v1 has no standalone department update/deactivate endpoints
    (create-with-company only), so is_active can't change post-creation yet —
    a default with is_active=False can't currently occur."""
    departments = _get_active_departments(db, company.id)

    if category_name:
        for dept in departments:
            if dept.name.lower() == category_name.lower():
                return dept.id, dept.name

    default_dept = next((d for d in departments if d.is_default), None)
    if default_dept:
        return default_dept.id, default_dept.name

    return None, None


@router.post("/session", response_model=ChatSessionResponse, status_code=201)
@limiter.limit("20/minute", key_func=key_func_by_api_key)
def create_widget_session(
    request: Request,
    response: Response,
    payload: WidgetSessionCreate,
    db: Session = Depends(get_db),
    _backstop: None = Depends(_session_ip_backstop),
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
@limiter.limit("10/minute", key_func=key_func_by_session_id)
@limiter.limit("30/minute", key_func=get_remote_address)
def widget_chat(
    request: Request,
    response: Response,
    payload: WidgetChatRequest,
    db: Session = Depends(get_db),
):
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    company = db.query(Company).filter(Company.id == session.company_id).first()
    if not company or not company.is_active:
        raise HTTPException(status_code=401, detail="This session is no longer active.")

    _check_origin_and_allow(request, response, company)

    if session.awaiting_lead_capture:
        pending = _load_pending_lead(session)
        extracted = extract_lead_info(payload.query)


        name = extracted.get("name") or pending["name"]
        email = extracted.get("email") or pending["email"]
        phone = extracted.get("phone") or pending["phone"]

        if name and email:
            department_id, resolved_category_name = _resolve_department(db, company, pending["category_name"])

            append_lead(
                company_id=str(company.id),
                company_name=company.name,
                name=name,
                email=email,
                phone=phone,
                question=pending["question"],
                session_id=str(session.id),
            )
            lead_row = Lead(
                company_id=company.id,
                session_id=session.id,
                question=pending["question"],
                name=name,
                email=email,
                phone=phone,
                department_id=department_id,
                category_name=resolved_category_name,
            )
            db.add(lead_row)
            session.awaiting_lead_capture = False
            session.pending_lead_query = None
            session.lead_capture_attempts = 0
            db.add(session)

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=LEAD_CAPTURE_THANK_YOU)
            db.add_all([user_msg, assistant_msg])
            db.commit()
            db.refresh(lead_row)

            department_email = next(
                (d.email for d in _get_active_departments(db, company.id) if d.id == department_id),
                None,
            )
            try:
                sent = send_lead_notification(company, lead_row, department_email)
                if sent:
                    lead_row.email_sent = True
                    db.add(lead_row)
                    db.commit()
            except Exception as e:
                
                print(f"[widget_chat] Unexpected error during lead notification: {e}")

            return WidgetChatResponse(session_id=payload.session_id, answer=LEAD_CAPTURE_THANK_YOU)

        session.lead_capture_attempts += 1

        if session.lead_capture_attempts >= MAX_LEAD_CAPTURE_ATTEMPTS:
            session.awaiting_lead_capture = False
            session.pending_lead_query = None
            session.lead_capture_attempts = 0
            db.add(session)

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=LEAD_CAPTURE_GIVE_UP)
            db.add_all([user_msg, assistant_msg])
            db.commit()

            return WidgetChatResponse(session_id=payload.session_id, answer=LEAD_CAPTURE_GIVE_UP)
        else:
            _save_pending_lead(session, pending["question"], pending["category_name"], name, email, phone)
            db.add(session)

            if not name and not email:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_BOTH
            elif not name:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_NAME
            else:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_EMAIL

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=reprompt_text)
            db.add_all([user_msg, assistant_msg])
            db.commit()

            return WidgetChatResponse(session_id=payload.session_id, answer=reprompt_text)

    # --- Branch 2: normal flow ---
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

    answer_text = llm_result["text"]

    # The ONLY detection point for lead capture, anywhere in the app.
    # Exact match against one constant — no substring/keyword checks.
    if answer_text == NO_ANSWER_TEXT:
        active_departments = _get_active_departments(db, company.id)
        category_name = classify_query(payload.query, [d.name for d in active_departments])

        session.awaiting_lead_capture = True
        session.lead_capture_attempts = 0
        _save_pending_lead(session, payload.query, category_name, None, None, None)
        db.add(session)
        answer_text = LEAD_CAPTURE_PROMPT

    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
    assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
    db.add_all([user_msg, assistant_msg])
    db.commit()

    return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)