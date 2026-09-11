import json

from fastapi import APIRouter, Depends, HTTPException, Header, Response
from sqlalchemy.orm import Session
from fastapi import Request
from slowapi.util import get_remote_address

from .database import get_db
from .models import ChatSession, ChatMessage, Company
from .schemas import (
    WidgetSessionCreate,
    ChatSessionResponse,
    WidgetChatRequest,
    WidgetChatResponse,
)
from .llm_service import generate_rag_answer_with_memory, extract_lead_info, NO_ANSWER_TEXT
from .lead_export import append_lead
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
    """
    The widget runs inside an iframe hosted on our own backend, so the
    browser's native Origin header on these requests is always our own
    domain, not the site that embedded the widget. The real embedding
    site is instead sent explicitly as X-Embed-Origin (set by chat.js,
    sourced from widget.js's window.location.origin — a value page
    JavaScript cannot forge). We check that value against the company's
    allowed list. The actual CORS response header still echoes the
    real browser Origin, since that's what the browser itself checks.
    """
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
    company = db.query(Company).filter(Company.api_key == x_api_key).first()
    if not company or not company.is_active:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    _check_origin_and_allow(request, response, company)
    return company


def _load_pending_lead(session: ChatSession) -> dict:
    """Parses the JSON blob stored in pending_lead_query into
    {"question", "name", "email", "phone"}. Never raises — falls back to an
    empty shell if the field is missing or somehow malformed, so a bad/old
    value can't crash the request."""
    empty = {"question": "", "name": None, "email": None, "phone": None}
    if not session.pending_lead_query:
        return empty
    try:
        data = json.loads(session.pending_lead_query)
        if not isinstance(data, dict):
            return empty
        return {
            "question": data.get("question", ""),
            "name": data.get("name"),
            "email": data.get("email"),
            "phone": data.get("phone"),
        }
    except Exception:
        return empty


def _save_pending_lead(session: ChatSession, question: str, name, email, phone) -> None:
    session.pending_lead_query = json.dumps(
        {"question": question, "name": name, "email": email, "phone": phone}
    )


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

    # --- Branch 1: this session is mid lead-capture ---
    if session.awaiting_lead_capture:
        pending = _load_pending_lead(session)
        extracted = extract_lead_info(payload.query)

        # Merge: a freshly-extracted field wins if present, otherwise keep
        # whatever was already captured in an earlier reply this round.
        name = extracted.get("name") or pending["name"]
        email = extracted.get("email") or pending["email"]
        phone = extracted.get("phone") or pending["phone"]

        if name and email:
            append_lead(
                company_id=str(company.id),
                company_name=company.name,
                name=name,
                email=email,
                phone=phone,
                question=pending["question"],
                session_id=str(session.id),
            )
            session.awaiting_lead_capture = False
            session.pending_lead_query = None
            session.lead_capture_attempts = 0
            db.add(session)

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=LEAD_CAPTURE_THANK_YOU)
            db.add_all([user_msg, assistant_msg])
            db.commit()

            return WidgetChatResponse(session_id=payload.session_id, answer=LEAD_CAPTURE_THANK_YOU)

        # Incomplete — persist whatever we got, then decide: ask again or give up.
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
            _save_pending_lead(session, pending["question"], name, email, phone)
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
        session.awaiting_lead_capture = True
        session.lead_capture_attempts = 0
        _save_pending_lead(session, payload.query, None, None, None)
        db.add(session)
        answer_text = LEAD_CAPTURE_PROMPT

    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
    assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
    db.add_all([user_msg, assistant_msg])
    db.commit()

    return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)