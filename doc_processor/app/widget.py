import json
import time

from fastapi import APIRouter, Depends, HTTPException, Header, Response
from sqlalchemy.orm import Session
from fastapi import Request
from slowapi.util import get_remote_address

from .database import get_db
from .models import ChatSession, ChatMessage, Company, Lead, CompanyDepartment,Document
from .schemas import (
    WidgetSessionCreate,
    ChatSessionResponse,
    WidgetChatRequest,
    WidgetChatResponse,
)
from .llm_service import (
    generate_rag_answer_with_memory,
    extract_lead_info, classify_query,
    resolve_standalone_query, 
    is_diverted_question,
    classify_remember_command, classify_conversational_intent_dynamic,
    is_greeting_or_thanks, is_lead_worthy_question,
    classify_summary_query, classify_summary_target, generate_chat_summary,
    classify_structural_query, answer_structural_query,
    MAX_SESSION_MEMORY_KEYS,
    NO_ANSWER_TEXT
    )
from .lead_export import append_lead
from .email_service import send_lead_notification
from .vector_store import search_similar_chunks
from .rate_limit import limiter, key_func_by_api_key, key_func_by_session_id


GREETING_TEXT = (
    "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, "
    "solutions, or company -- happy to help."
)

MAX_LEAD_CAPTURE_ATTEMPTS = 2

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
NO_MATCH_SOFT_REPLY = (
    "I'm not sure I caught that one. I can help with questions about "
    "{company} — our services, solutions, or how to get in touch. "
    "What would you like to know?"
)
AUTO_LEAD_FORWARDED_TEXT = (
    "I couldn't find that in our docs, but I've passed it along to our team — "
    "they'll follow up if needed."
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
    
    empty = {
        "question": "",
        "resolved_question": "",
        "category_name": None,
        "name": None,
        "email": None,
        "phone": None,
    }
    if not session.pending_lead_query:
        return empty
    try:
        data = json.loads(session.pending_lead_query)
        if not isinstance(data, dict):
            return empty
        raw_question = data.get("question", "")
        return {
            "question": raw_question,
            "resolved_question": data.get("resolved_question") or raw_question,
            "category_name": data.get("category_name"),
            "name": data.get("name"),
            "email": data.get("email"),
            "phone": data.get("phone"),
        }
    except Exception:
        return empty


def _save_pending_lead(session: ChatSession, question: str, resolved_question: str, category_name, name, email, phone) -> None:
    session.pending_lead_query = json.dumps(
        {
            "question": question,
            "resolved_question": resolved_question,
            "category_name": category_name,
            "name": name,
            "email": email,
            "phone": phone,
        }
    )
    
def _get_active_departments(db: Session, company_id) -> list[CompanyDepartment]:
    return (
        db.query(CompanyDepartment)
        .filter(CompanyDepartment.company_id == company_id, CompanyDepartment.is_active.is_(True))
        .all()
    )


def _resolve_department(db: Session, company: Company, category_name: str | None):
    
    departments = _get_active_departments(db, company.id)

    if category_name:
        for dept in departments:
            if dept.name.lower() == category_name.lower():
                return dept.id, dept.name

    default_dept = next((d for d in departments if d.is_default), None)
    if default_dept:
        return default_dept.id, default_dept.name

    return None, None

def _answer_with_rag(
    request: Request,
    db: Session,
    session: ChatSession,
    company: Company,
    payload: WidgetChatRequest,
    initial_stage_timings: dict | None = None,
) -> WidgetChatResponse:
    stage_timings: dict = dict(initial_stage_timings) if initial_stage_timings else {}

    all_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == payload.session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    history_payload = [{"role": msg.role, "content": msg.content} for msg in all_messages]
    
    structural = classify_structural_query(payload.query)
    if structural["is_structural"]:
        doc = db.query(Document).filter(Document.id == company.document_id).first()
        answer_text = answer_structural_query(structural["kind"], doc.structure if doc else None)
        user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
        assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
        db.add_all([user_msg, assistant_msg])
        db.commit()
        return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)


    if classify_summary_query(payload.query)["is_summary"] and classify_summary_target(payload.query) == "chat":
        unsummarized = history_payload[session.summarized_count:]
        answer_text = generate_chat_summary(unsummarized, session.running_summary or "")
        user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
        assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
        db.add_all([user_msg, assistant_msg])
        db.commit()
        return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)

    retrieved_chunks = search_similar_chunks(
        query_text=payload.query,
        top_k=7,
        document_id=str(company.document_id),
        db_session=db,
        timing_out=stage_timings,
    )

    try:
            
        llm_result = generate_rag_answer_with_memory(
            user_query=payload.query,
            retrieved_chunks=retrieved_chunks,
            chat_history=history_payload,
            session_facts=session.session_memory or None,
            session_summary=session.running_summary,
            session_summary_count=session.summarized_count,
        )
    except Exception:
        request.state.stage_timings = stage_timings
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again shortly.",
        )

    stage_timings["context_prep_ms"] = llm_result.get("context_prep_ms", 0)
    stage_timings["llm_generation_ms"] = llm_result.get("llm_generation_ms", 0)

    answer_text = llm_result["text"]

    if answer_text == NO_ANSWER_TEXT and not is_lead_worthy_question(payload.query):
        
        answer_text = NO_MATCH_SOFT_REPLY.format(company=company.name)

    elif answer_text == NO_ANSWER_TEXT:
        t0 = time.perf_counter()
        resolved_query = resolve_standalone_query(payload.query, history_payload)
        stage_timings["resolve_standalone_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        active_departments = _get_active_departments(db, company.id)

        t0 = time.perf_counter()
        category_name = classify_query(resolved_query, [d.name for d in active_departments])
        stage_timings["classify_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if session.captured_name and session.captured_email:
            department_id, resolved_category_name = _resolve_department(db, company, category_name)

            t0 = time.perf_counter()
            append_lead(
                company_id=str(company.id),
                company_name=company.name,
                name=session.captured_name,
                email=session.captured_email,
                phone=session.captured_phone,
                question=resolved_query,
                session_id=str(session.id),
            )
            stage_timings["excel_export_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            lead_row = Lead(
                company_id=company.id,
                session_id=session.id,
                question=resolved_query,
                name=session.captured_name,
                email=session.captured_email,
                phone=session.captured_phone,
                department_id=department_id,
                category_name=resolved_category_name,
            )
            db.add(lead_row)
            db.commit()
            db.refresh(lead_row)

            department_email = next(
                (d.email for d in active_departments if d.id == department_id),
                None,
            )
            t0 = time.perf_counter()
            try:
                sent = send_lead_notification(company, lead_row, department_email)
                if sent:
                    lead_row.email_sent = True
                    db.add(lead_row)
                    db.commit()
            except Exception as e:
                print(f"[_answer_with_rag] Unexpected error during lead notification: {e}")
            stage_timings["email_notification_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            answer_text = AUTO_LEAD_FORWARDED_TEXT
        else:
            session.awaiting_lead_capture = True
            session.lead_capture_attempts = 0
            _save_pending_lead(session, payload.query, resolved_query, category_name, None, None, None)
            db.add(session)
            answer_text = LEAD_CAPTURE_PROMPT

    session.running_summary = llm_result.get("updated_summary", session.running_summary)
    session.summarized_count = llm_result.get("summarized_count", session.summarized_count)

    user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
    assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
    db.add_all([user_msg, assistant_msg, session])
    db.commit()

    request.state.stage_timings = stage_timings

    return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)


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

    if not session.awaiting_lead_capture:
        remember_cmd = classify_remember_command(payload.query)
        if remember_cmd:
            memory = dict(session.session_memory or {})

            if remember_cmd["key"] not in memory and len(memory) >= MAX_SESSION_MEMORY_KEYS:
                ack_text = (
                    "I've already got quite a few things remembered for this chat, "
                    "so I can't add another one right now."
                )
            else:
                memory[remember_cmd["key"]] = remember_cmd["value"]
                session.session_memory = memory
                db.add(session)

                ack_text = (
                    f"Got it — I'll remember that {remember_cmd['display_key']} "
                    f"means \"{remember_cmd['value']}\"."
                )

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=ack_text)
            db.add_all([user_msg, assistant_msg])
            db.commit()
            return WidgetChatResponse(session_id=payload.session_id, answer=ack_text)

        existing_memory = session.session_memory or {}

        # Only the last couple of turns are needed to disambiguate a short
        # reply like "no" or "sure" — no need to load the full history here.
        recent_msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == payload.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(4)
            .all()
        )
        recent_history = [
            {"role": m.role, "content": m.content} for m in reversed(recent_msgs)
        ]

        smalltalk = classify_conversational_intent_dynamic(
            payload.query,
            chat_history=recent_history,
            company_name=company.name,
            visitor_name=existing_memory.get("visitor_name"),
            is_first_turn=False,  # session/create already sent GREETING_TEXT
        )

        if smalltalk["memory_update"] and smalltalk["is_smalltalk"]:
            memory = dict(existing_memory)
            memory.update(smalltalk["memory_update"])
            session.session_memory = memory
            db.add(session)

        if smalltalk["is_smalltalk"]:
            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=smalltalk["reply"])
            db.add_all([user_msg, assistant_msg])
            db.commit()
            return WidgetChatResponse(session_id=payload.session_id, answer=smalltalk["reply"])

        if classify_summary_query(payload.query)["is_summary"] and classify_summary_target(payload.query) == "chat":
            all_messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == payload.session_id)
                .order_by(ChatMessage.created_at.asc())
                .all()
            )
            history_payload = [{"role": m.role, "content": m.content} for m in all_messages]
            unsummarized = history_payload[session.summarized_count:]
            answer_text = generate_chat_summary(unsummarized, session.running_summary or "")

            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=answer_text)
            db.add_all([user_msg, assistant_msg])
            db.commit()
            return WidgetChatResponse(session_id=payload.session_id, answer=answer_text)

    if session.awaiting_lead_capture:
        stage_timings: dict = {}
        pending = _load_pending_lead(session)

        stripped_query = (payload.query or "").strip()
        if is_greeting_or_thanks(stripped_query):
            # This branch used to re-prompt forever — it never checked or
            # incremented lead_capture_attempts, so a visitor who just kept
            # saying "ok" got asked indefinitely. Now it counts toward the
            # same limit as every other non-answer in this flow.
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

            if not pending["name"] and not pending["email"]:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_BOTH
            elif not pending["name"]:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_NAME
            else:
                reprompt_text = LEAD_CAPTURE_REPROMPT_MISSING_EMAIL

            db.add(session)
            user_msg = ChatMessage(session_id=payload.session_id, role="user", content=payload.query)
            assistant_msg = ChatMessage(session_id=payload.session_id, role="assistant", content=reprompt_text)
            db.add_all([user_msg, assistant_msg])
            db.commit()

            return WidgetChatResponse(session_id=payload.session_id, answer=reprompt_text)

        t0 = time.perf_counter()
        extracted = extract_lead_info(payload.query)
        stage_timings["extract_lead_info_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        diverted = is_diverted_question(payload.query, extracted)
        stage_timings["diversion_check_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if diverted:
            session.awaiting_lead_capture = False
            session.pending_lead_query = None
            session.lead_capture_attempts = 0
            db.add(session)
            return _answer_with_rag(request, db, session, company, payload, initial_stage_timings=stage_timings)

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
                question=pending["resolved_question"],
                session_id=str(session.id),
            )
            lead_row = Lead(
                company_id=company.id,
                session_id=session.id,
                question=pending["resolved_question"],
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
            session.captured_name = name
            session.captured_email = email
            session.captured_phone = phone
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
            _save_pending_lead(
                session, pending["question"], pending["resolved_question"], pending["category_name"], name, email, phone
            )
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

    return _answer_with_rag(request, db, session, company, payload)