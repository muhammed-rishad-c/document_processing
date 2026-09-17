import os
import time
import re
import json as _json 

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from .service import count_token

load_dotenv()

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:9000",
    "X-Title": "LiquidLab RAG App",
}
MAX_CONTEXT_TOKENS = 4000

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.1.99:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma-4-e4b-it")

FALLBACK_BASE_URL = os.getenv("FALLBACK_LLM_BASE_URL", "https://openrouter.ai/api/v1")
FALLBACK_API_KEY = os.getenv("OPEN_API_KEY")
FALLBACK_LLM_MODEL_NAME="openrouter/free"

primary_llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0.3,
    default_headers=EXTRA_HEADERS,
    timeout=30,
    max_retries=0,
)


fallback_llm = ChatOpenAI(
    model=FALLBACK_LLM_MODEL_NAME,
    base_url=FALLBACK_BASE_URL,
    api_key=FALLBACK_API_KEY,
    temperature=0.3,
    default_headers=EXTRA_HEADERS,
    timeout=20,
    max_retries=1,
)


llm = primary_llm.with_fallbacks([fallback_llm])

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{user_query}"),
    ]
)
rag_chain = rag_prompt | llm

NO_ANSWER_TEXT = "I cannot find the answer in the provided document context."
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d[\d\-\s()]{7,}\d)")

SUMMARY_TRIGGER_RE = re.compile(
    r"\b(summari[sz]e|summary|summaries|recap|overview|tl;?dr|"
    r"main\s+points|key\s+points|highlights|gist|brief\s+me(\s+on)?)\b",
    re.IGNORECASE,
)

SUMMARY_OVERVIEW_SEARCH_QUERY = (
    "overview summary main background introduction key takeaways"
)

_SUMMARY_FILLER_WORDS = {
    "give", "me", "a", "an", "the", "of", "on", "for", "about", "regarding",
    "this", "that", "our", "my", "your", "please", "can", "you", "could",
    "i", "want", "need", "get", "provide", "short", "quick", "brief",
    "document", "doc", "chat", "conversation", "session", "everything",
    "all", "it", "thread", "file", "up", "with",
}

GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|yo|howdy|greetings|good\s+(?:morning|afternoon|evening))[\s!.,]*$",
    re.IGNORECASE,
)
THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank\s+you|thx|ty|ok(?:ay)?|cool|great|nice|got\s+it)[\s!.,]*$",
    re.IGNORECASE,
)
SELF_INTRO_RE = re.compile(
    r"\b(?:my\s+name\s+is|i\s*am|i'm|this\s+is|call\s+me)\s+([A-Z][a-zA-Z'-]{1,30})\b"
)

SMALLTALK_GREETING_REPLY = (
    "Hi there! I'm the LiquidLab Assistant -- ask me anything about our services, "
    "solutions, or company."
)

SMALLTALK_THANKS_REPLY = "You're welcome! Anything else I can help with?"

REMEMBER_AS_RE = re.compile(
    r"^\s*(?:remember|note|save)\s+(?:that\s+)?(.+?)\s+as\s+(.+?)[\s.!]*$",
    re.IGNORECASE,
)
REMEMBER_DEF_RE = re.compile(
    r"^\s*(?:remember|note|save)\s+(?:that\s+)?(.+?)\s+(?:means|stands\s+for|is)\s+(.+?)[\s.!]*$",
    re.IGNORECASE,
)

def classify_smalltalk(query: str) -> dict:
    """Cheap regex-only smalltalk / self-intro detector, mirroring
    classify_summary_query. Runs before vector search + lead capture so a
    bare 'hi' or 'my name is X' never falls through to NO_ANSWER_TEXT and
    never triggers lead capture.

    Returns {"is_smalltalk": bool, "reply": str|None, "memory_update": dict|None}.
    memory_update can be set even when is_smalltalk is False (e.g. a real
    question that happens to start with 'I'm Rishad, ...'), so the name is
    still remembered without short-circuiting the real RAG answer.
    """
    text = (query or "").strip()
    result = {"is_smalltalk": False, "reply": None, "memory_update": None}
    if not text:
        return result

    intro_match = SELF_INTRO_RE.search(text)
    intro_name = None
    if intro_match:
        intro_name = intro_match.group(1).strip().rstrip(".,!")
        result["memory_update"] = {"visitor_name": intro_name}

    if GREETING_RE.match(text):
        result["is_smalltalk"] = True
        result["reply"] = SMALLTALK_GREETING_REPLY
        return result

    if THANKS_RE.match(text):
        result["is_smalltalk"] = True
        result["reply"] = SMALLTALK_THANKS_REPLY
        return result

    if intro_name and len(text.split()) <= 8 and looks_like_new_question(text) is not True:
        result["is_smalltalk"] = True
        result["reply"] = f"Nice to meet you, {intro_name}! How can I help you today?"
        return result

    return result

def _normalize_memory_key(raw_key: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")
    return key or raw_key.strip().lower()


def classify_remember_command(query: str) -> dict | None:
    """Detects explicit 'remember X as Y' / 'remember X means Y' style
    instructions. Returns {"key", "display_key", "value"} or None. Checked
    before smalltalk and before RAG so it never gets routed into vector
    search or lead capture.

    'remember camera guiding parking system as cgpa' -> key='cgpa',
    value='camera guiding parking system' (short label -> meaning).
    'remember cgpa means camera guiding parking system' -> same result,
    other phrasing order.
    """
    text = (query or "").strip()
    if not text:
        return None

    m = REMEMBER_AS_RE.match(text)
    if m:
        value, display_key = m.group(1).strip(), m.group(2).strip()
        if value and display_key:
            return {"key": _normalize_memory_key(display_key), "display_key": display_key, "value": value}

    m = REMEMBER_DEF_RE.match(text)
    if m:
        display_key, value = m.group(1).strip(), m.group(2).strip()
        if display_key and value:
            return {"key": _normalize_memory_key(display_key), "display_key": display_key, "value": value}

    return None


def classify_summary_query(query: str) -> dict:
    
    if not query or not query.strip():
        return {"is_summary": False, "mode": None, "search_query": query}

    match = SUMMARY_TRIGGER_RE.search(query)
    if not match:
        return {"is_summary": False, "mode": None, "search_query": query}

    remainder = query[:match.start()] + " " + query[match.end():]
    tokens = re.findall(r"[a-zA-Z0-9']+", remainder.lower())
    content_tokens = [t for t in tokens if t not in _SUMMARY_FILLER_WORDS]

    if not content_tokens:
        return {
            "is_summary": True,
            "mode": "generic",
            "search_query": SUMMARY_OVERVIEW_SEARCH_QUERY,
        }

    return {"is_summary": True, "mode": "targeted", "search_query": query}


def _to_lc_messages(history: list[dict]) -> list:
    """Converts our plain role/content dicts into LangChain message objects.
    Passed via MessagesPlaceholder rather than string-templated, so message
    content is never re-parsed for {..} placeholders — safe even if a past
    turn happens to contain literal braces."""
    lc_messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def reduce_chat_history(chat_history: list[dict], max_history_tokens: int = 1200) -> list[dict]:
    if not chat_history:
        return []

    total_tokens = sum(count_token(msg.get("content", "")) for msg in chat_history)
    if total_tokens <= max_history_tokens:
        return chat_history

    recent_messages = chat_history[-4:]
    older_messages = chat_history[:-4]

    if older_messages:
        summary_lines = []
        for msg in older_messages:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            snippet = msg.get("content", "")[:120].replace("\n", " ")
            summary_lines.append(f"{role_label}: {snippet}...")

        condensed_text = (
            "[Prior Conversation Summary Block]:\n" + "\n".join(summary_lines)
        )
        return [{"role": "assistant", "content": condensed_text}] + recent_messages

    return recent_messages


def build_safe_context(
    retrieved_chunks: list[dict],
    query_text: str,
    chat_history: list[dict] | None = None,
) -> tuple[str, int]:
    selected_chunks = []
    base_tokens = 200 + count_token(query_text)

    if chat_history:
        for msg in chat_history:
            base_tokens += count_token(msg.get("content", ""))

    current_tokens = base_tokens

    for idx, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = f"\n--- Chunk {idx} (Doc ID: {chunk['document_id']}) ---\n{chunk['chunk_text']}\n"
        token_count = count_token(chunk_text)

        if current_tokens + token_count > MAX_CONTEXT_TOKENS:
            print(f"Token limit target reached. Omitting remaining chunks starting from index {idx}.")
            break

        selected_chunks.append(chunk_text)
        current_tokens += token_count

    combined_context = "".join(selected_chunks)
    return combined_context, current_tokens

 
LEAD_IN_PATTERN = re.compile(
    r"^(based on|according to|as (?:stated|mentioned|shown|outlined) in|as per|per|from)\s+"
    r"(the\s+)?(provided\s+|given\s+|available\s+)?"
    r"(document(s)?|context|text|information|content)\b[^.:\-\u2013\u2014\n]*[.:\-\u2013\u2014]\s*",
    re.IGNORECASE,
)

WH_WORDS = ("what", "how", "when", "where", "why", "which", "who")
REQUEST_TERMS = (
    "can you", "could you", "do you", "does it", "is it", "are you",
    "tell me", "explain", "show me", "help with", "reset", "cancel",
    "change", "update", "fix", "support", "pricing", "cost", "price", "refund",
)

EXTRACTION_PROMPT = (
    "Extract the visitor's contact details from the message below. "
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "in exactly this shape: "
    '{{"name": null or string, "email": null or string, "phone": null or string}}. '
    "If a field is not present in the message, use null for it. "
    "Do not guess or invent values.\n\n"
    "Message: {message}"
)

CLASSIFICATION_PROMPT = (
    "Classify the visitor's question below into exactly one of these department names, "
    "or null if none clearly fit or the question is general:\n"
    "{department_list}\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    'in exactly this shape: {{"category": null or "<one of the exact names above>"}}. '
    "Only return a name if it is a genuine match from the list above. "
    "Do not invent a new name. If uncertain, return null.\n\n"
    "Question: {query}"
)

STANDALONE_QUERY_PROMPT = (
    "Below is a conversation between a visitor and a support chatbot, followed by the "
    "visitor's latest message.\n\n"
    "Determine whether the latest message is already a standalone question (fully "
    "understandable on its own, with no missing subject/context) or a follow-up that "
    "depends on the prior conversation to make sense (e.g. it uses pronouns or vague "
    "references like 'it', 'that', 'those', 'these services', 'this feature', or omits a "
    "subject discussed earlier).\n\n"
    "If it is already standalone, return it unchanged (do not paraphrase or reword it). "
    "If it is a follow-up, rewrite it as a single standalone question that REPLACES every "
    "vague pronoun or reference with the specific subject, feature, product, or service "
    "name it refers to, taken from the conversation above. Name the actual thing being "
    "discussed explicitly — do not leave words like 'these services' or 'that' in the "
    "rewritten question. Do not add any information that isn't present in the conversation "
    "or the message itself, and do not list out every sub-detail — just name the subject "
    "clearly enough that the question is fully understandable on its own.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    'in exactly this shape: {{"standalone_query": "<string>"}}.\n\n'
    "--- CONVERSATION ---\n{conversation}\n\n"
    "--- LATEST MESSAGE ---\n{query}"
)

CLASSIFY_LEAD_REPLY_PROMPT = (
    "The visitor was asked for their name and email. Decide if the message "
    "below is trying to provide that (name/email/phone), or is a different "
    "question/request unrelated to giving contact info.\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    'in exactly this shape: {{"intent": "contact_info" or "new_question"}}.\n\n'
    "Message: {message}"
)


def looks_like_new_question(message: str) -> bool | None:
    """Heuristic, no LLM call. True = clearly a question/request.
    False = clearly a contact-info reply or filler. None = ambiguous,
    caller should fall back to classify_lead_reply_intent."""
    text = message.strip()
    if not text:
        return False
    if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text):
        return False

    lower = text.lower()
    if lower.startswith(WH_WORDS) or any(term in lower for term in REQUEST_TERMS):
        return True

    if len(text.split()) <= 3:
        return False

    return None

def classify_lead_reply_intent(message: str) -> str:
    try:
        ai_message = llm.invoke(CLASSIFY_LEAD_REPLY_PROMPT.format(message=message))
        raw = ai_message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = _json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("intent") in ("contact_info", "new_question"):
            return parsed["intent"]
        return "contact_info"
    except Exception as e:
        print(f"[classify_lead_reply_intent] LLM classification failed, defaulting to contact_info: {e}")
        return "contact_info"


def is_diverted_question(message: str, extracted: dict) -> bool:
    if extracted.get("name") or extracted.get("email") or extracted.get("phone"):
        return False

    result = looks_like_new_question(message)
    if result is not None:
        return result

    return classify_lead_reply_intent(message) == "new_question"


def generate_rag_answer_with_memory(
    user_query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
    session_facts: dict | None = None,   # NEW
) -> dict:
    chat_history = chat_history or []

    t_ctx_start = time.perf_counter()
    reduced_history = reduce_chat_history(chat_history)
    context_str, context_tokens = build_safe_context(retrieved_chunks, user_query, reduced_history)
    t_ctx_end = time.perf_counter()

    is_summary_query = classify_summary_query(user_query)["is_summary"]

    doc_context = context_str if context_str else "No specific document context found."

    facts_block = ""
    if session_facts:
        facts_lines = "\n".join(f"- {k}: {v}" for k, v in session_facts.items())
        facts_block = (
            "\n--- FACTS THE VISITOR ASKED YOU TO REMEMBER THIS CONVERSATION ---\n"
            f"{facts_lines}\n"
            "Treat these as authoritative for this conversation, even if they are "
            "not mentioned in the document context above. If the visitor's question "
            "matches one of these facts (by name, abbreviation, or close paraphrase), "
            "answer using it directly.\n"
        )

    if is_summary_query:
        system_prompt = (
            "You are LiquidLab AI, a helpful company chatbot answering visitor questions.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Match the length to the question: a single fact gets 1-2 sentences. A question covering "
            "3 or more distinct items (services, features, technologies, steps) gets a short bulleted list. "
            "Cap any list at 6 items MAXIMUM — pick the 6 most important/representative ones, even if the "
            "context has more. Never cram a multi-item answer into one dense sentence, and never pad a "
            "simple fact with filler.\n"
            "2. Keep each bullet to ONE short phrase (under ~12 words), bolded name first. No sub-explanations, "
            "no extra clauses, no second sentence per bullet.\n"
            "3. If you had to cut items to stay at 6, end with a brief one-line offer like 'Ask if you'd like "
            "the full list.' Do not do this if you listed everything already.\n"
            "4. Start your answer IMMEDIATELY with the core facts/summary. DO NOT use conversational greetings, "
            "meta-announcements, or ANY lead-in phrase referencing 'the document', 'the text', 'the context', "
            "or where the information came from — not even paraphrased. Just state the answer directly.\n"
            "5. Do not add a closing summary sentence after a bulleted list — the list IS the answer, stop there.\n"
            "6. Do not include chunk tags, document IDs, or metadata inside the answer text.\n"
            "7. NEVER output safety check results or metadata like 'User Safety:' or 'Response Safety:'. Output ONLY the answer to the user.\n"
            "8. If there is no document context, chat history, or remembered fact that answers the question, "
            "reply EXACTLY with: "
            f'"{NO_ANSWER_TEXT}"\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
            f"{facts_block}"
        )
    else:
        system_prompt = (
            "You are LiquidLab AI, a helpful company chatbot answering visitor questions.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Match the length to the question: a single fact (e.g. contact info, a yes/no) gets 1-2 short "
            "sentences. A question covering 3 or more distinct items (services, features, technologies, steps) "
            "gets a short bulleted list. Cap any list at 6 items MAXIMUM — pick the 6 most important/representative "
            "ones, even if the context has more. Never cram a multi-item answer into one dense sentence just to keep it short.\n"
            "2. Keep each bullet to ONE short phrase (under ~12 words), bolded name first. No sub-explanations, "
            "no extra clauses, no second sentence per bullet.\n"
            "3. If you had to cut items to stay at 6, end with a brief one-line offer like 'Ask if you'd like "
            "more detail.' Do not do this if you listed everything already.\n"
            "4. FOR FOLLOW-UP QUESTIONS: Answer only the specific new detail asked. NEVER repeat facts, background, or context already given in earlier conversation turns.\n"
            "5. Start your answer IMMEDIATELY with the factual response. NEVER lead with introductory, greeting, "
            "or preamble text — and NEVER reference 'the document', 'the text', 'the context', or where the "
            "information came from, in any phrasing, at the start or anywhere in the answer.\n"
            "6. Do not add a closing summary sentence after a bulleted list — the list IS the answer, stop there.\n"
            "7. Do not cite chunk tags, doc IDs, or metadata inside the answer text.\n"
            "8. NEVER output safety check results or metadata like 'User Safety:' or 'Response Safety:'. Output ONLY the answer to the user.\n"
            "9. If the answer cannot be found in the provided context, chat history, or remembered facts, "
            "reply EXACTLY with: "
            f'"{NO_ANSWER_TEXT}"\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
            f"{facts_block}"
        )

    t_llm_start = time.perf_counter()
    ai_message = rag_chain.invoke(
        {
            "system_prompt": system_prompt,
            "chat_history": _to_lc_messages(reduced_history),
            "user_query": user_query,
        }
    )
    t_llm_end = time.perf_counter()

    model_used = (
        ai_message.response_metadata.get("model_name")
        or ai_message.response_metadata.get("model")
        or "unknown"
    )
    is_fallback = model_used.strip().lower() not in (MODEL_NAME.strip().lower(), "")
    print(
        f"[generate_rag_answer_with_memory] model_used={model_used} "
        f"fallback={is_fallback} llm_ms={round((t_llm_end - t_llm_start) * 1000, 2)}"
    )

    raw_text = ai_message.content.strip()
    cleaned_lines = [
        line
        for line in raw_text.splitlines()
        if not line.strip().startswith(("User Safety:", "Response Safety:"))
    ]
    cleaned_text = "\n".join(cleaned_lines).strip()

    match = LEAD_IN_PATTERN.match(cleaned_text)
    if match:
        remainder = cleaned_text[match.end():].lstrip()
        if remainder:
            cleaned_text = remainder[0].upper() + remainder[1:]

    lines = cleaned_text.splitlines()
    bullet_idx = [i for i, ln in enumerate(lines) if ln.strip().startswith(("- ", "* ", "\u2022 "))]
    if len(bullet_idx) > 6:
        cutoff = bullet_idx[5]
        cleaned_text = "\n".join(lines[: cutoff + 1]).strip()
        cleaned_text += "\n\nAsk if you'd like the full list."

    if not cleaned_text:
        cleaned_text = NO_ANSWER_TEXT

    usage = ai_message.usage_metadata or {}

    return {
        "text": cleaned_text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "context_tokens": context_tokens,
        "context_prep_ms": round((t_ctx_end - t_ctx_start) * 1000, 2),
        "llm_generation_ms": round((t_llm_end - t_llm_start) * 1000, 2),
        "model_used": model_used,
    }
    


def extract_lead_info(message: str) -> dict:
    
    result = {"name": None, "email": None, "phone": None}

    try:
        ai_message = llm.invoke(EXTRACTION_PROMPT.format(message=message))
        raw = ai_message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = _json.loads(raw)
        if isinstance(parsed, dict):
            result["name"] = parsed.get("name") or None
            result["email"] = parsed.get("email") or None
            result["phone"] = parsed.get("phone") or None
    except Exception as e:
        print(f"[extract_lead_info] LLM extraction failed, falling back to regex only: {e}")

    
    if not result["email"] or not EMAIL_PATTERN.fullmatch(result["email"].strip()):
        email_match = EMAIL_PATTERN.search(message)
        result["email"] = email_match.group(0) if email_match else None

    if not result["phone"]:
        phone_match = PHONE_PATTERN.search(message)
        result["phone"] = phone_match.group(0).strip() if phone_match else None

    if isinstance(result["name"], str):
        result["name"] = result["name"].strip() or None

    return result

def resolve_standalone_query(query: str, chat_history: list[dict] | None = None) -> str:
    """Resolves a possibly context-dependent visitor message into a standalone
    question, using recent chat history. Used only on the NO_ANSWER_TEXT
    fallback path (see widget.py) — never touches ChatMessage storage, and
    never raises. On empty history there is nothing to resolve against, so
    the LLM call is skipped and the raw query is returned unchanged. On any
    LLM/parsing failure, falls back to the raw query, exactly like
    extract_lead_info / classify_query do."""
    if not chat_history:
        return query

    recent_history = chat_history[-6:]
    conversation_lines = []
    for msg in recent_history:
        role_label = "User" if msg.get("role") == "user" else "Assistant"
        conversation_lines.append(f"{role_label}: {msg.get('content', '')}")
    conversation = "\n".join(conversation_lines)

    if not conversation.strip():
        return query

    try:
        ai_message = llm.invoke(
            STANDALONE_QUERY_PROMPT.format(conversation=conversation, query=query)
        )
        raw = ai_message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = _json.loads(raw)

        if isinstance(parsed, dict):
            standalone_query = parsed.get("standalone_query")
            if isinstance(standalone_query, str) and standalone_query.strip():
                return standalone_query.strip()

        return query
    except Exception as e:
        print(f"[resolve_standalone_query] LLM resolution failed, falling back to raw query: {e}")
        return query

def classify_query(query: str, department_names: list[str]) -> str | None:
    
    if not department_names:
        return None

    department_list = "\n".join(f"- {name}" for name in department_names)

    try:
        ai_message = llm.invoke(
            CLASSIFICATION_PROMPT.format(department_list=department_list, query=query)
        )
        raw = ai_message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = _json.loads(raw)

        if isinstance(parsed, dict):
            category = parsed.get("category")
            if isinstance(category, str):
                category = category.strip()
                for name in department_names:
                    if name.lower() == category.lower():
                        return name

        return None
    except Exception as e:
        print(f"[classify_query] LLM classification failed, defaulting to None: {e}")
        return None