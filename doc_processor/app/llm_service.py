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

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPEN_API_KEY"))
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openrouter/free")

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0.3,
    default_headers=EXTRA_HEADERS,
)

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

EXTRACTION_PROMPT = (
    "Extract the visitor's contact details from the message below. "
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "in exactly this shape: "
    '{{"name": null or string, "email": null or string, "phone": null or string}}. '
    "If a field is not present in the message, use null for it. "
    "Do not guess or invent values.\n\n"
    "Message: {message}"
)

def generate_rag_answer_with_memory(
    user_query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> dict:
    chat_history = chat_history or []

    t_ctx_start = time.perf_counter()
    reduced_history = reduce_chat_history(chat_history)
    context_str, context_tokens = build_safe_context(retrieved_chunks, user_query, reduced_history)
    t_ctx_end = time.perf_counter()

    summary_keywords = ["summarize", "summary", "recap", "overview", "main points"]
    is_summary_query = any(kw in user_query.lower() for kw in summary_keywords)

    doc_context = context_str if context_str else "No specific document context found."

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
            "8. If there is no document context or chat history available, reply EXACTLY with: "
            f'"{NO_ANSWER_TEXT}"\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
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
            "9. If the answer cannot be found in the provided context or chat history, reply EXACTLY with: "
            f'"{NO_ANSWER_TEXT}"\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
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
    }
    


def extract_lead_info(message: str) -> dict:
    """Best-effort extraction of name/email/phone from a free-text visitor reply.
    Never raises — always returns a dict with the three keys, using None for
    anything it couldn't confidently find. LLM does name extraction (no
    reliable regex for that); email/phone are validated/recovered with regex
    since those have unambiguous formats."""
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