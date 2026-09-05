import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from .service import count_token

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPEN_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MAX_CONTEXT_TOKENS = 4000
MODEL_NAME = "openrouter/free"
EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:9000",
    "X-Title": "LiquidLab RAG App",
}

def reduce_chat_history(chat_history: list[dict], max_history_tokens: int=1200) -> list[dict]:
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
            print(
                f"Token limit target reached. Omitting remaining chunks starting from index {idx}."
            )
            break

        selected_chunks.append(chunk_text)
        current_tokens += token_count

    combined_context = "".join(selected_chunks)
    return combined_context, current_tokens

def generate_rag_answer_with_memory(
    user_query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> dict:
    chat_history = chat_history or []
    
    t_ctx_start = time.perf_counter()

    reduced_history = reduce_chat_history(chat_history)

    context_str, context_tokens = build_safe_context(     
        retrieved_chunks, user_query, reduced_history
    )
    t_ctx_end = time.perf_counter()
    
    

    summary_keywords = [
        "summarize",
        "summary",
        "recap",
        "overview",
        "main points",
        "about",
    ]
    is_summary_query = any(
        kw in user_query.lower() for kw in summary_keywords
    )

    doc_context = (
        context_str if context_str else "No specific document context found."
    )

    if is_summary_query:
        system_prompt = (
            "You are LiquidLab AI, a concise document assistant.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. BE COMPACT: Keep summaries focused and brief. Use max 3-4 bullet points or 1 short paragraph (under 80 words).\n"
            "2. Start your answer IMMEDIATELY with the core facts/summary. DO NOT use conversational greetings or meta-announcements.\n"
            "3. STRICTLY FORBIDDEN PREVIEW PHRASES (NEVER USE THESE): "
            "'Based on the provided document', 'According to the text', 'Based on the text', 'In the document provided', 'Based on the context'.\n"
            "4. Example of BAD response: 'Based on the provided text, the factors affecting...'\n"
            "5. Example of GOOD response: 'The factors affecting global warming are primarily human activities...'\n"
            "6. Do not include chunk tags, document IDs, or metadata inside the answer text.\n"
            "7. NEVER output safety check results or metadata like 'User Safety:' or 'Response Safety:'. Output ONLY the answer to the user.\n"
            "8. If there is no document context or chat history available, reply EXACTLY with: "
            '"I cannot find the answer in the provided document context."\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
        )
    else:
        system_prompt = (
            "You are LiquidLab AI, a concise document assistant.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. BE EXTREMELY COMPACT AND DIRECT: Keep responses under 2-3 short sentences (max 50 words).\n"
            "2. FOR FOLLOW-UP QUESTIONS: Answer only the specific new detail asked. NEVER repeat facts, background, or context already given in earlier conversation turns.\n"
            "3. Start your answer IMMEDIATELY with the factual response. NEVER lead with introductory, greeting, or preamble text.\n"
            "4. STRICTLY FORBIDDEN PREVIEW PHRASES (NEVER USE THESE): "
            "'Based on the provided document', 'According to the text', 'Based on the text', 'In the provided document', 'Based on the context'.\n"
            "5. Example of BAD response: 'Based on the text, the main factors are...'\n"
            "6. Example of GOOD response: 'The main factors are...'\n"
            "7. Do not cite chunk tags, doc IDs, or metadata inside the answer text.\n"
            "8. NEVER output safety check results or metadata like 'User Safety:' or 'Response Safety:'. Output ONLY the answer to the user.\n"
            "9. If the answer cannot be found in the provided context or chat history, reply EXACTLY with: "
            '"I cannot find the answer in the provided document context."\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in reduced_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    t_llm_start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers=EXTRA_HEADERS,
        temperature=0.3,
    )
    t_llm_end = time.perf_counter()
    
    

    raw_text = response.choices[0].message.content.strip()

    cleaned_lines = [
        line
        for line in raw_text.splitlines()
        if not line.strip().startswith(("User Safety:", "Response Safety:"))
    ]
    cleaned_text = "\n".join(cleaned_lines).strip()

    if not cleaned_text:
        cleaned_text = "I cannot find the answer in the provided document context."

    usage = response.usage
    return {
        "text": cleaned_text,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "context_tokens": context_tokens,       
        "context_prep_ms": round((t_ctx_end - t_ctx_start) * 1000, 2),      
        "llm_generation_ms": round((t_llm_end - t_llm_start) * 1000, 2),   
    }