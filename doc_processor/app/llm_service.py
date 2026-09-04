import os
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
    """Generates a RAG response, handling optional conversational history and summary detection."""
    chat_history = chat_history or []
    context_str, _ = build_safe_context(
        retrieved_chunks, user_query, chat_history
    )

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
            "You are LiquidLab AI, an intelligent document assistant.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Start your answer IMMEDIATELY with the core facts/summary. DO NOT use conversational greetings or meta-announcements.\n"
            "2. STRICTLY FORBIDDEN PREVIEW PHRASES (NEVER USE THESE): "
            "'Based on the provided document', 'According to the text', 'Based on the text', 'In the document provided', 'Based on the context'.\n"
            "3. Example of BAD response: 'Based on the provided text, the factors affecting...'\n"
            "4. Example of GOOD response: 'The factors affecting global warming are primarily human activities...'\n"
            "5. Do not include chunk tags, document IDs, or metadata inside the answer text.\n"
            "6. If there is no document context or chat history available, reply EXACTLY with: "
            '"I cannot find the answer in the provided document context."\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
        )
    else:
        system_prompt = (
            "You are LiquidLab AI, an intelligent document assistant.\n\n"
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Answer ONLY the specific question asked in the latest user query using the provided context.\n"
            "2. Start your answer IMMEDIATELY with the factual response. NEVER lead with introductory or preamble text.\n"
            "3. STRICTLY FORBIDDEN PREVIEW PHRASES (NEVER USE THESE): "
            "'Based on the provided document', 'According to the text', 'Based on the text', 'In the provided document', 'Based on the context'.\n"
            "4. Example of BAD response: 'Based on the text, the main factors are...'\n"
            "5. Example of GOOD response: 'The main factors are...'\n"
            "6. Do NOT repeat facts or details already given in earlier conversation history unless explicitly asked.\n"
            "7. Do not cite chunk tags, doc IDs, or metadata inside the answer text.\n"
            "8. If the answer cannot be found in the provided context or chat history, reply EXACTLY with: "
            '"I cannot find the answer in the provided document context."\n\n'
            f"--- DOCUMENT CONTEXT ---\n{doc_context}\n"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers=EXTRA_HEADERS,
        temperature=0.3,
    )

    usage = response.usage
    return {
        "text": response.choices[0].message.content.strip(),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }