import os
from openai import OpenAI
from dotenv import load_dotenv
from .service import count_token


load_dotenv()

OPENROUTER_API_KEY=os.getenv("OPEN_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MAX_CONTEXT_TOKENS = 4000 
MODEL_NAME = "openrouter/free"

def build_safe_context(retrieved_chunks: list[dict], query_text: str) -> tuple[str, int]:
    """
    Iterates through retrieved chunks by relevance score and fits as many full chunks
    as possible into the prompt without exceeding MAX_CONTEXT_TOKENS.
    """
    selected_chunks = []
    current_tokens = 0

    for idx, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = f"\n--- Chunk {idx} (Doc ID: {chunk['document_id']}) ---\n{chunk['chunk_text']}\n"
        
        token_count=count_token(chunk_text)

        if current_tokens + token_count > MAX_CONTEXT_TOKENS:
            print(f"Token limit target reached. Omitting remaining chunks starting from index {idx}.")
            break

        selected_chunks.append(chunk_text)
        current_tokens += token_count

    combined_context = "".join(selected_chunks)
    return combined_context, current_tokens

def generate_rag_answer(user_query: str, retrieved_chunks: list[dict]) -> dict:
    context_str, context_token_count = build_safe_context(retrieved_chunks, user_query)

    system_prompt = (
        "You are an expert AI assistant answering questions based strictly on retrieved context.\n\n"
        "STRICT RESPONSE RULES:\n"
        "1. Answer the query directly, concisely, and factually.\n"
        "2. NEVER use preamble phrases like 'Based on the provided document context', 'According to the text', "
        "'Based on the context', or 'Chunk 1 states'. Jump straight into the answer.\n"
        "3. Do not include metadata citations or chunk references in your text.\n"
        "4. If the answer cannot be found in the context, reply EXACTLY with: "
        "\"I cannot find the answer in the provided document context.\""
    )

    user_content = f"Context:\n{context_str}\n\nQuestion: {user_query}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        extra_headers={
            "HTTP-Referer": "http://localhost:9000",
            "X-Title": "LiquidLab RAG App"
        },
        temperature=0.3
    )
    
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return {
        "text": response.choices[0].message.content.strip(),
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }


def generate_rag_answer_with_memory(
    user_query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict]
) -> dict:

    context_str, _ = build_safe_context(retrieved_chunks, user_query)

    system_prompt = (
        "You are LiquidLab AI, an intelligent document assistant.\n\n"
        "STRICT RESPONSE RULES:\n"
        "1. Answer the user's question directly, factually, and concisely using the provided context.\n"
        "2. Maintain continuity with the previous conversation history if relevant.\n"
        "3. NEVER use preamble phrases such as 'Based on the provided document context', 'According to the text', "
        "'Based on the context', or 'In the provided document'. Lead directly with the factual response.\n"
        "4. Do not cite chunk tags or metadata inside the answer text.\n"
        "5. If the answer cannot be found in the provided context or chat history, reply EXACTLY with: "
        "\"I cannot find the answer in the provided document context.\"\n\n"
        f"--- DOCUMENT CONTEXT ---\n{context_str if context_str else 'No specific document context found.'}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:9000",
            "X-Title": "LiquidLab RAG App"
        },
        temperature=0.3
    )

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return {
        "text": response.choices[0].message.content.strip(),
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }