import os
from openai import OpenAI
from dotenv import load_dotenv
from .service import count_token

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_API_KEY")
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

    system_prompt = """You are an expert AI assistant answering questions based strictly on retrieved context.
Rules:
1. Use ONLY the provided context chunks to answer the user's question.
2. If the answer cannot be found in the context, explicitly state: "I cannot find the answer in the provided document context."
3. Keep the response concise, factual, and direct."""

    user_content = f"Context:\n{context_str}\n\nQuestion: {user_query}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Document RAG App"
        },
        temperature=0.3
    )
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    

    return {
        "text": response.choices[0].message.content,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }