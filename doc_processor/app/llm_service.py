import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MAX_CONTEXT_TOKENS = 4000 
MODEL_NAME = "gemini-3.6-flash"

def build_safe_context(retrieved_chunks: list[dict], query_text: str) -> tuple[str, int]:
    """
    Iterates through retrieved chunks by relevance score and fits as many full chunks
    as possible into the prompt without exceeding MAX_CONTEXT_TOKENS.
    """
    selected_chunks = []
    current_tokens = 0

    for idx, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = f"\n--- Chunk {idx} (Doc ID: {chunk['document_id']}) ---\n{chunk['chunk_text']}\n"
        
        token_count = client.models.count_tokens(
            model=MODEL_NAME,
            contents=chunk_text
        ).total_tokens

        if current_tokens + token_count > MAX_CONTEXT_TOKENS:
            print(f"Token limit target reached. Omitting remaining chunks starting from index {idx}.")
            break

        selected_chunks.append(chunk_text)
        current_tokens += token_count

    combined_context = "".join(selected_chunks)
    return combined_context, current_tokens


def generate_rag_answer(user_query: str, retrieved_chunks: list[dict]) -> dict:
    context_str, context_token_count = build_safe_context(retrieved_chunks, user_query)

    full_prompt = f"""You are an expert AI assistant answering questions based strictly on retrieved context.
Rules:
1. Use ONLY the provided context chunks to answer the user's question.
2. If the answer cannot be found in the context, explicitly state: "I cannot find the answer in the provided document context."
3. Keep the response concise, factual, and direct.

Context:
{context_str}

Question: {user_query}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt
    )

    prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

    return {
        "text": response.text,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }