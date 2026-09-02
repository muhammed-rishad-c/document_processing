import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_rag_answer(user_query: str, retrieved_chunks: list[dict]) -> str:
    # Build context payload from Qdrant results
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"\n--- Chunk {idx} (Doc ID: {chunk['document_id']}) ---\n"
        context_str += f"{chunk['chunk_text']}\n"

    # Single combined prompt containing system instructions, context, and query
    full_prompt = f"""You are an expert AI assistant answering questions based strictly on retrieved context.
Rules:
1. Use ONLY the provided context chunks to answer the user's question.
2. If the answer cannot be found in the context, explicitly state: "I cannot find the answer in the provided document context."
3. Keep the response concise, factual, and direct.

Context:
{context_str}

Question: {user_query}
"""

    # Call the active model endpoint
    response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=full_prompt
)

    return response.text