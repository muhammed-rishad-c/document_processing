import io
import re
import csv
import tiktoken
import os,uuid

from collections import Counter
import pymupdf as fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


TOKENIZER_ENCODING="cl100k_base"
tokenizer=tiktoken.get_encoding(TOKENIZER_ENCODING)


STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "he", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with", "or", "this", "but"
}

def extract_text_from_file(file_bytes:bytes,filename:str)->tuple[str,str]:
    try:
        
        if filename.endswith(".txt"):
            file_type=".txt"
            extracted_text=file_bytes.decode("utf-8",errors="ignore").replace("\x00", "")
        elif filename.endswith(".pdf"):
            file_type = ".pdf"
            extracted_text = ""
            
            with fitz.open(stream=file_bytes,filetype="pdf") as doc:
                for page in doc:
                    text=page.get_text("text")
                    if text:
                        extracted_text+=text+"\n"
            extracted_text=extracted_text.replace("\x00","")
        else:
            raise ValueError("Unsupported file type. Only .txt and .pdf are allowed.")

        cleaned_text = extracted_text.strip()
        if not cleaned_text:
            raise ValueError("Document is empty or unreadable.")

        return cleaned_text, file_type
    except ValueError:
        raise 
    except Exception as e:
        raise ValueError(f"failed to process file : {str(e)}")

def calculate_document_stats(text: str) -> dict:
    try:
        char_count = len(text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        filtered_words = [w for w in words if w not in STOP_WORDS and not w.isdigit()]
        top_10_words = dict(Counter(filtered_words).most_common(10))
        
        return {
            "total_words": word_count,
            "total_characters": char_count,
            "total_sentences": sentence_count,
            "total_paragraphs": paragraph_count,
            "top_10_words": top_10_words
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate document statistics: {str(e)}")


def search_text_in_document(text: str, query: str) -> dict:
    try:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
            
        query_lower = query.lower()
        occurrences = len(re.findall(re.escape(query_lower), text.lower()))
        sentences = re.split(r'(?<=[.!?])\s+', text)
        matching_sentences = [
            s.strip() for s in sentences if query_lower in s.lower()
        ]
        
        return {
            "query": query,
            "occurrences": occurrences,
            "matching_sentences": matching_sentences
        }
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"error occurred during search execution: {str(e)}")


def count_token(text:str)->int:
    return len(tokenizer.encode(text))

def generate_chunk_token_sequence_csv(chunks: list[dict], output_path: str) -> dict:

    import csv
    if not chunks:
        raise ValueError("Cannot generate token report for an empty chunk list.")
    rows = []
    for chunk in chunks:
        chunk_index = chunk["chunk_index"]
        token_ids = tokenizer.encode(chunk["chunk_text"])
        for position, token_id in enumerate(token_ids):
            try:
                token_text = tokenizer.decode([token_id])
            except Exception:
                token_text = "<decode_error>"
            rows.append((chunk_index, position, token_id, token_text))

    rows.sort(key=lambda r: r[2])

    seen_token_ids = set()
    unique_rows = []
    for row in rows:
        token_id = row[2]
        if token_id not in seen_token_ids:
            seen_token_ids.add(token_id)
            unique_rows.append(row)
    rows = unique_rows
   

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_index", "position", "token_id", "token_text"])
        writer.writerows(rows)
    return {
        "total_chunks": len(chunks),
        "total_token_rows": len(rows),
        "output_path": output_path,
    }

def chunk_text(text: str, max_chunk_size: int = 300, chunk_overlap: int = 50) -> list[dict]:
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    return [
        {
            "chunk_index": idx,
            "chunk_text": chunk,
            "token_count": count_token(chunk),
        }
        for idx, chunk in enumerate(raw_chunks)
    ]
    
    
    

