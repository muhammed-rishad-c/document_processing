import io
import re
from collections import Counter
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tiktoken

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
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            extracted_text = extracted_text.replace("\x00", "")
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
        return {
            
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

def chunk_text(text:str,max_chunk_size:int=300,chunk_overlap:int=50)->list[dict]:
    
    tokens=tokenizer.encode(text)
    total_token=len(tokens)
    
    if total_token==0:
        return []
    
    chunks=[]
    chunk_index=0
    start_idx=0
    
    step=max_chunk_size-chunk_overlap
    if step<=0:
        raise ValueError("max chunk size is less than overlap size")
    
    while start_idx<total_token:
        
        end_idx=min(start_idx+max_chunk_size,total_token)
        chunk_tokens=tokens[start_idx:end_idx]
        chunk_text_str=tokenizer.decode(chunk_tokens)
        
        chunks.append({
            "chunk_index":chunk_index,
            "chunk_text":chunk_text_str,
            "token_count":len(chunk_tokens)
        })
        
        chunk_index+=1
        if end_idx>=total_token:
            break
        
        start_idx+=step
        
    return chunks
    
    
    

