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
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import tempfile


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
    
STRUCTURAL_EXCLUDE_RE = re.compile(
    r"^(title page|contents|about the author|authors? note|"
    r"by the same author|dedication|acknowledge?ments|copyright|index|"
    r"follow penguin|the beginning of the stories?|"
    r"cover( page)?|colophon|half title|frontispiece|"
    r"the end|epilogue|prologue|afterword|"
    r"appendix|glossary|bibliography)",
    re.IGNORECASE,
)


def _normalize_title_for_match(title: str) -> str:
    
    normalized = (
        title.strip()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return normalized.replace("'", "")

HEADING_LINE_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
MARKDOWN_EMPHASIS_RE = re.compile(r"^\*{1,3}(.+?)\*{1,3}$")
CHAPTER_NUMBERED_RE = re.compile(r"^chapter\s+\d+", re.IGNORECASE) 


def _extract_tier2_font_headings(file_bytes: bytes, page_count: int) -> dict:
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyMuPDF4LLMLoader(tmp_path)
        docs = loader.load()
        full_markdown = "\n".join(d.page_content for d in docs)

        candidates = []
        for match in HEADING_LINE_RE.finditer(full_markdown):
            title = match.group(1).strip()
            if not title:
                continue

            emphasis_match = MARKDOWN_EMPHASIS_RE.match(title)
            if emphasis_match:
                title = emphasis_match.group(1).strip()

            if not title:
                continue
            if STRUCTURAL_EXCLUDE_RE.match(_normalize_title_for_match(title)):
                continue

            candidates.append({"title": title, "page": None})
        
        numbered = [c for c in candidates if CHAPTER_NUMBERED_RE.match(c["title"])]
        if numbered and len(numbered) >= len(candidates) / 2:
            candidates = numbered

        if not candidates:
            return {
                "source": "none",
                "page_count": page_count,
                "raw_toc": [],
                "chapters": [],
                "chapter_count": 0,
            }

        return {
            "source": "font_heuristic",
            "page_count": page_count,
            "raw_toc": [],
            "chapters": candidates,
            "chapter_count": len(candidates),
        }

    except Exception as e:
        print(f"[_extract_tier2_font_headings] failed: {e}")
        return {
            "source": "none",
            "page_count": page_count,
            "raw_toc": [],
            "chapters": [],
            "chapter_count": 0,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"[_extract_tier2_font_headings] temp file cleanup failed: {cleanup_err}")


def extract_document_structure(file_bytes: bytes, filename: str) -> dict:
    
    if not filename.endswith(".pdf"):
        return {
            "source": "none",
            "page_count": None,
            "raw_toc": [],
            "chapters": [],
            "chapter_count": 0,
        }

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
            raw_toc_entries = doc.get_toc()
    except Exception as e:
        print(f"[extract_document_structure] failed to open PDF: {e}")
        return {
            "source": "none",
            "page_count": None,
            "raw_toc": [],
            "chapters": [],
            "chapter_count": 0,
        }

    raw_toc = [
        {"level": level, "title": title, "page": page}
        for level, title, page in raw_toc_entries
    ]

    if not raw_toc:
        return _extract_tier2_font_headings(file_bytes, page_count)

    top_level_entries = [e for e in raw_toc if e["level"] == 1]
    candidates = top_level_entries if top_level_entries else raw_toc

    chapters = [
        {"title": e["title"], "page": e["page"]}
        for e in candidates
        if not STRUCTURAL_EXCLUDE_RE.match(_normalize_title_for_match(e["title"]))
    ]

    return {
        "source": "toc",
        "page_count": page_count,
        "raw_toc": raw_toc,
        "chapters": chapters,
        "chapter_count": len(chapters),
    }

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

def chunk_text(text: str, max_chunk_size: int = 600, chunk_overlap: int = 50) -> list[dict]:
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
    
    
    

