import re
import csv
import tiktoken
import os

from collections import Counter
import pymupdf as fitz
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

def _extract_tier3_llm_structure(extracted_text: str, page_count: int | None) -> dict:
    
    from .llm_service import generate_chapter_list_llm_from_text

    try:
        result = generate_chapter_list_llm_from_text(extracted_text)

        chapters = [
            {"title": title, "page": None}
            for title in result.get("chapters", [])
        ]

        return {
            "source": "llm_inferred" if chapters else "none",
            "page_count": page_count,
            "raw_toc": [],
            "chapters": chapters,
            "chapter_count": len(chapters),
        }

    except Exception as e:
        print(f"[_extract_tier3_llm_structure] failed: {e}")
        return {
            "source": "none",
            "page_count": page_count,
            "raw_toc": [],
            "chapters": [],
            "chapter_count": 0,
        }


def extract_document_structure(file_bytes: bytes, filename: str) -> dict:
    """Tier 1 (embedded PDF TOC) ONLY.

    Tier 2 (font-size heading heuristics via PyMuPDF4LLMLoader) has been
    removed: on real documents it routinely mistakes decorative front-matter
    text, "(Continued)"/"(Cont.)" markers, and back-matter as new chapters,
    since it has no way to distinguish "styled like a heading" from
    "actually a new section." See llm_service.py's Tier 3 prompt, which
    does that disambiguation instead.

    This is deliberately synchronous and does NOT call the LLM — it is meant
    to run inline during the upload request. Tier 1 is local/fast/free, so
    there's no latency reason to defer it.

    If Tier 1 finds nothing (or the file isn't a PDF at all), this returns
    a "none" result. The caller (main.py's upload endpoint) uses that
    "none" as the signal to schedule Tier 3 (LLM inference) as a background
    task via needs_tier3_llm_fallback(), rather than blocking the upload
    response on an LLM call.
    """

    if filename.endswith(".pdf"):
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                page_count = doc.page_count
                raw_toc_entries = doc.get_toc()
        except Exception as e:
            print(f"[extract_document_structure] failed to open PDF: {e}")
            page_count = None
            raw_toc_entries = []

        raw_toc = [
            {"level": level, "title": title, "page": page}
            for level, title, page in raw_toc_entries
        ]

        # -----------------------------
        # Tier 1: Embedded PDF TOC
        # -----------------------------
        if raw_toc:
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

        # No embedded TOC: Tier 2 has been removed, so this falls straight
        # through to "none", which needs_tier3_llm_fallback() picks up.
        return {
            "source": "none",
            "page_count": page_count,
            "raw_toc": raw_toc,
            "chapters": [],
            "chapter_count": 0,
        }

    # TXT or other non-PDF text-based document: Tier 1/Tier 2 don't apply.
    return {
        "source": "none",
        "page_count": None,
        "raw_toc": [],
        "chapters": [],
        "chapter_count": 0,
    }


def needs_tier3_llm_fallback(structure: dict) -> bool:
    """True when Tier 1 found nothing and Tier 3 (LLM, run in the
    background) should be scheduled. Centralized here so main.py doesn't
    need to know the internal shape of a "none" result."""
    return bool(structure) and structure.get("source") == "none"

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


def count_token(text:str)->int:
    return len(tokenizer.encode(text))

def generate_chunk_token_sequence_csv(chunks: list[dict], output_path: str) -> dict:

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
    
def chunk_text_parent_child(
    text: str,
    parent_chunk_size: int = 900,
    parent_overlap: int = 100,
    child_chunk_size: int = 250,
    child_overlap: int = 30,
) -> list[dict]:
    
    if not text.strip():
        return []

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKENIZER_ENCODING,
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_blocks = parent_splitter.split_text(text)

    results: list[dict] = []
    child_idx = 0

    for parent_idx, parent_block in enumerate(parent_blocks):
        results.append({
            "chunk_index": parent_idx,
            "chunk_text": parent_block,
            "token_count": count_token(parent_block),
            "is_parent": True,
            "parent_index": None,
        })

        child_texts = child_splitter.split_text(parent_block)
        for child_text in child_texts:
            results.append({
                "chunk_index": child_idx,
                "chunk_text": child_text,
                "token_count": count_token(child_text),
                "is_parent": False,
                "parent_index": parent_idx,
            })
            child_idx += 1

    return results

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