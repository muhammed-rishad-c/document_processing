from uuid import UUID
from datetime import datetime
from pydantic import BaseModel,ConfigDict
from typing import List,Dict,Any,Optional

class DocumentStats(BaseModel):
    total_words: int
    total_characters: int
    total_sentences: int
    total_paragraphs: int
    top_10_words: dict[str, int]
    
    model_config = ConfigDict(from_attributes=True)
    
class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: str
    upload_time: datetime
    stats: DocumentStats

    class Config:
        from_attributes = True
        
class DocumentDetailResponse(DocumentResponse):
    extracted_text: str
    
class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    total_chunks: int
    stats: Dict[str, Any]
    message: str


class SearchResponse(BaseModel):
    query: str
    occurrences: int
    matching_sentences: List[str]

class SimilarityRequest(BaseModel):
    doc_id_1: UUID
    doc_id_2: UUID

class SimilarityResponse(BaseModel):
    doc_id_1: UUID
    doc_id_2: UUID
    similarity_score: float
    
class SimilarDocumentMatch(BaseModel):
    document_id: str
    filename: str
    score: float


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    document_id: Optional[UUID] = None 

class ChunkSearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    chunk_text: str
    token_count: int
    similarity_score: float

class SemanticSearchResponse(BaseModel):
    query: str
    results: List[ChunkSearchResult]
    
class RAGRequest(BaseModel):
    query: str
    top_k: int = 3
    document_id: Optional[str] = None

class ChunkSource(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    similarity_score: float

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkSource]