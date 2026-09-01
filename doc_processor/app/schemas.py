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


class SearchResponse(BaseModel):
    query: str
    occurrences: int
    matching_sentences: list[str]


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
    
class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: str
    upload_time: datetime
    extracted_text: str
    stats: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)

class UploadDocumentResponse(BaseModel):
    document: DocumentResponse
    similar_documents: List[SimilarDocumentMatch]