import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, Text,Integer,ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from .database import Base

class Document(Base):
    __tablename__ = "documents"  

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    
    file_type = Column("filetype", String, nullable=False)  
    upload_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    extracted_text = Column(Text, nullable=False)
    stats = Column(JSONB, nullable=False)
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
class DocumentChunk(Base):
    __tablename__="document_chunk"
    
    id=Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)
    document_id=Column(UUID(as_uuid=True),ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")
    