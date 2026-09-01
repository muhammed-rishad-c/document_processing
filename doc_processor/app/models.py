import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from .database import Base

class Document(Base):
    __tablename__ = "documents"  

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    
    file_type = Column("filetype", String, nullable=False)  
    upload_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    extracted_text = Column(Text, nullable=False)
    stats = Column(JSONB, nullable=False)