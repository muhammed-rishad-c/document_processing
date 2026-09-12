import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, Text, Integer, ForeignKey, Boolean, Index
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
    __tablename__ = "document_chunk"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")
    
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=True, default="New Conversation")
    document_id = Column(UUID(as_uuid=True), nullable=True)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    awaiting_lead_capture = Column(Boolean, nullable=False, default=False, server_default="false")
    pending_lead_query = Column(Text, nullable=True)
    lead_capture_attempts = Column(Integer, nullable=False, default=0, server_default="0")

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    company = relationship("Company", back_populates="chat_sessions")\
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")
    
class Company(Base):
    __tablename__ = "companies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    tenant_id = Column(String, nullable=False, unique=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    allowed_origins = Column(JSONB, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chat_sessions = relationship("ChatSession", back_populates="company", cascade="all, delete-orphan")
    leads = relationship("Lead", back_populates="company", cascade="all, delete-orphan")
    departments = relationship("CompanyDepartment", back_populates="company", cascade="all, delete-orphan")


class CompanyDepartment(Base):

    __tablename__ = "company_departments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    is_default = Column(Boolean, nullable=False, default=False, server_default="false")
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    company = relationship("Company", back_populates="departments")
    leads = relationship("Lead", back_populates="department")

    __table_args__ = (
        Index(
            "uq_company_departments_one_default",
            "company_id",
            unique=True,
            postgresql_where=(is_default == True),  
        ),
        Index(
            "uq_company_departments_name_per_company",
            "company_id",
            "name",
            unique=True,
        ),
    )


class Lead(Base):
    __tablename__ = "leads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True)
    question = Column(Text, nullable=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # --- Added for query classification + department email routing ---
    department_id = Column(
        UUID(as_uuid=True),
        ForeignKey("company_departments.id", ondelete="SET NULL"),
        nullable=True,
    )
    category_name = Column(Text, nullable=True)  # snapshot of department name at classification time
    email_sent = Column(Boolean, nullable=False, default=False, server_default="false")

    company = relationship("Company", back_populates="leads")
    department = relationship("CompanyDepartment", back_populates="leads")