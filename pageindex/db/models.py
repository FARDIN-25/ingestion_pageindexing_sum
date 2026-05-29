from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from .database import Base

class DocumentJob(Base):
    __tablename__ = "document_jobs"

    seq_id = Column(Integer, unique=True, index=True, nullable=True)
    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=True)
    status = Column(String, default="pending", nullable=False)  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    results = Column(JSONB, nullable=True)  # full json response from pageindex
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class DocumentNode(Base):
    __tablename__ = "document_nodes"

    seq_id = Column(Integer, index=True, nullable=True)
    doc_id = Column(String, ForeignKey("document_jobs.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = Column(String, nullable=True)
    id = Column(String, primary_key=True)  # uuid
    node_id = Column(String, nullable=False)
    parent_id = Column(String, nullable=True)
    type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    path = Column(Text, nullable=True)
    level = Column(Integer, nullable=False)
    
    raw_content = Column(Text, nullable=True)
    compressed_content = Column(Text, nullable=True)
    micro_summary = Column(Text, nullable=True)
    content_hash = Column(String, nullable=True)
    
    retrieval_ready = Column(Boolean, default=False)
    is_front_matter = Column(Boolean, default=False)
    metadata_json = Column(JSONB, nullable=True)  # page ranges, synonyms, etc.
    node_json = Column(JSONB, nullable=True)  # raw json for this specific node
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_document_nodes_doc_node", "doc_id", "node_id"),
        Index("ix_document_nodes_seq_node", "seq_id", "node_id"),
    )
