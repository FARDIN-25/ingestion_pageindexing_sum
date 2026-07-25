import uuid

from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from pageindex.env_settings import settings

from .database import Base


class DocumentJob(Base):
    __tablename__ = "document_jobs"

    seq_id = Column(Integer, unique=True, index=True, nullable=True)
    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=True)
    status = Column(String, default="pending", nullable=False)
    error_message = Column(Text, nullable=True)
    results = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DocumentNode(Base):
    __tablename__ = "document_nodes"

    seq_id = Column(Integer, index=True, nullable=True)
    doc_id = Column(String, ForeignKey("document_jobs.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = Column(String, nullable=True)
    id = Column(String, primary_key=True)
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
    metadata_json = Column(JSONB, nullable=True)
    node_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("doc_id", "node_id", name="uq_document_nodes_doc_node"),
        Index("ix_document_nodes_doc_node", "doc_id", "node_id"),
        Index("ix_document_nodes_seq_node", "seq_id", "node_id"),
    )


class Vector(Base):
    """Standalone Vector service rows — no FK to document_jobs / document_nodes.

    PageIndex owns document_jobs + document_nodes.
    Vector owns only this table (job_id / node_id are free identifiers).
    """

    __tablename__ = "vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String, nullable=False, index=True)
    # Free integer correlator for a vector ingest run (NOT document_jobs.seq_id).
    job_id = Column(Integer, nullable=False)
    # Chunk / role key within this doc (e.g. v0000, nr_notice) — not a document_nodes FK.
    node_id = Column(String, nullable=False)
    chunk_index = Column(SmallInteger, nullable=False, default=0)
    chunk_text = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding = Column(PgVector(settings.EMBEDDING_DIM), nullable=True)
    # Nullable so notice_reply/faq "reply"/"answer" rows can store embedding=NULL
    # without a placeholder model (status=not_embedded).
    embedding_dim = Column(SmallInteger, nullable=True, default=settings.EMBEDDING_DIM)
    model_name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    # Do NOT name the attribute `metadata` — clashes with SQLAlchemy MetaData.
    chunk_meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint(
            "doc_id",
            "node_id",
            "chunk_index",
            "model_name",
            name="uq_vectors_doc_node_chunk_model",
        ),
        Index("idx_vectors_doc_id", "doc_id"),
        Index("idx_vectors_content_hash", "content_hash"),
        Index("idx_vectors_job_id", "job_id"),
    )
