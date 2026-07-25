"""Recreate vectors table for dual PageIndex + Vector services.

Revision ID: 002_vectors_v2
Revises: 001_add_vectors
Create Date: 2026-07-23

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from pageindex.env_settings import settings

revision: str = "002_vectors_v2"
down_revision: Union[str, None] = "001_add_vectors"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    dim = int(settings.EMBEDDING_DIM)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS page_index_jobs (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR NOT NULL UNIQUE,
            file_name VARCHAR,
            status VARCHAR NOT NULL DEFAULT 'pending',
            error_message TEXT,
            source VARCHAR NOT NULL DEFAULT 'upload',
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'uq_page_index_nodes_doc_node'
            ) THEN
                ALTER TABLE page_index_nodes
                ADD CONSTRAINT uq_page_index_nodes_doc_node UNIQUE (doc_id, node_id);
            END IF;
        END $$;
        """
    )
    op.execute("DROP TABLE IF EXISTS vectors CASCADE")
    op.execute(
        f"""
        CREATE TABLE vectors (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id VARCHAR NOT NULL,
            job_id INTEGER NOT NULL REFERENCES page_index_jobs(id) ON DELETE CASCADE,
            node_id VARCHAR NOT NULL,
            chunk_index SMALLINT NOT NULL DEFAULT 0,
            chunk_text TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            token_count INTEGER,
            embedding vector({dim}),
            embedding_dim SMALLINT NOT NULL,
            model_name VARCHAR NOT NULL,
            status VARCHAR NOT NULL DEFAULT 'pending',
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_vectors_doc_node_chunk_model
                UNIQUE (doc_id, node_id, chunk_index, model_name),
            FOREIGN KEY (doc_id, node_id)
                REFERENCES page_index_nodes (doc_id, node_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX idx_vectors_doc_id ON vectors (doc_id)")
    op.execute(
        "CREATE INDEX idx_vectors_status ON vectors (status) WHERE status != 'completed'"
    )
    op.execute("CREATE INDEX idx_vectors_content_hash ON vectors (content_hash)")
    op.execute(
        """
        CREATE INDEX idx_vectors_embedding
        ON vectors USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS vectors CASCADE")
