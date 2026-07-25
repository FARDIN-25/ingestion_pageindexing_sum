"""Add vectors table with pgvector + IVFFlat cosine index.

Revision ID: 001_add_vectors
Revises:
Create Date: 2026-07-23

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from pageindex.env_settings import settings

# revision identifiers, used by Alembic.
revision: str = "001_add_vectors"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    dim = int(settings.EMBEDDING_DIM)

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "vectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("node_id", sa.String(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dim), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["document_jobs.doc_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "doc_id",
            "node_id",
            "model_name",
            name="uq_vectors_doc_node_model",
        ),
    )
    op.create_index("ix_vectors_doc_id", "vectors", ["doc_id"])
    op.create_index("ix_vectors_job_id", "vectors", ["job_id"])

    # Cosine IVFFlat index for ANN search.
    op.execute(
        """
        CREATE INDEX ix_vectors_embedding_ivfflat
        ON vectors USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_vectors_embedding_ivfflat")
    op.drop_index("ix_vectors_job_id", table_name="vectors")
    op.drop_index("ix_vectors_doc_id", table_name="vectors")
    op.drop_table("vectors")
