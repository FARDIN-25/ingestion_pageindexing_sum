"""Make vectors table standalone (no FK to document_jobs / document_nodes).

Revision ID: 005_vectors_standalone
Revises: 004_vectors_nullable_embed_meta
Create Date: 2026-07-24
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "005_vectors_standalone"
down_revision: Union[str, None] = "004_vectors_nullable_embed_meta"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop FKs so Vector path does not depend on PageIndex tables.
    op.execute("ALTER TABLE vectors DROP CONSTRAINT IF EXISTS fk_vectors_doc_node")
    op.execute("ALTER TABLE vectors DROP CONSTRAINT IF EXISTS vectors_job_id_fkey")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vectors_job_id ON vectors (job_id)")


def downgrade() -> None:
    # Best-effort restore — may fail if orphan vector rows exist.
    op.execute(
        """
        ALTER TABLE vectors
        ADD CONSTRAINT vectors_job_id_fkey
        FOREIGN KEY (job_id) REFERENCES document_jobs(seq_id) ON DELETE CASCADE
        """
    )
    op.execute(
        """
        ALTER TABLE vectors
        ADD CONSTRAINT fk_vectors_doc_node
        FOREIGN KEY (doc_id, node_id)
        REFERENCES document_nodes (doc_id, node_id) ON DELETE CASCADE
        """
    )
