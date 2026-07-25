"""Allow NULL embedding_dim/model_name for non-embedded vector rows.

Revision ID: 004_vectors_nullable_embed_meta
Revises: 003_vectors_document_only
Create Date: 2026-07-24
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "004_vectors_nullable_embed_meta"
down_revision: Union[str, None] = "003_vectors_document_only"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE vectors ALTER COLUMN embedding_dim DROP NOT NULL")
    op.execute("ALTER TABLE vectors ALTER COLUMN model_name DROP NOT NULL")


def downgrade() -> None:
    op.execute(
        """
        UPDATE vectors
        SET embedding_dim = COALESCE(embedding_dim, 0)
        WHERE embedding_dim IS NULL
        """
    )
    op.execute(
        """
        UPDATE vectors
        SET model_name = COALESCE(model_name, '')
        WHERE model_name IS NULL
        """
    )
    op.execute("ALTER TABLE vectors ALTER COLUMN embedding_dim SET NOT NULL")
    op.execute("ALTER TABLE vectors ALTER COLUMN model_name SET NOT NULL")
