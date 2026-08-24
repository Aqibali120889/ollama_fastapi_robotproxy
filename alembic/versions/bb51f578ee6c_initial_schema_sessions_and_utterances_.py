"""initial schema: sessions and utterances hypertable

Revision ID: bb51f578ee6c
Revises:
Create Date: 2026-08-24 10:23:00.839385

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bb51f578ee6c"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="voice"),
    )

    op.create_table(
        "utterances",
        # Explicit Identity() rather than relying on implicit autoincrement
        # detection: SQLAlchemy only auto-applies that to a *sole* integer PK
        # column, and this one is part of a composite PK (see below).
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("speaker", sa.String(length=10), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("extra", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id", "timestamp"),
        sa.CheckConstraint("speaker IN ('human', 'robot')", name="ck_utterances_speaker"),
    )
    op.create_index("ix_utterances_session_id", "utterances", ["session_id"])

    # Must run after the table exists, and the partitioning column ("timestamp")
    # must already be part of the primary key (done above) -- a TimescaleDB
    # requirement for hypertables, not just a convention.
    op.execute("SELECT create_hypertable('utterances', 'timestamp', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_table("utterances")
    op.drop_table("sessions")
