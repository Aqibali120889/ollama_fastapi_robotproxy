import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ConversationSession(Base):
    """One conversation -- today, one /ws/voice websocket connection.

    Named ConversationSession (not Session) to avoid clashing with
    SQLAlchemy's own Session/AsyncSession classes used throughout this
    package.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="voice")

    utterances: Mapped[List["Utterance"]] = relationship(
        back_populates="session", order_by="Utterance.timestamp", cascade="all, delete-orphan"
    )


class Utterance(Base):
    """One turn of speech, either side of the conversation.

    This table is a TimescaleDB hypertable partitioned on `timestamp` (see
    alembic/versions/0001_initial.py). The composite primary key
    (id, timestamp) is a TimescaleDB requirement, not a design choice:
    hypertables require the partitioning column to be part of any
    unique/primary key constraint.
    """

    __tablename__ = "utterances"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    speaker: Mapped[str] = mapped_column(String(10), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    __table_args__ = (CheckConstraint("speaker IN ('human', 'robot')", name="ck_utterances_speaker"),)

    session: Mapped["ConversationSession"] = relationship(back_populates="utterances")
