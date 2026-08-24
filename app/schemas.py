import uuid
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    mode: Literal["chat", "code", "vision"] = "chat"
    model: Optional[str] = None
    session_id: Optional[str] = Field(default=None, min_length=1)
    metadata_filters: Optional[Dict[str, str]] = None
    options: Optional[Dict[str, object]] = None

    @model_validator(mode="after")
    def validate_input(self) -> "ChatRequest":
        if not self.prompt and not self.messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        return self


class ChatResponse(BaseModel):
    model: str
    output: str
    source: str
    session_id: str
    used_rag: bool = False
    timings: Dict[str, float] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)


class HealthCheckResult(BaseModel):
    url: str
    reachable: bool
    latency_ms: Optional[int] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    models: dict
    cuda: dict
    ollama: dict


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    session_id: Optional[str] = Field(default=None, min_length=1)
    metadata_filters: Optional[Dict[str, str]] = None


class RagResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str]
    retrieved_metadata: List[Dict[str, str]]
    retrieved_chunks: List[str]
    distances: List[float]
    used_rag: bool
    search_query: str
    timings: dict


# ---- Persistent session/utterance storage (PostgreSQL + TimescaleDB) ----


class CreateSessionRequest(BaseModel):
    source: str = Field(default="voice", min_length=1, max_length=20)


class SessionResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    closed_at: Optional[datetime] = None
    source: str

    model_config = {"from_attributes": True}


class AppendUtteranceRequest(BaseModel):
    speaker: Literal["human", "robot"]
    text: str = Field(min_length=1)
    extra: Optional[Dict[str, object]] = None


class UtteranceResponse(BaseModel):
    id: int
    session_id: uuid.UUID
    speaker: str
    timestamp: datetime
    text: str
    extra: Optional[Dict[str, object]] = None

    model_config = {"from_attributes": True}


class SessionReplayResponse(BaseModel):
    session: SessionResponse
    utterances: List[UtteranceResponse]
