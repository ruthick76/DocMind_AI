from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentInfo(StrictModel):
    id: UUID
    filename: str = Field(min_length=1, max_length=180)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    created_at: datetime


class UploadResult(StrictModel):
    document: DocumentInfo
    duplicate: bool


class Chunk(StrictModel):
    id: UUID
    document_id: UUID
    text: str = Field(min_length=1, max_length=5000)
    page: int | None = Field(default=None, ge=1)


class Source(StrictModel):
    citation_id: int = Field(ge=1)
    document_id: UUID
    filename: str
    page: int | None = None
    excerpt: str
    similarity: float


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime


class Conversation(StrictModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = Field(default_factory=list)


class ChatSummary(StrictModel):
    id: UUID
    title: str
    updated_at: datetime
    message_count: int


class ChatRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    chat_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        if not value.strip():
            raise ValueError("Question must contain text")
        return value.strip()


class ChatResponse(StrictModel):
    chat_id: UUID
    answer: str
    sources: list[Source]
    insufficient_context: bool


class GroundedAnswer(StrictModel):
    answer: str = Field(min_length=1, max_length=12000)
    sufficient_context: bool
    citation_ids: list[int] = Field(description="IDs of supporting retrieved excerpts")


class HealthResponse(StrictModel):
    status: Literal["ok", "configuration_required", "storage_error"]
    configured: bool
    missing: list[str]
    documents: int
    chunks: int
    llm_model: str
    embedding_model: str
    storage_ready: bool
    message: str | None = None
