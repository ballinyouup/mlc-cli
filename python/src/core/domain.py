"""
Domain models and Data Transfer Objects (DTOs).

This module contains all Pydantic models that define the data contracts
between layers. These are analogous to Java Records or Go structs.

IMPORTANT: Never pass raw dicts between layers - always use these DTOs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Role in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: ChatRole
    content: str


class Document(BaseModel):
    """A document stored in the vector database."""

    id: str
    content: str
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None

    class Config:
        frozen = True


class Chunk(BaseModel):
    """A chunk of a document with its embedding."""

    id: str
    document_id: str
    content: str
    embedding: list[float]
    position: int
    created_at: Optional[datetime] = None

    class Config:
        frozen = True


class SearchResult(BaseModel):
    """Result from a vector similarity search."""

    chunk_id: str
    document_id: str
    content: str
    position: int
    similarity: float
    metadata: Optional[dict] = None

    class Config:
        frozen = True


class LaMPTask(BaseModel):
    """A LaMP benchmark task definition."""

    id: str
    task_number: int
    name: str
    description: Optional[str] = None
    sample_count: int = 0

    class Config:
        frozen = True


class LaMPSample(BaseModel):
    """A sample from a LaMP task."""

    id: str
    sample_id: str
    input: str
    output: Optional[str] = None
    split: str
    variant: str
    task: LaMPTask
    profile_count: int = 0

    class Config:
        frozen = True


class ProfileDocument(BaseModel):
    """A profile document for LaMP personalization."""

    profile_id: str
    title: Optional[str] = None
    content: str
    similarity: float = 0.0
    sample_id: Optional[str] = None
    task_name: Optional[str] = None

    class Config:
        frozen = True


class LaMPContext(BaseModel):
    """Context built for LaMP personalization inference."""

    sample: LaMPSample
    input: str
    profile_context: str
    expected_output: Optional[str] = None
    retrieved_profiles: int = 0

    class Config:
        frozen = True


class LaMPContextError(BaseModel):
    """Error response when building LaMP context fails."""

    error: str

    class Config:
        frozen = True


class DocumentCreateRequest(BaseModel):
    """Request to create a new document."""

    content: str
    metadata: Optional[dict] = None
    chunk_size: int = Field(default=500, ge=100, le=2000)


class DocumentSummary(BaseModel):
    """Summary view of a document for listing."""

    id: str
    content_preview: str
    metadata: Optional[dict] = None
    chunk_count: int
    created_at: Optional[datetime] = None

    class Config:
        frozen = True


class LaMPSampleSummary(BaseModel):
    """Summary view of a LaMP sample for listing."""

    id: str
    sample_id: str
    input_preview: str
    output: Optional[str] = None
    split: str
    variant: str
    task_name: str
    task_number: int
    profile_count: int = 0

    class Config:
        frozen = True


class LaMPTaskSummary(BaseModel):
    """Summary view of a LaMP task for listing."""

    id: str
    number: int
    name: str
    description: Optional[str] = None
    sample_count: int = 0

    class Config:
        frozen = True
