from __future__ import annotations

from pydantic import BaseModel, Field


class ContextPayload(BaseModel):
    """Three-level hierarchical context summaries."""

    rag_keys: str = Field(..., description="Fine-grained summary suitable for RAG key chunks")
    mid_summary: str = Field(..., description="Mid-level summary")
    global_summary: str = Field(..., description="High-level summary of entire text") 