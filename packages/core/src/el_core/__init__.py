"""EL Core - LLM-native interview agent with knowledge graph integration."""

from el_core.agent import ELAgent
from el_core.schemas import (
    ConversationTurn,
    DateType,
    DocumentChunk,
    Insight,
    Message,
    Session,
    Tag,
    TaggedItem,
    TaggedItemType,
    TagMergeRequest,
    TagMergeResult,
    TagStats,
    TagSuggestion,
    TagSuggestionResult,
)

__version__ = "2.0.0"
__all__ = [
    "ELAgent",
    "ConversationTurn",
    "DateType",
    "DocumentChunk",
    "Insight",
    "Message",
    "Session",
    "Tag",
    "TaggedItem",
    "TaggedItemType",
    "TagMergeRequest",
    "TagMergeResult",
    "TagStats",
    "TagSuggestion",
    "TagSuggestionResult",
]
