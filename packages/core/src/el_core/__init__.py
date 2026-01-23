"""EL Core - LLM-native interview agent with knowledge graph integration."""

from el_core.agent import ELAgent
from el_core.schemas import (
    ConversationTurn,
    Insight,
    Message,
    Session,
)

__version__ = "2.0.0"
__all__ = [
    "ELAgent",
    "ConversationTurn",
    "Insight",
    "Message",
    "Session",
]
