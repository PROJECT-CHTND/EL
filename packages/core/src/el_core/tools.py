"""Tool definitions for Eager Learner (EL) function calling."""

from __future__ import annotations

import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from el_core.schemas import Domain, Insight, KnowledgeItem
from el_core.stores.kg_store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


# Tool definitions in OpenAI format

SEARCH_KNOWLEDGE_GRAPH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "search_knowledge_graph",
        "description": (
            "過去の対話や保存された知識から、関連する情報を検索します。"
            "共感的な対話や文脈理解に活用してください。"
            "例：ユーザーが言及したトピックに関連する過去の洞察を見つける"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "検索クエリ（キーワードや概念）",
                },
                "limit": {
                    "type": "integer",
                    "description": "取得する最大件数（デフォルト: 5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

SAVE_INSIGHT_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "save_insight",
        "description": (
            "対話から得られた重要な洞察や事実を知識グラフに保存します。"
            "暗黙知や重要なポイントを記録する際に使用してください。"
            "保存した後は「これは重要なポイントですね、記録しておきますね」と伝えてください。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "主語となるエンティティ（例：ユーザー名、プロジェクト名、概念）",
                },
                "predicate": {
                    "type": "string",
                    "description": (
                        "関係性のタイプ。例：\n"
                        "- has_insight: 洞察を持っている\n"
                        "- learned_that: 〜を学んだ\n"
                        "- prefers: 〜を好む\n"
                        "- struggled_with: 〜で苦労した\n"
                        "- succeeded_in: 〜で成功した\n"
                        "- believes: 〜と信じている"
                    ),
                },
                "object": {
                    "type": "string",
                    "description": "目的語（洞察の内容、学んだこと、好みの詳細など）",
                },
                "confidence": {
                    "type": "number",
                    "description": "確信度 (0.0-1.0)。明確な発言は0.9以上、推測は0.6程度",
                    "default": 0.8,
                },
                "domain": {
                    "type": "string",
                    "description": "推定されたドメイン",
                    "enum": ["daily_work", "recipe", "postmortem", "creative", "general"],
                    "default": "general",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
    },
}

# All available tools
ALL_TOOLS: list[ChatCompletionToolParam] = [
    SEARCH_KNOWLEDGE_GRAPH_TOOL,
    SAVE_INSIGHT_TOOL,
]

# Tools for when knowledge has already been pre-searched and injected into the prompt.
# Only save_insight is needed; search would be redundant and waste an LLM round-trip.
SAVE_ONLY_TOOLS: list[ChatCompletionToolParam] = [
    SAVE_INSIGHT_TOOL,
]


class ToolExecutor:
    """Executes tools and manages knowledge graph interactions."""

    def __init__(
        self,
        kg_store: KnowledgeGraphStore | None = None,
        session_id: str | None = None,
        session_domain: Domain | None = None,
    ) -> None:
        """Initialize the tool executor.

        Args:
            kg_store: Knowledge graph store instance. If None, tools will be no-ops.
            session_id: Session ID to associate with saved insights.
            session_domain: Current session domain for filtering search results.
        """
        self._kg_store = kg_store
        self._session_id = session_id
        self._session_domain = session_domain
        self._saved_insights: list[Insight] = []
        self._saved_insight_ids: list[tuple[str, str]] = []  # (insight_id, content) for auto-tagging
        self._used_knowledge: list[KnowledgeItem] = []

    @property
    def saved_insights(self) -> list[Insight]:
        """Get insights saved during this execution."""
        return self._saved_insights.copy()

    @property
    def saved_insight_ids(self) -> list[tuple[str, str]]:
        """Get (insight_id, content) tuples for insights saved during this execution."""
        return self._saved_insight_ids.copy()

    @property
    def used_knowledge(self) -> list[KnowledgeItem]:
        """Get knowledge items used during this execution."""
        return self._used_knowledge.copy()

    def reset(self) -> None:
        """Reset the saved insights and used knowledge."""
        self._saved_insights.clear()
        self._saved_insight_ids.clear()
        self._used_knowledge.clear()

    async def search_knowledge_graph(
        self,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search the knowledge graph for relevant information.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            Dict with search results or empty message.
        """
        if self._kg_store is None:
            return {
                "found": False,
                "message": "知識グラフが利用できません。",
                "items": [],
            }

        try:
            # Apply domain filter if session has a non-general domain
            search_domain = self._session_domain if self._session_domain and self._session_domain != Domain.GENERAL else None
            items = await self._kg_store.search(query, limit=limit, domain=search_domain)
            self._used_knowledge.extend(items)

            if not items:
                return {
                    "found": False,
                    "message": f"「{query}」に関連する情報は見つかりませんでした。",
                    "items": [],
                }

            return {
                "found": True,
                "message": f"{len(items)}件の関連情報が見つかりました。",
                "items": [
                    {
                        "subject": item.subject,
                        "predicate": item.predicate,
                        "object": item.object,
                        "confidence": item.confidence,
                        "domain": item.domain.value,
                    }
                    for item in items
                ],
            }
        except Exception as e:
            logger.error(f"Knowledge graph search failed: {e}")
            return {
                "found": False,
                "message": f"検索中にエラーが発生しました: {e}",
                "items": [],
            }

    async def save_insight(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        domain: str = "general",
    ) -> dict[str, Any]:
        """Save an insight to the knowledge graph.

        Args:
            subject: Subject entity.
            predicate: Relationship type.
            object: Object/content.
            confidence: Confidence score.
            domain: Domain classification.

        Returns:
            Dict with save result.
        """
        try:
            domain_enum = Domain(domain)
        except ValueError:
            domain_enum = Domain.GENERAL

        insight = Insight(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            domain=domain_enum,
        )

        self._saved_insights.append(insight)

        if self._kg_store is None:
            return {
                "saved": True,
                "message": "洞察を記録しました（ローカルのみ）。",
                "insight": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                },
            }

        try:
            insight_id = await self._kg_store.save_insight(insight, session_id=self._session_id)
            # Track for auto-tagging
            content_for_tagging = f"{subject} {predicate} {object}"
            self._saved_insight_ids.append((insight_id, content_for_tagging))
            return {
                "saved": True,
                "message": "洞察を知識グラフに保存しました。",
                "insight_id": insight_id,
                "insight": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                },
            }
        except Exception as e:
            logger.error(f"Failed to save insight: {e}")
            return {
                "saved": False,
                "message": f"保存中にエラーが発生しました: {e}",
                "insight": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                },
            }

    def get_tool_handlers(self) -> dict[str, Any]:
        """Get a dict mapping tool names to handler methods.

        Returns:
            Dict of tool name to async handler function.
        """
        return {
            "search_knowledge_graph": self.search_knowledge_graph,
            "save_insight": self.save_insight,
        }
