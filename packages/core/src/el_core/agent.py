"""EL Agent - LLM-native interview agent with knowledge graph integration."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from el_core.llm.client import LLMClient
from el_core.prompts import get_system_prompt
from el_core.schemas import (
    AgentResponse,
    ConsistencyIssue,
    ConsistencyIssueKind,
    ConversationTurn,
    Domain,
    Insight,
    KnowledgeItem,
    Session,
    SessionSummary,
)
from el_core.stores.kg_store import KnowledgeGraphStore
from el_core.tools import ALL_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect if text is Japanese or English.

    Args:
        text: Text to analyze.

    Returns:
        "Japanese" or "English".
    """
    # Check for Japanese characters
    for char in text:
        if (
            "\u3040" <= char <= "\u309f"  # Hiragana
            or "\u30a0" <= char <= "\u30ff"  # Katakana
            or "\u4e00" <= char <= "\u9faf"  # Kanji
        ):
            return "Japanese"
    return "English"


class ELAgent:
    """EL Agent - A curious and empathetic interviewer powered by LLM.

    This agent uses GPT-5.2 with function calling to:
    - Search knowledge graph for relevant context
    - Save important insights from conversations
    - Dynamically detect conversation domains
    - Generate empathetic, insightful questions
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        kg_store: KnowledgeGraphStore | None = None,
    ) -> None:
        """Initialize the EL Agent.

        Args:
            llm_client: LLM client for chat completions. Creates default if None.
            kg_store: Knowledge graph store. Tools will work in local-only mode if None.
        """
        self._llm = llm_client or LLMClient()
        self._kg_store = kg_store
        self._sessions: dict[str, Session] = {}

    async def start_session(
        self,
        user_id: str,
        topic: str,
        language: str | None = None,
    ) -> tuple[str, str]:
        """Start a new conversation session.

        Args:
            user_id: User identifier.
            topic: Conversation topic.
            language: Language preference. Auto-detected if None.

        Returns:
            Tuple of (session_id, opening_message).
        """
        detected_lang = language or detect_language(topic)
        session_id = str(uuid.uuid4())

        session = Session(
            id=session_id,
            user_id=user_id,
            topic=topic,
            language=detected_lang,
        )
        
        # Pre-fetch related knowledge from KG
        if self._kg_store:
            try:
                related_items = await self._kg_store.search(topic, limit=5)
                if related_items:
                    session.prior_knowledge = related_items
                    session.prior_context = self._format_prior_knowledge(related_items, detected_lang)
                    logger.info(f"Found {len(related_items)} related knowledge items for topic: {topic}")
            except Exception as e:
                logger.warning(f"Failed to pre-fetch knowledge: {e}")
        
        self._sessions[session_id] = session

        # Save session metadata to Neo4j
        if self._kg_store:
            try:
                await self._save_session_metadata(session)
            except Exception as e:
                logger.warning(f"Failed to save session metadata: {e}")

        # Generate natural opening response using LLM
        opening = await self._generate_opening_response(session, topic, detected_lang)
        logger.info(f"Started session {session_id} for user {user_id} on topic: {topic}")

        return session_id, opening
    
    async def _save_session_metadata(self, session: Session) -> None:
        """Save session metadata to Neo4j.
        
        Args:
            session: Session to save metadata for.
        """
        if self._kg_store is None:
            return
        
        await self._kg_store.save_session_metadata(
            session_id=session.id,
            user_id=session.user_id,
            topic=session.topic,
            domain=session.domain,
            turn_count=len(session.turns),
            insights_count=session.insights_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
    
    async def _generate_opening_response(
        self,
        session: Session,
        user_input: str,
        language: str,
    ) -> str:
        """Generate a natural opening response using LLM.
        
        Instead of using a template, we let the LLM respond naturally to
        whatever the user typed - whether it's a topic, a request, or a question.
        
        Args:
            session: The session object.
            user_input: What the user typed to start the conversation.
            language: Detected language.
            
        Returns:
            Natural opening response from the LLM.
        """
        # Use a simplified prompt for opening - no tool instructions
        if language.lower() in ("english", "en"):
            system_content = (
                "You are EL, a curious and empathetic interviewer. "
                "Your role is to deeply understand what people share and help them discover new insights.\n\n"
                "Guidelines:\n"
                "- Show genuine interest and empathy\n"
                "- Keep responses conversational and warm\n"
                "- Ask 1-2 open-ended questions to start the dialogue\n"
                "- Be concise but engaging\n"
                "- Do NOT output any JSON, tool calls, or code\n"
            )
        else:
            system_content = (
                "あなたは「EL」です。好奇心旺盛で共感性の高いインタビュワーとして、"
                "相手の話を深く理解し、新たな気づきを引き出す対話を行います。\n\n"
                "ガイドライン：\n"
                "- 共感と関心を持って応答する\n"
                "- 自然で温かい会話調を心がける\n"
                "- 会話を始めるために1〜2個の質問をする\n"
                "- 簡潔だけど魅力的に\n"
                "- JSONやツールコール、コードは絶対に出力しない\n"
            )
        
        # Add prior context if available
        if session.prior_context:
            system_content += session.prior_context
        
        # Add instruction for first response
        if language.lower() in ("english", "en"):
            system_content += (
                "\n\n### First Response Instructions\n"
                "This is the start of a new conversation. The user has just shared "
                "what they want to discuss. Respond naturally and warmly - acknowledge "
                "their input, show interest, and ask a relevant follow-up question to "
                "get the conversation started. Keep it conversational and friendly. "
                "Output ONLY natural conversational text."
            )
        else:
            system_content += (
                "\n\n### 最初の応答に関する指示\n"
                "これは新しい会話の開始です。ユーザーが話したいことを伝えてきました。"
                "自然で温かく応答してください。ユーザーの入力を受け止め、関心を示し、"
                "会話を始めるための適切なフォローアップの質問をしてください。"
                "会話調でフレンドリーに、かつ簡潔に応答してください。"
                "自然な会話テキストのみを出力してください。"
            )
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]
        
        try:
            # Simple chat completion without tools for the opening
            response = await self._llm.chat(messages=messages)  # type: ignore
            return response.content or ""
        except Exception as e:
            logger.warning(f"Failed to generate opening response via LLM: {e}")
            # Fallback to a simple acknowledgment
            if language.lower() in ("english", "en"):
                return "I'd be happy to discuss that with you. What would you like to start with?"
            return "承知しました。どのようなことからお話しましょうか？"
    
    def _format_prior_knowledge(self, items: list[KnowledgeItem], language: str) -> str:
        """Format knowledge items into context string.
        
        Args:
            items: List of knowledge items.
            language: Language for formatting.
            
        Returns:
            Formatted context string.
        """
        if not items:
            return ""
        
        if language.lower() in ("english", "en"):
            header = "\n\n### Related Past Knowledge\n\nThe following are relevant insights from previous conversations:\n\n"
            item_format = "- {subject} {predicate}: {object} (recorded on {date})\n"
        else:
            header = "\n\n### 関連する過去の知識\n\n以下は過去の対話で得られた関連情報です。これらを踏まえて会話してください：\n\n"
            item_format = "- {subject}は{predicate}：{object}（{date}に記録）\n"
        
        context = header
        for item in items:
            date_str = item.created_at.strftime("%Y年%m月%d日") if language.lower() not in ("english", "en") else item.created_at.strftime("%Y-%m-%d")
            context += item_format.format(
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                date=date_str,
            )
        
        return context

    async def resume_session(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[str, str, list[KnowledgeItem]]:
        """Resume an existing session from Neo4j.

        Args:
            session_id: Session ID to resume.
            user_id: User ID (for verification).

        Returns:
            Tuple of (session_id, resume_message, prior_insights).

        Raises:
            ValueError: If session not found or user mismatch.
        """
        if self._kg_store is None:
            raise ValueError("Knowledge graph store not available")

        # Get session metadata from Neo4j
        metadata = await self._kg_store.get_session_metadata(session_id)
        if metadata is None:
            raise ValueError(f"Session {session_id} not found")
        
        if metadata.user_id != user_id:
            raise ValueError("User ID mismatch")

        # Get insights saved during this session
        session_insights = await self._kg_store.get_session_insights(session_id)

        # Detect language from topic
        detected_lang = detect_language(metadata.topic)

        # Create new session object with prior context
        session = Session(
            id=metadata.id,
            user_id=metadata.user_id,
            topic=metadata.topic,
            language=detected_lang,
            domain=metadata.domain,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
        )

        # Set prior context from session insights
        if session_insights:
            session.prior_knowledge = session_insights
            session.prior_context = self._format_prior_knowledge(session_insights, detected_lang)

        # Store in active sessions
        self._sessions[session_id] = session

        # Generate resume message
        if detected_lang.lower() in ("english", "en"):
            resume_msg = (
                f"Welcome back! We were discussing \"{metadata.topic}\". "
                f"Last time we had {metadata.turn_count} exchanges and saved {metadata.insights_count} insights. "
                f"Would you like to continue from where we left off?"
            )
        else:
            resume_msg = (
                f"「{metadata.topic}」の続きですね。お帰りなさい！\n"
                f"前回は{metadata.turn_count}ターンの会話で、{metadata.insights_count}件の洞察を記録しました。\n"
                f"どこから続けましょうか？"
            )

        logger.info(f"Resumed session {session_id} for user {user_id}")

        return session_id, resume_msg, session_insights

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session or None if not found.
        """
        return self._sessions.get(session_id)

    async def end_session(self, session_id: str, generate_summary: bool = True) -> tuple[Session | None, SessionSummary | None]:
        """End a session and optionally generate a summary.

        Args:
            session_id: Session identifier.
            generate_summary: Whether to generate and save a summary.

        Returns:
            Tuple of (ended session, generated summary or None).
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None, None

        summary = None
        if generate_summary and len(session.turns) > 0:
            # Generate summary
            summary = await self._generate_session_summary(session)
            
            # Save summary to KG
            if self._kg_store and summary:
                try:
                    await self._kg_store.save_session_summary(summary)
                    logger.info(f"Saved summary for session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to save session summary: {e}")

        # Update session status in KG
        if self._kg_store:
            try:
                await self._kg_store.update_session_status(session_id, "ended")
            except Exception as e:
                logger.warning(f"Failed to update session status: {e}")

        logger.info(f"Ended session {session_id}")
        return session, summary

    async def _generate_session_summary(self, session: Session) -> SessionSummary | None:
        """Generate a summary of the session using LLM.

        Args:
            session: The session to summarize.

        Returns:
            SessionSummary or None if generation fails.
        """
        if not session.turns:
            return None

        # Build conversation text
        conversation_text = ""
        for i, turn in enumerate(session.turns, 1):
            conversation_text += f"ターン{i}:\n"
            conversation_text += f"ユーザー: {turn.user_message}\n"
            conversation_text += f"アシスタント: {turn.assistant_response}\n\n"

        # Collect all insights from the session
        all_insights = []
        for turn in session.turns:
            for insight in turn.insights_saved:
                all_insights.append(f"{insight.subject} - {insight.predicate} - {insight.object}")

        insights_text = "\n".join(all_insights) if all_insights else "なし"

        if session.language.lower() in ("english", "en"):
            system_prompt = """You are a conversation summarizer. Summarize the following conversation.

Output ONLY valid JSON with these fields:
- content: 2-3 sentence natural summary
- key_points: Array of important facts (max 5)
- topics: Array of main topics discussed
- entities_mentioned: Array of people, projects, places mentioned

Example:
{"content":"Discussed project A progress and confirmed the deadline.","key_points":["Deadline is Feb 28","Tanaka is responsible"],"topics":["Project A","Progress report"],"entities_mentioned":["Tanaka","Project A"]}"""
        else:
            system_prompt = """あなたは会話要約者です。以下の会話を要約してください。

以下のフィールドを持つ有効なJSONのみを出力してください：
- content: 2-3文の自然な要約
- key_points: 重要な事実の配列（最大5つ）
- topics: 話し合われた主なトピックの配列
- entities_mentioned: 言及された人名、プロジェクト名、場所などの配列

例:
{"content":"プロジェクトAの進捗について話し合い、締め切りを確認した。","key_points":["締め切りは2月28日","田中さんが担当"],"topics":["プロジェクトA","進捗報告"],"entities_mentioned":["田中さん","プロジェクトA"]}"""

        user_prompt = f"""トピック: {session.topic}
ドメイン: {session.domain.value}

【会話】
{conversation_text}

【抽出された洞察】
{insights_text}

上記の会話をJSON形式で要約してください。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import json
            response = await self._llm.chat(messages=messages)  # type: ignore
            content = response.content or "{}"
            
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()
            
            data = json.loads(content)
            
            summary = SessionSummary(
                id=str(uuid.uuid4()),
                session_id=session.id,
                content=data.get("content", "会話の要約"),
                key_points=data.get("key_points", []),
                topics=data.get("topics", [session.topic]),
                entities_mentioned=data.get("entities_mentioned", []),
                turn_range=(1, len(session.turns)),
            )
            
            logger.info(f"Generated summary for session {session.id}")
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate session summary: {e}")
            # Return a basic summary on failure
            return SessionSummary(
                id=str(uuid.uuid4()),
                session_id=session.id,
                content=f"{session.topic}についての会話（{len(session.turns)}ターン）",
                key_points=[],
                topics=[session.topic],
                entities_mentioned=[],
                turn_range=(1, len(session.turns)),
            )

    async def respond(
        self,
        session_id: str,
        user_message: str,
    ) -> AgentResponse:
        """Generate a response to a user message.

        Args:
            session_id: Session identifier.
            user_message: The user's message.

        Returns:
            AgentResponse with the assistant's message and metadata.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # Pre-search knowledge graph for potential consistency issues
        pre_search_knowledge: list[KnowledgeItem] = []
        consistency_issues: list[ConsistencyIssue] = []
        consistency_context = ""
        
        if self._kg_store:
            try:
                # Search for related knowledge before LLM response
                pre_search_knowledge = await self._kg_store.search(user_message, limit=5)
                
                if pre_search_knowledge:
                    # Check for consistency issues BEFORE generating response
                    consistency_issues = await self._detect_consistency_issues(
                        user_message=user_message,
                        knowledge_used=pre_search_knowledge,
                        language=session.language,
                    )
                    
                    # If issues found, add context for LLM to address them
                    if consistency_issues:
                        consistency_context = self._format_consistency_context(
                            consistency_issues, session.language
                        )
                        logger.info(f"Found {len(consistency_issues)} consistency issues to address")
                        
            except Exception as e:
                logger.warning(f"Pre-search for consistency failed: {e}")

        # Build message history with prior context
        system_content = get_system_prompt(session.language)
        if session.prior_context:
            system_content += session.prior_context
        
        # Add consistency context if there are issues to address
        if consistency_context:
            system_content += consistency_context
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        # Add conversation history
        messages.extend(session.message_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Set up tool executor with session_id for insight tracking
        tool_executor = ToolExecutor(self._kg_store, session_id=session_id)
        
        # Add pre-searched knowledge to tool executor so it's included in response
        tool_executor.used_knowledge.extend(pre_search_knowledge)

        # Call LLM with tools
        response_text, tool_results = await self._llm.chat_with_tools(
            messages=messages,  # type: ignore
            tools=ALL_TOOLS,
            tool_handlers=tool_executor.get_tool_handlers(),
        )

        # Detect domain from response and tool usage
        detected_domain = self._detect_domain(user_message, tool_results)

        # Create conversation turn
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=response_text,
            insights_saved=tool_executor.saved_insights,
            knowledge_used=tool_executor.used_knowledge,
            detected_domain=detected_domain,
        )
        session.add_turn(turn)

        # Update session metadata in Neo4j
        if self._kg_store:
            try:
                await self._save_session_metadata(session)
            except Exception as e:
                logger.warning(f"Failed to update session metadata: {e}")

        # Build tool calls list for response
        tool_calls = [
            {
                "id": tr.get("tool_call_id", ""),
                "name": tr.get("name", ""),
                "arguments": tr.get("arguments", {}),
            }
            for tr in tool_results
        ]

        logger.info(
            f"Session {session_id}: processed message, "
            f"saved {len(tool_executor.saved_insights)} insights, "
            f"used {len(tool_executor.used_knowledge)} knowledge items, "
            f"detected {len(consistency_issues)} consistency issues"
        )

        return AgentResponse(
            message=response_text,
            tool_calls=tool_calls,  # type: ignore
            insights_saved=tool_executor.saved_insights,
            knowledge_used=tool_executor.used_knowledge,
            detected_domain=detected_domain,
            consistency_issues=consistency_issues,
        )

    def _format_consistency_context(
        self,
        issues: list[ConsistencyIssue],
        language: str,
    ) -> str:
        """Format consistency issues as context for LLM.

        Args:
            issues: List of detected consistency issues.
            language: Session language.

        Returns:
            Formatted context string to append to system prompt.
        """
        if not issues:
            return ""

        if language.lower() in ("english", "en"):
            context = "\n\n### IMPORTANT: Consistency Issues Detected\n"
            context += "The user's current message appears to contradict or change previous information.\n"
            context += "**You MUST address these in your response** - ask for clarification naturally.\n\n"
            
            for issue in issues:
                issue_type = "CONTRADICTION" if issue.kind.value == "contradiction" else "CHANGE"
                context += f"- [{issue_type}] {issue.title}\n"
                context += f"  Previously: \"{issue.previous_text}\"\n"
                context += f"  Now: \"{issue.current_text}\"\n"
                context += f"  Suggested question: \"{issue.suggested_question}\"\n\n"
            
            context += "Address this naturally by asking about the change/contradiction.\n"
        else:
            context = "\n\n### 重要：整合性の問題を検出\n"
            context += "ユーザーの現在のメッセージが過去の情報と矛盾または変更があります。\n"
            context += "**必ずこの点について確認してください** - 自然な形で質問してください。\n\n"
            
            for issue in issues:
                issue_type = "矛盾" if issue.kind.value == "contradiction" else "変更"
                context += f"- 【{issue_type}】{issue.title}\n"
                context += f"  以前：「{issue.previous_text}」\n"
                context += f"  今回：「{issue.current_text}」\n"
                context += f"  確認すべき質問：「{issue.suggested_question}」\n\n"
            
            context += "この変更/矛盾について自然に確認してください。\n"

        return context

    def _detect_domain(
        self,
        user_message: str,
        tool_results: list[dict[str, Any]],
    ) -> Domain:
        """Detect the conversation domain.

        Args:
            user_message: The user's message.
            tool_results: Results from tool calls.

        Returns:
            Detected domain.
        """
        text_lower = user_message.lower()

        # Check for domain-specific keywords
        daily_work_keywords = [
            "タスク",
            "task",
            "プロジェクト",
            "project",
            "業務",
            "work",
            "ミーティング",
            "meeting",
            "進捗",
            "progress",
            "今日",
            "today",
            "明日",
            "tomorrow",
            "締め切り",
            "deadline",
            "pr",
            "コードレビュー",
            "デプロイ",
            "deploy",
        ]

        recipe_keywords = [
            "料理",
            "cook",
            "レシピ",
            "recipe",
            "材料",
            "ingredient",
            "調理",
            "焼く",
            "煮る",
            "炒める",
            "分量",
            "オーブン",
            "oven",
            "鍋",
            "フライパン",
        ]

        postmortem_keywords = [
            "障害",
            "incident",
            "インシデント",
            "ダウン",
            "down",
            "復旧",
            "recover",
            "根本原因",
            "root cause",
            "タイムライン",
            "timeline",
            "再発防止",
            "アラート",
            "alert",
            "エラー",
            "error",
        ]

        creative_keywords = [
            "アイデア",
            "idea",
            "創作",
            "creative",
            "デザイン",
            "design",
            "イラスト",
            "illustration",
            "物語",
            "story",
            "音楽",
            "music",
            "インスピレーション",
            "inspiration",
        ]

        # Check save_insight tool calls for domain hints
        for result in tool_results:
            if result.get("name") == "save_insight":
                args = result.get("arguments", {})
                if "domain" in args:
                    try:
                        return Domain(args["domain"])
                    except ValueError:
                        pass

        # Check keywords
        if any(kw in text_lower for kw in daily_work_keywords):
            return Domain.DAILY_WORK
        if any(kw in text_lower for kw in recipe_keywords):
            return Domain.RECIPE
        if any(kw in text_lower for kw in postmortem_keywords):
            return Domain.POSTMORTEM
        if any(kw in text_lower for kw in creative_keywords):
            return Domain.CREATIVE

        return Domain.GENERAL

    async def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of the session.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with session summary.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        all_insights: list[Insight] = []
        all_knowledge: list[KnowledgeItem] = []

        for turn in session.turns:
            all_insights.extend(turn.insights_saved)
            all_knowledge.extend(turn.knowledge_used)

        return {
            "session_id": session.id,
            "topic": session.topic,
            "language": session.language,
            "domain": session.domain.value,
            "turn_count": len(session.turns),
            "insights_saved": len(all_insights),
            "knowledge_used": len(all_knowledge),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    async def _detect_consistency_issues(
        self,
        user_message: str,
        knowledge_used: list[KnowledgeItem],
        language: str,
    ) -> list[ConsistencyIssue]:
        """Detect potential consistency issues between user message and past knowledge.

        Args:
            user_message: Current user message.
            knowledge_used: Knowledge items retrieved for this turn.
            language: Session language.

        Returns:
            List of detected consistency issues.
        """
        if not knowledge_used:
            return []

        # Build context for LLM to analyze, including fact_id for matching
        knowledge_context = "\n".join([
            f"- [ID:{item.id}] {item.subject}は{item.predicate} 「{item.object}」"
            for item in knowledge_used
        ])
        
        # Create a lookup map for fact_id matching
        knowledge_map = {item.id: item for item in knowledge_used}

        if language.lower() in ("english", "en"):
            system_prompt = """You are a consistency checker. Analyze if the user's current message contradicts or changes any past recorded information.

**CRITICAL: Distinguish between "contradiction" and "change":**
- "contradiction": Impossible to be both true (e.g., "A is responsible" vs "B is responsible for the same task")
- "change": Information has been updated/revised (e.g., deadline moved from date A to date B)

Output ONLY valid JSON array. Each item should have:
- kind: "contradiction" or "change" (use "contradiction" when both cannot be true simultaneously)
- fact_id: The ID from the past record (e.g., "abc-123")
- title: Brief title in 5 words or less
- previous_text: What was recorded before
- current_text: What user says now
- suggested_question: Question to clarify

If no issues found, output empty array: []

Contradiction example:
[{"kind":"contradiction","fact_id":"abc-123","title":"Different person responsible","previous_text":"Tanaka is responsible","current_text":"Yamada is responsible","suggested_question":"Previously Tanaka was mentioned as responsible. Has this changed to Yamada?"}]

Change example:
[{"kind":"change","fact_id":"abc-123","title":"Deadline moved","previous_text":"Deadline is March 31","current_text":"Deadline is February 28","suggested_question":"The deadline was March 31 before. Has it been moved to February 28?"}]"""
        else:
            system_prompt = """あなたは整合性チェッカーです。ユーザーの現在のメッセージが過去の記録と矛盾または変化しているか分析してください。

**重要：「矛盾」と「変更」を区別してください：**
- "contradiction"（矛盾）: 両方が同時に真であることが不可能（例：「Aさんが担当」vs「同じタスクをBさんが担当」）
- "change"（変更）: 情報が更新・修正された（例：締め切りが日付Aから日付Bに変更）

有効なJSON配列のみを出力してください。各項目には以下を含めます：
- kind: "contradiction" または "change"（両方が同時に真であり得ない場合は "contradiction"）
- fact_id: 過去の記録のID（例: "abc-123"）
- title: 5語以内の簡潔なタイトル
- previous_text: 以前記録されていた内容
- current_text: 現在ユーザーが言っている内容
- suggested_question: 確認のための質問

問題がない場合は空の配列を出力: []

矛盾の例:
[{"kind":"contradiction","fact_id":"abc-123","title":"担当者が異なる","previous_text":"田中さんが担当","current_text":"山田さんが担当","suggested_question":"以前は田中さんが担当とのことでしたが、山田さんに変更になりましたか？"}]

変更の例:
[{"kind":"change","fact_id":"abc-123","title":"締め切り変更","previous_text":"締め切りは3月31日","current_text":"締め切りは2月28日","suggested_question":"以前は3月31日が締め切りでしたが、2月28日に変更になりましたか？"}]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"過去の記録:\n{knowledge_context}\n\n現在のメッセージ:\n{user_message}"},
        ]

        try:
            import json
            response = await self._llm.chat(messages=messages)  # type: ignore
            content = response.content or "[]"
            
            # Clean up response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()
            
            # Parse JSON
            issues_data = json.loads(content)
            if not isinstance(issues_data, list):
                return []

            issues = []
            for item in issues_data:
                try:
                    kind = ConsistencyIssueKind(item.get("kind", "change"))
                    fact_id = item.get("fact_id")
                    
                    # Validate fact_id exists in our knowledge
                    if fact_id and fact_id not in knowledge_map:
                        fact_id = None  # Invalid ID, set to None
                    
                    issues.append(ConsistencyIssue(
                        kind=kind,
                        title=item.get("title", "整合性チェック"),
                        fact_id=fact_id,
                        previous_text=item.get("previous_text", ""),
                        previous_source="過去の記録",
                        current_text=item.get("current_text", ""),
                        current_source="現在の会話",
                        suggested_question=item.get("suggested_question", ""),
                        confidence=0.7,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse consistency issue: {e}")
                    continue

            if issues:
                logger.info(f"Detected {len(issues)} consistency issues")

            return issues

        except Exception as e:
            logger.warning(f"Failed to detect consistency issues: {e}")
            return []
