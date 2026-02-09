"""Eager Learner (EL) - LLM-native interview agent with knowledge graph integration."""

from __future__ import annotations

import asyncio
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
    DateType,
    Document,
    DocumentChunk,
    DocumentExtractionResult,
    Domain,
    ExtractedFact,
    Insight,
    KnowledgeItem,
    PendingQuestion,
    QuestionKind,
    QuestionStatus,
    Session,
    SessionSummary,
    Tag,
    TagSuggestion,
    TagSuggestionResult,
    TopicSummary,
)
from el_core.stores.kg_store import KnowledgeGraphStore
from el_core.tools import ALL_TOOLS, SAVE_ONLY_TOOLS, ToolExecutor

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


def extract_dates_from_text(text: str) -> tuple[Any, Any]:
    """Extract date references from text for search filtering.

    Supports:
    - Japanese relative dates: 今日, 昨日, 一昨日, 先週, 今週, etc.
    - English relative dates: today, yesterday, last week, etc.
    - Japanese date formats: 2024年5月1日, 5月1日, 5/1, 5月
    - English date formats: May 1, 2024, May 1st, May 2024
    - ISO format: 2024-05-01

    Args:
        text: Text to extract dates from.

    Returns:
        Tuple of (start_date, end_date) as datetime objects, or (None, None) if no dates found.
    """
    import re
    from datetime import datetime, timedelta

    now = datetime.now()
    current_year = now.year
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ===== RELATIVE DATES (Japanese) =====
    # Check for relative date references first (highest priority)
    text_lower = text.lower()
    
    # 一昨日 (day before yesterday)
    if "一昨日" in text or "おととい" in text:
        target = today - timedelta(days=2)
        return target, target
    
    # 昨日 (yesterday)
    if "昨日" in text or "きのう" in text:
        target = today - timedelta(days=1)
        return target, target
    
    # 今日 (today)
    if "今日" in text or "きょう" in text or "本日" in text:
        return today, today
    
    # 明日 (tomorrow)
    if "明日" in text or "あした" in text or "あす" in text:
        target = today + timedelta(days=1)
        return target, target
    
    # 明後日 (day after tomorrow)
    if "明後日" in text or "あさって" in text:
        target = today + timedelta(days=2)
        return target, target
    
    # 先週 (last week)
    if "先週" in text:
        # Last week: Monday to Sunday of the previous week
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday
    
    # 今週 (this week)
    if "今週" in text:
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        this_sunday = this_monday + timedelta(days=6)
        return this_monday, this_sunday
    
    # 先月 (last month)
    if "先月" in text:
        first_of_this_month = today.replace(day=1)
        last_of_last_month = first_of_this_month - timedelta(days=1)
        first_of_last_month = last_of_last_month.replace(day=1)
        return first_of_last_month, last_of_last_month
    
    # 今月 (this month)
    if "今月" in text:
        first_of_this_month = today.replace(day=1)
        if today.month == 12:
            last_of_this_month = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            last_of_this_month = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return first_of_this_month, last_of_this_month
    
    # N日前 (N days ago)
    n_days_ago_match = re.search(r"(\d+)日前", text)
    if n_days_ago_match:
        days = int(n_days_ago_match.group(1))
        target = today - timedelta(days=days)
        return target, target
    
    # N週間前 (N weeks ago)
    n_weeks_ago_match = re.search(r"(\d+)週間前", text)
    if n_weeks_ago_match:
        weeks = int(n_weeks_ago_match.group(1))
        target = today - timedelta(weeks=weeks)
        return target, target
    
    # ===== RELATIVE DATES (English) =====
    if "day before yesterday" in text_lower:
        target = today - timedelta(days=2)
        return target, target
    
    if "yesterday" in text_lower:
        target = today - timedelta(days=1)
        return target, target
    
    if "today" in text_lower:
        return today, today
    
    if "tomorrow" in text_lower:
        target = today + timedelta(days=1)
        return target, target
    
    if "day after tomorrow" in text_lower:
        target = today + timedelta(days=2)
        return target, target
    
    if "last week" in text_lower:
        days_since_monday = today.weekday()
        last_monday = today - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        return last_monday, last_sunday
    
    if "this week" in text_lower:
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        this_sunday = this_monday + timedelta(days=6)
        return this_monday, this_sunday
    
    if "last month" in text_lower:
        first_of_this_month = today.replace(day=1)
        last_of_last_month = first_of_this_month - timedelta(days=1)
        first_of_last_month = last_of_last_month.replace(day=1)
        return first_of_last_month, last_of_last_month
    
    # N days ago
    n_days_ago_en = re.search(r"(\d+)\s*days?\s*ago", text_lower)
    if n_days_ago_en:
        days = int(n_days_ago_en.group(1))
        target = today - timedelta(days=days)
        return target, target
    
    # ===== ABSOLUTE DATES =====
    # Japanese date patterns
    jp_full_date = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    jp_month_day = re.search(r"(\d{1,2})月(\d{1,2})日", text)
    jp_month_only = re.search(r"(\d{1,2})月(?!日)", text)
    
    # Slash format: 1/12, 12/25 (month/day) - common Japanese diary format
    slash_date = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", text)

    # English date patterns
    en_full_date = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})",
        text,
        re.IGNORECASE,
    )
    en_month_day = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?",
        text,
        re.IGNORECASE,
    )
    en_month_only = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*(\d{4})?",
        text,
        re.IGNORECASE,
    )

    # ISO format
    iso_date = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    try:
        # Try Japanese full date first
        if jp_full_date:
            year, month, day = int(jp_full_date.group(1)), int(jp_full_date.group(2)), int(jp_full_date.group(3))
            date = datetime(year, month, day)
            return date, date

        # ISO format
        if iso_date:
            year, month, day = int(iso_date.group(1)), int(iso_date.group(2)), int(iso_date.group(3))
            date = datetime(year, month, day)
            return date, date

        # English full date
        if en_full_date:
            month = month_map[en_full_date.group(1).lower()[:3]]
            day = int(en_full_date.group(2))
            year = int(en_full_date.group(3))
            date = datetime(year, month, day)
            return date, date

        # Japanese month + day
        if jp_month_day:
            month, day = int(jp_month_day.group(1)), int(jp_month_day.group(2))
            date = datetime(current_year, month, day)
            return date, date

        # Slash format: 1/12 -> January 12
        if slash_date:
            month, day = int(slash_date.group(1)), int(slash_date.group(2))
            # Validate month and day
            if 1 <= month <= 12 and 1 <= day <= 31:
                date = datetime(current_year, month, day)
                return date, date

        # English month + day
        if en_month_day:
            month = month_map[en_month_day.group(1).lower()[:3]]
            day = int(en_month_day.group(2))
            date = datetime(current_year, month, day)
            return date, date

        # Japanese month only - return range for the whole month
        if jp_month_only:
            month = int(jp_month_only.group(1))
            start_date = datetime(current_year, month, 1)
            # End of month
            if month == 12:
                end_date = datetime(current_year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(current_year, month + 1, 1) - timedelta(days=1)
            return start_date, end_date

        # English month only
        if en_month_only:
            month = month_map[en_month_only.group(1).lower()[:3]]
            year = int(en_month_only.group(2)) if en_month_only.group(2) else current_year
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            return start_date, end_date

    except (ValueError, AttributeError):
        pass

    return None, None


class ELAgent:
    """Eager Learner (EL) - A curious and empathetic interviewer powered by LLM.

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
        """Initialize the Eager Learner agent.

        Args:
            llm_client: LLM client for chat completions. Creates default if None.
            kg_store: Knowledge graph store. Tools will work in local-only mode if None.
        """
        self._llm = llm_client or LLMClient()
        self._kg_store = kg_store
        self._sessions: dict[str, Session] = {}
        # Set to hold references to background tasks and prevent GC
        self._background_tasks: set[asyncio.Task[Any]] = set()

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

        # Save the initial exchange (user topic + opening message) as the first conversation turn
        # This ensures the user's initial input is preserved in conversation history
        initial_turn = ConversationTurn(
            user_message=topic,
            assistant_response=opening,
        )
        session.add_turn(initial_turn)

        # Persist the initial turn to Neo4j
        if self._kg_store:
            try:
                await self._kg_store.save_conversation_turn(
                    session_id=session_id,
                    turn_index=0,
                    user_message=topic,
                    assistant_response=opening,
                    timestamp=initial_turn.timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to save initial conversation turn: {e}")

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
                "- Keep responses conversational and warm, but always polite\n"
                "- Ask 1-2 open-ended questions to start the dialogue\n"
                "- Be concise but engaging\n"
                "- Do NOT output any JSON, tool calls, or code\n"
            )
        else:
            system_content = (
                "あなたは「EL」です。知的好奇心が旺盛で、品のある丁寧な言葉遣いをするインタビュワーです。"
                "相手の話に心から関心を持ち、深く理解しようとする姿勢で対話を行います。\n\n"
                "話し方のルール（厳守）：\n"
                "- 必ず「です・ます」調の丁寧語を使う（タメ口・くだけた口語は禁止）\n"
                "- 上品で知的、かつ温かみのある話し方を心がける\n"
                "- 「しよ」「っぽい」「だよね」「じゃん」のようなカジュアル表現は使わない\n"
                "- 「〜なんですね」「〜でしょうか？」「〜いただけますか？」のような柔らかい敬体を使う\n\n"
                "ガイドライン：\n"
                "- 共感と関心を丁寧に示す\n"
                "- 会話を始めるために1つだけ質問をする\n"
                "- 簡潔に、3〜4文以内で応答する\n"
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
                "丁寧で温かく応答してください。ユーザーの入力を受け止め、関心を示し、"
                "会話を始めるための適切なフォローアップの質問を1つしてください。"
                "必ず「です・ます」調で、上品かつ簡潔に応答してください。"
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
        # First check if session is already in memory (with conversation history)
        existing_session = self._sessions.get(session_id)
        if existing_session is not None and existing_session.user_id == user_id:
            # Session is already loaded with conversation history
            logger.info(f"Resuming session {session_id} from memory ({len(existing_session.turns)} turns)")
            
            # Count pending questions
            pending_count = sum(
                1 for q in existing_session.pending_questions
                if q.status == QuestionStatus.PENDING
            )
            
            # Generate resume message
            lang = existing_session.language
            if lang.lower() in ("english", "en"):
                resume_msg = (
                    f"Welcome back! We were discussing \"{existing_session.topic}\". "
                    f"You have {len(existing_session.turns)} exchanges so far. "
                )
                if pending_count > 0:
                    resume_msg += f"You have {pending_count} unanswered questions from our last conversation. "
                resume_msg += "Let's continue!"
            else:
                resume_msg = (
                    f"「{existing_session.topic}」の続きですね。\n"
                    f"これまで{len(existing_session.turns)}ターンの会話がありました。\n"
                )
                if pending_count > 0:
                    resume_msg += f"前回の未回答質問が{pending_count}件あります。\n"
                resume_msg += "続けましょう！"
            
            return session_id, resume_msg, existing_session.prior_knowledge
        
        # Session not in memory, load from Neo4j
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

        # Load conversation history from Neo4j
        conversation_turns = await self._kg_store.get_conversation_history(session_id)
        for turn_data in conversation_turns:
            # Restore turn to session
            turn = ConversationTurn(
                user_message=turn_data["user_message"],
                assistant_response=turn_data["assistant_response"],
                timestamp=turn_data["timestamp"],
            )
            session.turns.append(turn)
            # Also restore to message_history for LLM context
            session.message_history.append({"role": "user", "content": turn_data["user_message"]})
            session.message_history.append({"role": "assistant", "content": turn_data["assistant_response"]})

        # Restore pending questions from Neo4j
        restored_questions: list[PendingQuestion] = []
        try:
            restored_questions = await self._kg_store.get_unresolved_questions_for_session(session_id)
            if restored_questions:
                session.pending_questions = restored_questions
                logger.info(f"Restored {len(restored_questions)} pending questions for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to restore pending questions: {e}")

        # Check if this is a document review session and restore document_id
        try:
            review_doc_id = await self._kg_store.get_session_review_document_id(session_id)
            if review_doc_id:
                session.review_document_id = review_doc_id
                logger.info(f"Restored review_document_id={review_doc_id} for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to restore review_document_id: {e}")

        # Store in active sessions
        self._sessions[session_id] = session

        # Generate resume message
        has_history = len(conversation_turns) > 0
        pending_count = len(restored_questions)

        if detected_lang.lower() in ("english", "en"):
            if has_history:
                resume_msg = (
                    f"Welcome back! We were discussing \"{metadata.topic}\". "
                    f"Your previous {len(conversation_turns)} exchanges have been restored. "
                )
                if pending_count > 0:
                    resume_msg += f"You have {pending_count} unanswered questions from our last conversation. "
                resume_msg += "Let's continue!"
            else:
                resume_msg = (
                    f"Welcome back! We were discussing \"{metadata.topic}\". "
                    f"(Note: No conversation history found. Starting fresh but with your insights!)"
                )
        else:
            if has_history:
                resume_msg = (
                    f"「{metadata.topic}」の続きですね。お帰りなさい！\n"
                    f"前回の{len(conversation_turns)}ターンの会話を復元しました。\n"
                )
                if pending_count > 0:
                    resume_msg += f"前回の未回答質問が{pending_count}件あります。\n"
                resume_msg += "続けましょう！"
            else:
                resume_msg = (
                    f"「{metadata.topic}」の続きですね。お帰りなさい！\n"
                    f"（※会話履歴が見つかりませんでしたが、記録した事実は引き継いでいます）"
                )

        logger.info(f"Resumed session {session_id} for user {user_id} ({len(conversation_turns)} turns, {pending_count} pending questions restored)")

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
            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_MID)  # type: ignore
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

        Enhanced flow:
        1. Retrieve pending questions from session
        2. Analyze if user message answers any pending questions
        3. Apply answers to knowledge graph automatically
        4. Detect new consistency issues and missing information
        5. Generate response with question context
        6. Track questions asked/answered in the turn

        Args:
            session_id: Session identifier.
            user_message: The user's message.

        Returns:
            AgentResponse with the assistant's message and metadata.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        from datetime import datetime

        # ===== Phase A: Analyze answers + Pre-search KG in parallel =====
        answered_question_ids: list[str] = []
        existing_pending = [q for q in session.pending_questions if q.status == QuestionStatus.PENDING]

        # Detect domain early so we can use it for filtering searches
        detected_domain = self._detect_domain(user_message, [])
        # Use session domain if current message domain is general (to maintain context)
        search_domain = detected_domain if detected_domain != Domain.GENERAL else session.domain if session.domain != Domain.GENERAL else None

        # Prepare parallel tasks for Phase A
        answer_analysis_coro = (
            self._analyze_answer_to_questions(
                user_message=user_message,
                pending_questions=existing_pending,
                language=session.language,
            )
            if existing_pending
            else None
        )
        kg_search_coro = (
            self._pre_search_knowledge(
                user_message=user_message,
                session=session,
                search_domain=search_domain,
            )
            if self._kg_store
            else None
        )

        # Run Phase A tasks in parallel
        if answer_analysis_coro and kg_search_coro:
            answer_analyses, kg_result = await asyncio.gather(
                answer_analysis_coro, kg_search_coro
            )
        elif answer_analysis_coro:
            answer_analyses = await answer_analysis_coro
            kg_result = None
        elif kg_search_coro:
            answer_analyses = []
            kg_result = await kg_search_coro
        else:
            answer_analyses = []
            kg_result = None

        # ===== Step 2: Apply answers to knowledge graph =====
        if existing_pending and answer_analyses:
            pending_map = {q.id: q for q in existing_pending}
            for analysis in answer_analyses:
                qid = analysis["question_id"]
                pq = pending_map.get(qid)
                if pq and analysis.get("answered"):
                    pq.status = QuestionStatus.ANSWERED
                    pq.answer = analysis.get("new_value") or user_message
                    pq.answered_at = datetime.now()
                    answered_question_ids.append(qid)
                    await self._apply_answer_to_knowledge(analysis, pq, session_id)
                    logger.info(f"Question {qid} answered: action={analysis.get('action')}")
                elif pq and analysis.get("action") == "skip":
                    pq.status = QuestionStatus.SKIPPED
                    answered_question_ids.append(qid)

        # Unpack KG search results
        pre_search_knowledge: list[KnowledgeItem] = []
        consistency_issues: list[ConsistencyIssue] = []
        new_pending_questions: list[PendingQuestion] = []
        consistency_context = ""
        questions_context = ""
        chunk_context = ""
        relevant_chunks: list[DocumentChunk] = []

        if kg_result:
            pre_search_knowledge = kg_result.get("knowledge", [])
            relevant_chunks = kg_result.get("chunks", [])
            chunk_context = kg_result.get("chunk_context", "")

        # ===== Document Review Context: load reviewed document content =====
        review_document_context = ""
        if session.review_document_id and self._kg_store:
            try:
                doc_chunks = await self._kg_store.get_chunks_by_document(session.review_document_id)
                if doc_chunks:
                    # Include the full document content as authoritative context
                    review_document_context = self._format_review_document_context(
                        doc_chunks, session.language
                    )
                    logger.info(
                        f"Loaded {len(doc_chunks)} chunks for review document "
                        f"{session.review_document_id}"
                    )
                else:
                    # Fallback: load document's raw content preview
                    doc = await self._kg_store.get_document(session.review_document_id)
                    if doc and doc.raw_content_preview:
                        review_document_context = self._format_review_document_preview(
                            doc, session.language
                        )
                        logger.info(f"Loaded document preview for review")
            except Exception as e:
                logger.warning(f"Failed to load review document context: {e}")

        # ===== Phase B: Detect new questions (contradictions + missing info) =====
        if self._kg_store:
            try:
                new_pending_questions, consistency_issues = await self._generate_all_questions(
                    user_message=user_message,
                    knowledge_used=pre_search_knowledge,
                    extracted_facts=None,
                    domain=detected_domain,
                    language=session.language,
                    session_id=session_id,
                )

                if consistency_issues:
                    consistency_context = self._format_consistency_context(
                        consistency_issues, session.language
                    )
                    logger.info(f"Found {len(consistency_issues)} consistency issues to address")

            except Exception as e:
                logger.warning(f"Pre-search for consistency failed: {e}")

        # ===== Step 5: Build pending questions context for LLM =====
        # Combine remaining unanswered questions + new questions
        still_pending = [q for q in session.pending_questions if q.status == QuestionStatus.PENDING]
        all_active_questions = still_pending + new_pending_questions

        # Sort by priority and deduplicate
        seen_question_texts = set()
        deduped_questions: list[PendingQuestion] = []
        for q in sorted(all_active_questions, key=lambda x: x.priority, reverse=True):
            if q.question not in seen_question_texts:
                deduped_questions.append(q)
                seen_question_texts.add(q.question)
        all_active_questions = deduped_questions

        if all_active_questions:
            questions_context = self._format_questions_context(all_active_questions, session.language)

        # ===== Step 6: Build messages for LLM =====
        system_content = get_system_prompt(session.language)
        if session.prior_context:
            system_content += session.prior_context
        
        # Inject reviewed document content as primary context
        if review_document_context:
            system_content += review_document_context
        
        if chunk_context:
            system_content += chunk_context
        
        # Inject pre-searched knowledge (extracted facts) as reference context
        if pre_search_knowledge:
            knowledge_context = self._format_knowledge_context(
                pre_search_knowledge, session.language
            )
            system_content += knowledge_context
        
        if consistency_context:
            system_content += consistency_context

        if questions_context:
            system_content += questions_context
        
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        messages.extend(session.message_history)
        messages.append({"role": "user", "content": user_message})

        # Set up tool executor with session_id for insight tracking
        tool_executor = ToolExecutor(
            self._kg_store,
            session_id=session_id,
            session_domain=search_domain,
        )

        # Call LLM with tools
        # Only strip search_knowledge_graph when date-matched chunks exist (high confidence).
        # Keyword-based pre_search_knowledge may include false positives, so keep ALL_TOOLS
        # to let the LLM do its own targeted search when the injected context isn't relevant.
        if chunk_context:
            # Date-based chunk matches are high-confidence — no need for LLM to re-search
            tools_to_use = SAVE_ONLY_TOOLS
            tool_executor.used_knowledge.extend(pre_search_knowledge)
        else:
            # Keyword search results may be irrelevant; let LLM search if needed.
            # pre_search_knowledge is still injected into the prompt as context
            # but NOT tracked as "used" to avoid showing false positives in the UI.
            tools_to_use = ALL_TOOLS
        response_text, tool_results = await self._llm.chat_with_tools(
            messages=messages,  # type: ignore
            tools=tools_to_use,
            tool_handlers=tool_executor.get_tool_handlers(),
        )

        # Detect domain from response and tool usage
        detected_domain = self._detect_domain(user_message, tool_results)

        # ===== Step 7: Update session state =====
        # Mark asked_at for new questions
        questions_asked_ids: list[str] = []
        for q in new_pending_questions:
            q.asked_at = datetime.now()
            questions_asked_ids.append(q.id)

        # Update session's pending_questions: keep answered/skipped for history, add new ones
        session.pending_questions = [
            q for q in session.pending_questions
        ] + new_pending_questions

        # Create conversation turn with question tracking
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=response_text,
            insights_saved=tool_executor.saved_insights,
            knowledge_used=tool_executor.used_knowledge,
            detected_domain=detected_domain,
            questions_asked=questions_asked_ids,
            questions_answered=answered_question_ids,
        )
        session.add_turn(turn)

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
            f"detected {len(consistency_issues)} consistency issues, "
            f"new questions {len(new_pending_questions)}, "
            f"answered {len(answered_question_ids)}"
        )

        # ===== Offload KG persistence + tagging to background =====
        if self._kg_store:
            # Capture values needed by background task before returning
            _saved_insight_ids = list(tool_executor.saved_insight_ids) if tool_executor.saved_insight_ids else []
            _turn_index = len(session.turns) - 1
            _turn_timestamp = turn.timestamp
            _pending_question_statuses = []
            if answered_question_ids:
                for qid in answered_question_ids:
                    pq = next((q for q in session.pending_questions if q.id == qid), None)
                    if pq:
                        _pending_question_statuses.append((qid, pq.status.value, pq.answer))

            bg_task = asyncio.create_task(
                self._post_respond_background(
                    session=session,
                    session_id=session_id,
                    user_message=user_message,
                    response_text=response_text,
                    turn_index=_turn_index,
                    turn_timestamp=_turn_timestamp,
                    saved_insight_ids=_saved_insight_ids,
                    new_pending_questions=new_pending_questions,
                    pending_question_statuses=_pending_question_statuses,
                )
            )
            # Hold a reference to prevent GC from collecting the task
            self._background_tasks.add(bg_task)
            bg_task.add_done_callback(self._background_tasks.discard)

        return AgentResponse(
            message=response_text,
            tool_calls=tool_calls,  # type: ignore
            insights_saved=tool_executor.saved_insights,
            knowledge_used=tool_executor.used_knowledge,
            detected_domain=detected_domain,
            consistency_issues=consistency_issues,
            pending_questions=[q for q in all_active_questions if q.status == QuestionStatus.PENDING],
            questions_answered=answered_question_ids,
        )

    async def _post_respond_background(
        self,
        *,
        session: Session,
        session_id: str,
        user_message: str,
        response_text: str,
        turn_index: int,
        turn_timestamp: Any,
        saved_insight_ids: list[tuple[str, str]],
        new_pending_questions: list[PendingQuestion],
        pending_question_statuses: list[tuple[str, str, str | None]],
    ) -> None:
        """Execute post-respond persistence and tagging in the background.

        This runs as an asyncio.create_task after the response is returned
        to the user, so it doesn't add latency to the user experience.
        """
        # 1. Save session metadata
        try:
            await self._save_session_metadata(session)
        except Exception as e:
            logger.warning(f"[BG] Failed to update session metadata: {e}")

        # 2. Save conversation turn
        try:
            await self._kg_store.save_conversation_turn(  # type: ignore[union-attr]
                session_id=session_id,
                turn_index=turn_index,
                user_message=user_message,
                assistant_response=response_text,
                timestamp=turn_timestamp,
            )
        except Exception as e:
            logger.warning(f"[BG] Failed to save conversation turn: {e}")

        # 3. Auto-tag saved insights
        if saved_insight_ids:
            try:
                for insight_id, content in saved_insight_ids:
                    try:
                        await self.auto_tag_insight(
                            insight_id=insight_id,
                            content=content,
                            max_tags=2,
                        )
                    except Exception as e:
                        logger.warning(f"[BG] Failed to auto-tag insight {insight_id}: {e}")

                logger.info(
                    f"[BG] Auto-tagged {len(saved_insight_ids)} conversation-derived insights"
                )
            except Exception as e:
                logger.warning(f"[BG] Failed to auto-tag conversation insights: {e}")

        # 4. Persist new pending questions
        if new_pending_questions:
            try:
                await self._kg_store.save_pending_questions_batch(  # type: ignore[union-attr]
                    new_pending_questions, session_id
                )
            except Exception as e:
                logger.warning(f"[BG] Failed to persist pending questions: {e}")

        # 5. Persist answered question status updates
        for qid, status, answer in pending_question_statuses:
            try:
                await self._kg_store.update_question_status(  # type: ignore[union-attr]
                    question_id=qid,
                    status=status,
                    answer=answer,
                )
            except Exception as e:
                logger.warning(f"[BG] Failed to update question {qid} status in KG: {e}")

        logger.info(f"[BG] Post-respond background tasks completed for session {session_id}")

    async def _pre_search_knowledge(
        self,
        user_message: str,
        session: Session,
        search_domain: Domain | None,
    ) -> dict[str, Any]:
        """Pre-search the knowledge graph for relevant context.

        Extracted from respond() to enable parallel execution with answer analysis.

        Returns:
            Dict with keys: knowledge (list[KnowledgeItem]), chunks (list[DocumentChunk]),
            chunk_context (str).
        """
        pre_search_knowledge: list[KnowledgeItem] = []
        relevant_chunks: list[DocumentChunk] = []
        chunk_context = ""

        try:
            # Extract dates from user message for temporal filtering
            start_date, end_date = extract_dates_from_text(user_message)

            # If a date is detected, save it to session for future reference
            if start_date:
                if start_date not in session.referenced_dates:
                    session.referenced_dates.append(start_date)
                    logger.info(f"Added date to session context: {start_date}")

            # If no date in current message, use previously referenced dates
            if not start_date and session.referenced_dates:
                start_date = session.referenced_dates[-1]
                end_date = start_date
                logger.info(f"Using previously referenced date: {start_date}")

            if start_date or end_date:
                logger.info(f"Detected date reference: {start_date} to {end_date}")

                if start_date and end_date and start_date == end_date:
                    relevant_chunks = await self._kg_store.get_chunks_by_date(start_date)  # type: ignore[union-attr]
                elif start_date and end_date:
                    relevant_chunks = await self._kg_store.get_chunks_by_date_range(start_date, end_date)  # type: ignore[union-attr]
                elif start_date:
                    relevant_chunks = await self._kg_store.get_chunks_by_date(start_date)  # type: ignore[union-attr]

                if relevant_chunks:
                    chunk_context = self._format_chunk_context(relevant_chunks, session.language)
                    logger.info(f"Found {len(relevant_chunks)} relevant chunks for date query")
                    logger.info("Skipping fact search - using chunk original content as authoritative source")
                else:
                    date_filtered_knowledge = await self._kg_store.search_by_date_range(  # type: ignore[union-attr]
                        start_date=start_date,
                        end_date=end_date,
                        query=user_message,
                        domain=search_domain,
                        limit=5,
                    )
                    pre_search_knowledge.extend(date_filtered_knowledge)

            if not relevant_chunks:
                keyword_knowledge = await self._kg_store.search(  # type: ignore[union-attr]
                    user_message,
                    limit=5,
                    domain=search_domain,
                    start_date=start_date,
                    end_date=end_date,
                )

                seen_ids = {k.id for k in pre_search_knowledge}
                for item in keyword_knowledge:
                    if item.id not in seen_ids:
                        pre_search_knowledge.append(item)
                        seen_ids.add(item.id)

        except Exception as e:
            logger.warning(f"Pre-search knowledge failed: {e}")

        return {
            "knowledge": pre_search_knowledge,
            "chunks": relevant_chunks,
            "chunk_context": chunk_context,
        }

    def _format_knowledge_context(
        self,
        knowledge: list[KnowledgeItem],
        language: str,
    ) -> str:
        """Format pre-searched knowledge items as interviewing context for LLM.

        EL is an interviewer. These facts are provided so EL can form better,
        more targeted questions — NOT to answer the user's questions directly.

        Args:
            knowledge: List of relevant knowledge items from pre-search.
            language: Session language.

        Returns:
            Formatted context string to append to system prompt.
        """
        if not knowledge:
            return ""

        if language.lower() in ("english", "en"):
            context = "\n\n### 📚 Known Facts (for forming better questions)\n"
            context += "The following facts are already recorded in the knowledge base.\n"
            context += "**Use these to ask deeper, more specific follow-up questions:**\n"
            context += "- Reference these facts to explore contradictions, changes, or missing details\n"
            context += "- Ask about the 'why' behind these facts, or probe for nuance\n"
            context += "- Do NOT simply recite these facts back as answers\n\n"
        else:
            context = "\n\n### 📚 記録済みファクト（より良い質問を形成するために使用）\n"
            context += "以下は知識ベースに記録済みのファクトです。\n"
            context += "**これらを元に、より深い質問を行ってください：**\n"
            context += "- これらのファクトを参照して、矛盾・変化・不足している詳細を探る質問をする\n"
            context += "- ファクトの背景や理由を深掘りする質問をする\n"
            context += "- ファクトをそのまま回答として返すのではなく、質問の材料として活用する\n\n"

        for i, item in enumerate(knowledge, 1):
            date_str = ""
            if item.event_date:
                date_str = f" [{item.event_date.strftime('%Y-%m-%d')}"
                if item.event_date_end and item.event_date_end != item.event_date:
                    date_str += f"〜{item.event_date_end.strftime('%Y-%m-%d')}"
                date_str += "]"
            context += f"- {item.subject} {item.predicate} {item.object}{date_str}\n"

        context += "\n"
        return context

    def _format_chunk_context(
        self,
        chunks: list[DocumentChunk],
        language: str,
    ) -> str:
        """Format document chunks as context for LLM.

        Phase 2: Provides the original document content for accurate reference.

        Args:
            chunks: List of relevant document chunks.
            language: Session language.

        Returns:
            Formatted context string to append to system prompt.
        """
        if not chunks:
            return ""

        if language.lower() in ("english", "en"):
            context = "\n\n### 📄 Document Content (Reference for Deeper Questions)\n"
            context += "The following is content from the user's uploaded documents.\n"
            context += "**Use this to form informed, specific questions:**\n"
            context += "1. Reference specific details from the document to ask follow-up questions\n"
            context += "2. If the document mentions something vague or incomplete, ask about it\n"
            context += "3. Use document content to probe for context, reasons, or details behind the facts\n"
            context += "4. When quoting the document, be accurate — do not paraphrase or alter the original text\n\n"
        else:
            context = "\n\n### 📄 ドキュメント内容（より深い質問のための参照情報）\n"
            context += "以下はユーザーがアップロードしたドキュメントの内容です。\n"
            context += "**これを活用して、より具体的な質問を行ってください：**\n"
            context += "1. ドキュメントの具体的な記述を引用しながら、深掘り質問をする\n"
            context += "2. 曖昧・不完全な記載があれば、その詳細を質問する\n"
            context += "3. ドキュメントの内容をもとに、背景・理由・経緯を探る質問をする\n"
            context += "4. ドキュメントを引用する際は、原文を正確に引用すること\n\n"

        for chunk in chunks:
            # Add date header if available
            if chunk.chunk_date:
                date_str = chunk.chunk_date.strftime("%Y年%m月%d日") if language.lower() not in ("english", "en") else chunk.chunk_date.strftime("%Y-%m-%d")
                context += f"---\n📅 {date_str}"
                if chunk.heading:
                    context += f" - {chunk.heading}"
                context += "\n\n"
            elif chunk.heading:
                context += f"---\n📄 {chunk.heading}\n\n"
            else:
                context += "---\n\n"

            # Add the original content (preserved exactly as uploaded)
            context += "【原文ここから】\n"
            context += chunk.content
            context += "\n【原文ここまで】\n\n"

        context += "---\n"
        context += "上記の原文から、質問に該当する部分をそのまま引用して回答してください。\n"
        
        return context

    def _format_review_document_context(
        self,
        chunks: list[DocumentChunk],
        language: str,
    ) -> str:
        """Format the reviewed document's full content as context for LLM.

        Unlike _format_chunk_context (which is for date-based search results),
        this method provides the entire reviewed document as authoritative context
        for document review sessions.

        Args:
            chunks: All chunks from the reviewed document.
            language: Session language.

        Returns:
            Formatted context string to prepend to system prompt.
        """
        if not chunks:
            return ""

        if language.lower() in ("english", "en"):
            context = "\n\n### 📋 Reviewed Document Content\n"
            context += "You are reviewing the following document. "
            context += "You have full access to its content — DO NOT ask the user to paste it.\n"
            context += "Use this content to:\n"
            context += "1. Answer questions about the document accurately\n"
            context += "2. Reference specific details when discussing the document\n"
            context += "3. Identify missing information or inconsistencies\n\n"
        else:
            context = "\n\n### 📋 レビュー対象ドキュメントの内容\n"
            context += "あなたは以下のドキュメントをレビューしています。"
            context += "ドキュメントの全文にアクセスできます。ユーザーに貼り付けを求めないでください。\n"
            context += "このドキュメント内容を活用して：\n"
            context += "1. ドキュメントに関する質問に正確に回答する\n"
            context += "2. 議論する際にドキュメントの具体的な記述を引用する\n"
            context += "3. 不足情報や矛盾を特定する\n\n"

        for chunk in sorted(chunks, key=lambda c: c.chunk_index):
            if chunk.chunk_date:
                date_str = (
                    chunk.chunk_date.strftime("%Y年%m月%d日")
                    if language.lower() not in ("english", "en")
                    else chunk.chunk_date.strftime("%Y-%m-%d")
                )
                context += f"---\n📅 {date_str}"
                if chunk.heading:
                    context += f" - {chunk.heading}"
                context += "\n\n"
            elif chunk.heading:
                context += f"---\n📄 {chunk.heading}\n\n"

            context += chunk.content
            context += "\n\n"

        context += "---\n"
        return context

    def _format_review_document_preview(
        self,
        doc: Document,
        language: str,
    ) -> str:
        """Format a document's preview content as fallback context.

        Used when chunks are not available but the document's raw preview exists.

        Args:
            doc: Document object with raw_content_preview.
            language: Session language.

        Returns:
            Formatted context string.
        """
        if language.lower() in ("english", "en"):
            context = "\n\n### 📋 Reviewed Document Preview\n"
            context += f"Document: {doc.filename}\n"
            context += "You have the following preview of the document content. "
            context += "DO NOT ask the user to paste the document.\n\n"
        else:
            context = "\n\n### 📋 レビュー対象ドキュメントのプレビュー\n"
            context += f"ドキュメント: {doc.filename}\n"
            context += "以下はドキュメント内容のプレビューです。"
            context += "ユーザーにドキュメントの貼り付けを求めないでください。\n\n"

        context += doc.raw_content_preview
        context += "\n\n---\n"
        return context

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

    def _format_questions_context(
        self,
        questions: list[PendingQuestion],
        language: str,
    ) -> str:
        """Format pending questions as context for LLM.

        Args:
            questions: List of active pending questions.
            language: Session language.

        Returns:
            Formatted context string to append to system prompt.
        """
        if not questions:
            return ""

        # Only include top priority questions (max 5)
        top_questions = questions[:5]

        if language.lower() in ("english", "en"):
            context = "\n\n### Pending Questions to Address\n"
            context += "The following questions are waiting for user response.\n"
            context += "If the user's message answers any of these, acknowledge it. "
            context += "Otherwise, naturally incorporate the highest-priority unanswered question.\n\n"

            for q in top_questions:
                kind_label = {
                    QuestionKind.CONTRADICTION: "CONTRADICTION",
                    QuestionKind.CHANGE: "CHANGE",
                    QuestionKind.MISSING: "MISSING INFO",
                    QuestionKind.CLARIFICATION: "CLARIFICATION",
                }.get(q.kind, "QUESTION")
                context += f"- [{kind_label}] {q.question}\n"
                if q.context:
                    context += f"  Context: {q.context}\n"
        else:
            context = "\n\n### 未回答の質問\n"
            context += "以下の質問がユーザーからの回答を待っています。\n"
            context += "ユーザーのメッセージがこれらに回答している場合は確認してください。\n"
            context += "そうでなければ、優先度の高い未回答質問を自然に会話に組み込んでください。\n\n"

            for q in top_questions:
                kind_label = {
                    QuestionKind.CONTRADICTION: "矛盾",
                    QuestionKind.CHANGE: "変更",
                    QuestionKind.MISSING: "不足情報",
                    QuestionKind.CLARIFICATION: "確認",
                }.get(q.kind, "質問")
                context += f"- 【{kind_label}】{q.question}\n"
                if q.context:
                    context += f"  背景: {q.context}\n"

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
        session_id: str | None = None,
    ) -> list[ConsistencyIssue]:
        """Detect potential consistency issues between user message and past knowledge.

        Args:
            user_message: Current user message.
            knowledge_used: Knowledge items retrieved for this turn.
            language: Session language.
            session_id: Session ID for persistence.

        Returns:
            List of detected consistency issues.
        """
        if not knowledge_used:
            return []

        # Build rich context for LLM including dates and source info
        knowledge_lines = []
        for item in knowledge_used:
            line = f"- [ID:{item.id}] {item.subject}は{item.predicate}「{item.object}」"
            # Add temporal context
            if item.event_date:
                date_str = item.event_date.strftime("%Y年%m月%d日")
                if item.event_date_end:
                    date_str += f"〜{item.event_date_end.strftime('%Y年%m月%d日')}"
                line += f"（イベント日: {date_str}）"
            # Add recording date for context
            if item.created_at:
                line += f"（記録日: {item.created_at.strftime('%Y年%m月%d日')}）"
            knowledge_lines.append(line)
        knowledge_context = "\n".join(knowledge_lines)
        
        # Create a lookup map for fact_id matching
        knowledge_map = {item.id: item for item in knowledge_used}

        if language.lower() in ("english", "en"):
            system_prompt = """You are a consistency checker. Analyze if the user's current message contradicts or changes any past recorded information.

**CRITICAL: Distinguish between "contradiction" and "change":**
- "contradiction": Impossible to be both true (e.g., "A is responsible" vs "B is responsible for the same task")
- "change": Information has been updated/revised (e.g., deadline moved from date A to date B)

Output ONLY valid JSON array. Each item should have:
- kind: "contradiction" or "change"
- fact_id: The ID from the past record (e.g., "abc-123")
- title: Brief descriptive title (e.g., "Exercise routine change")
- previous_text: Full description of what was recorded before, including context (e.g., "On 1/14 (Wed), the recorded exercise routine was: back bends, planks, push-ups, twists, etc.")
- current_text: Full description of what user says now, including context (e.g., "Today 1/4 (Sun), the exercise routine mentioned is: toe touches, forward bends, splits (left/right), frog (front), side plank (left/right)")
- explanation: Detailed 2-3 sentence explanation of what exactly differs and why this matters. Explain the specific discrepancy clearly so the user can make an informed decision.
- suggested_question: A specific, contextual question to help the user resolve this.

**IMPORTANT**: previous_text and current_text should be descriptive full sentences with context, NOT just raw values. Include dates, subjects, and surrounding context so the user understands the full picture.

**CRITICAL - DO NOT flag these as issues:**
- Records with DIFFERENT event dates that describe the same type of activity (e.g., exercise on Jan 4 vs exercise on Jan 14) are NOT contradictions — they are separate events on different days.
- Diary/journal entries for different dates are independent records, not updates to each other.
- Only flag as "change" or "contradiction" when two records describe the SAME event/date but with conflicting details.

If no issues found, output empty array: []"""
        else:
            system_prompt = """あなたは整合性チェッカーです。ユーザーの現在のメッセージが過去の記録と矛盾または変化しているか分析してください。

**重要：「矛盾」と「変更」を区別してください：**
- "contradiction"（矛盾）: 両方が同時に真であることが不可能（例：「Aさんが担当」vs「同じタスクをBさんが担当」）
- "change"（変更）: 情報が更新・修正された（例：締め切りが日付Aから日付Bに変更）

有効なJSON配列のみを出力してください。各項目には以下を含めます：
- kind: "contradiction" または "change"
- fact_id: 過去の記録のID（例: "abc-123"）
- title: 内容が分かる簡潔なタイトル（例:「運動メニューの変更」「日記の日付の不一致」）
- previous_text: 以前の記録の**詳細な説明**（日付・文脈を含む完全な文で記述。例:「1/14(水)の記録では、運動メニューは反り・プランク前後・腕立て・捻り等でした」）
- current_text: 現在の情報の**詳細な説明**（日付・文脈を含む完全な文で記述。例:「今回1/4(日)の記録では、運動メニューは足先・前屈・開脚(左右)・カエル(前)・横プランク(左右)に変わっています」）
- explanation: **具体的に何がどう異なるのか**を2〜3文で詳しく説明してください。ユーザーが判断できるよう、相違点を明確に述べてください。
- suggested_question: ユーザーがこの問題を解決するための**具体的で文脈に即した質問**

**重要**: previous_text と current_text は生の値だけでなく、日付・主語・文脈を含む説明的な完全文にしてください。ユーザーが前後の状況を理解できるようにしてください。

**絶対に矛盾として検出しないでください：**
- イベント日（event_date）が異なる同種の活動の記録（例：1/4の運動メニューと1/14の運動メニュー）は矛盾ではありません。それぞれ別の日の独立した記録です。
- 日記・日誌の異なる日付のエントリは独立した記録であり、互いの更新ではありません。
- 「矛盾」や「変更」として検出するのは、**同じ日付・同じ出来事**について内容が食い違っている場合のみです。

問題がない場合は空の配列を出力: []"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"過去の記録:\n{knowledge_context}\n\n現在のメッセージ:\n{user_message}"},
        ]

        try:
            import json
            from uuid import uuid4
            
            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_MID)  # type: ignore
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
                    
                    # Build source info from matched knowledge item
                    previous_source = "過去の記録"
                    if fact_id and fact_id in knowledge_map:
                        matched = knowledge_map[fact_id]
                        if matched.event_date:
                            previous_source = f"過去の記録（{matched.event_date.strftime('%Y/%m/%d')}）"
                        elif matched.created_at:
                            previous_source = f"過去の記録（{matched.created_at.strftime('%Y/%m/%d')}記録）"
                    
                    # Create issue with ID and session_id
                    issue = ConsistencyIssue(
                        id=str(uuid4()),
                        kind=kind,
                        title=item.get("title", "整合性チェック"),
                        fact_id=fact_id,
                        previous_text=item.get("previous_text", ""),
                        previous_source=previous_source,
                        current_text=item.get("current_text", ""),
                        current_source="現在の会話",
                        suggested_question=item.get("suggested_question", ""),
                        explanation=item.get("explanation", ""),
                        confidence=0.7,
                        session_id=session_id,
                    )
                    issues.append(issue)
                    
                    # Persist to knowledge graph
                    if self._kg_store:
                        try:
                            await self._kg_store.save_consistency_issue(issue, session_id)
                            logger.info(f"Saved consistency issue {issue.id} to KG")
                        except Exception as save_error:
                            logger.warning(f"Failed to save consistency issue to KG: {save_error}")
                            
                except Exception as e:
                    logger.warning(f"Failed to parse consistency issue: {e}")
                    continue

            if issues:
                logger.info(f"Detected {len(issues)} consistency issues")

            return issues

        except Exception as e:
            logger.warning(f"Failed to detect consistency issues: {e}")
            return []

    def _consistency_issue_to_pending_question(
        self,
        issue: ConsistencyIssue,
        session_id: str | None = None,
    ) -> PendingQuestion:
        """Convert a ConsistencyIssue to a PendingQuestion for unified tracking.

        Args:
            issue: The consistency issue to convert.
            session_id: Session ID for the question.

        Returns:
            A PendingQuestion instance.
        """
        kind_map = {
            ConsistencyIssueKind.CONTRADICTION: QuestionKind.CONTRADICTION,
            ConsistencyIssueKind.CHANGE: QuestionKind.CHANGE,
        }
        return PendingQuestion(
            id=issue.id or str(uuid.uuid4()),
            kind=kind_map.get(issue.kind, QuestionKind.CONTRADICTION),
            question=issue.suggested_question,
            context=f"以前: 「{issue.previous_text}」 → 現在: 「{issue.current_text}」",
            related_fact_id=issue.fact_id,
            related_entity=None,
            priority=8 if issue.kind == ConsistencyIssueKind.CONTRADICTION else 5,
            status=QuestionStatus.PENDING,
            session_id=session_id,
            created_at=issue.created_at,
        )

    async def _detect_missing_information(
        self,
        extracted_facts: list[ExtractedFact] | None,
        user_message: str,
        existing_knowledge: list[KnowledgeItem],
        domain: Domain,
        language: str,
        session_id: str | None = None,
    ) -> list[PendingQuestion]:
        """Detect missing information that should be asked about.

        Analyzes extracted facts and existing knowledge to identify gaps.
        Uses LLM to determine what information is expected but missing.

        Args:
            extracted_facts: Facts extracted from a document or message.
            user_message: The current user message for context.
            existing_knowledge: Previously stored knowledge items.
            domain: The detected domain of the conversation.
            language: Session language.
            session_id: Session ID for persistence.

        Returns:
            List of PendingQuestion objects for missing information.
        """
        # Build context from facts and existing knowledge
        facts_text = ""
        if extracted_facts:
            facts_text = "\n".join([
                f"- {f.subject}は{f.predicate}「{f.object}」"
                for f in extracted_facts
            ])

        knowledge_text = ""
        if existing_knowledge:
            knowledge_text = "\n".join([
                f"- [ID:{item.id}] {item.subject}は{item.predicate}「{item.object}」"
                for item in existing_knowledge
            ])

        if language.lower() in ("english", "en"):
            system_prompt = """You are a missing information detector. Analyze the given facts and existing knowledge to identify important gaps.

Think about what information SHOULD exist but is missing. Consider:
- If a project is mentioned, do we know: responsible person, deadline, status, priority?
- If a person is mentioned, do we know: role, team, contact info?
- If an event is mentioned, do we know: date, location, participants, outcome?
- If a decision is mentioned, do we know: reason, alternatives considered, who decided?
- If a problem is mentioned, do we know: impact, root cause, resolution plan?

Only suggest questions for GENUINELY USEFUL missing information. Don't ask trivial questions.

Output ONLY valid JSON array. Each item should have:
- kind: "missing" or "clarification"
- question: The question to ask (natural, conversational)
- context: Why this information would be useful
- related_entity: The entity this question is about (or null)
- priority: 1-10 (10 = most important)

If no important information is missing, output: []

Example:
[{"kind":"missing","question":"Who is responsible for Project A?","context":"Project A was mentioned but no responsible person is assigned","related_entity":"Project A","priority":7}]"""
        else:
            system_prompt = """あなたは不足情報の検出者です。与えられた事実と既存の知識を分析し、重要な情報の欠落を特定してください。

存在すべきだが不足している情報を考えてください：
- プロジェクトが言及されている場合：担当者、締め切り、ステータス、優先度はわかっているか？
- 人物が言及されている場合：役割、チーム、連絡先はわかっているか？
- イベントが言及されている場合：日時、場所、参加者、結果はわかっているか？
- 決定事項が言及されている場合：理由、検討した代替案、誰が決めたかはわかっているか？
- 問題が言及されている場合：影響範囲、根本原因、解決計画はわかっているか？

**本当に有用な**不足情報のみ質問してください。些末な質問は避けてください。

有効なJSON配列のみを出力してください。各項目には以下を含めます：
- kind: "missing" または "clarification"
- question: 質問文（自然な会話調で）
- context: この情報がなぜ有用か
- related_entity: 質問が関連するエンティティ（またはnull）
- priority: 1〜10（10が最も重要）

重要な不足情報がない場合は空の配列を出力: []

例:
[{"kind":"missing","question":"プロジェクトAの担当者は誰ですか？","context":"プロジェクトAが言及されていますが担当者が不明","related_entity":"プロジェクトA","priority":7}]"""

        user_prompt_parts = []
        if facts_text:
            user_prompt_parts.append(f"【新しい事実】\n{facts_text}")
        if knowledge_text:
            user_prompt_parts.append(f"【既存の知識】\n{knowledge_text}")
        if user_message:
            user_prompt_parts.append(f"【ユーザーメッセージ】\n{user_message}")
        user_prompt_parts.append(f"【ドメイン】\n{domain.value}")

        user_prompt = "\n\n".join(user_prompt_parts) + "\n\n上記の情報から、不足している重要な情報を特定してください。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import json
            from datetime import datetime

            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_MID)  # type: ignore
            content = response.content or "[]"

            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            items_data = json.loads(content)
            if not isinstance(items_data, list):
                return []

            questions: list[PendingQuestion] = []
            for item in items_data:
                try:
                    kind_str = item.get("kind", "missing")
                    kind = QuestionKind.MISSING if kind_str == "missing" else QuestionKind.CLARIFICATION

                    question = PendingQuestion(
                        id=str(uuid.uuid4()),
                        kind=kind,
                        question=item.get("question", ""),
                        context=item.get("context", ""),
                        related_fact_id=None,
                        related_entity=item.get("related_entity"),
                        priority=min(10, max(1, int(item.get("priority", 5)))),
                        status=QuestionStatus.PENDING,
                        session_id=session_id,
                        created_at=datetime.now(),
                    )
                    questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to parse missing info question: {e}")
                    continue

            if questions:
                logger.info(f"Detected {len(questions)} missing information questions")

            return questions

        except Exception as e:
            logger.warning(f"Failed to detect missing information: {e}")
            return []

    async def _analyze_answer_to_questions(
        self,
        user_message: str,
        pending_questions: list[PendingQuestion],
        language: str,
    ) -> list[dict]:
        """Analyze user's message to determine if it answers any pending questions.

        Uses LLM to match user responses to pending questions and determine
        what knowledge updates should be made.

        Args:
            user_message: The user's message.
            pending_questions: List of currently pending questions.
            language: Session language.

        Returns:
            List of dicts with keys:
                - question_id: ID of the answered question
                - answered: True if the question was answered
                - action: "accept_current" | "keep_previous" | "new_value" | "skip"
                - new_value: The new value from the user's answer (if any)
                - confidence: How confident we are in this analysis
        """
        if not pending_questions:
            return []

        # Build question list for LLM
        questions_text = "\n".join([
            f"- [ID:{q.id}] (種類:{q.kind.value}) {q.question}\n  背景: {q.context}"
            for q in pending_questions
        ])

        if language.lower() in ("english", "en"):
            system_prompt = """You are a response analyzer. Determine if the user's message answers any of the pending questions.

For each question that is addressed by the user's message, provide:
- question_id: The ID of the question being answered
- answered: true if the user provided an answer, false if unclear
- action: 
  - "accept_current" = user confirms the new/current information is correct
  - "keep_previous" = user says the old information was correct  
  - "new_value" = user provides a completely different value
  - "skip" = user explicitly skips/ignores the question
- new_value: The specific value from the user's answer (null if not applicable)
- confidence: 0.0-1.0 how confident you are

Output ONLY valid JSON array. Only include questions that are actually addressed. If no questions are answered, output: []

Example:
[{"question_id":"abc-123","answered":true,"action":"accept_current","new_value":null,"confidence":0.9}]"""
        else:
            system_prompt = """あなたは回答分析者です。ユーザーのメッセージが未回答の質問に答えているかを判定してください。

ユーザーのメッセージが対応している各質問について以下を提供してください：
- question_id: 回答されている質問のID
- answered: ユーザーが回答を提供した場合true、不明確な場合false
- action:
  - "accept_current" = ユーザーが新しい/現在の情報が正しいと確認
  - "keep_previous" = ユーザーが古い情報が正しいと回答
  - "new_value" = ユーザーがまったく異なる値を提供
  - "skip" = ユーザーが明示的に質問をスキップ/無視
- new_value: ユーザーの回答から得られた具体的な値（該当しない場合はnull）
- confidence: 0.0〜1.0で確信度

有効なJSON配列のみを出力してください。実際に対応されている質問のみ含めてください。
回答がない場合は空の配列を出力: []

例:
[{"question_id":"abc-123","answered":true,"action":"accept_current","new_value":null,"confidence":0.9}]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【未回答の質問】\n{questions_text}\n\n【ユーザーのメッセージ】\n{user_message}"},
        ]

        try:
            import json

            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_FAST)  # type: ignore
            content = response.content or "[]"

            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            results = json.loads(content)
            if not isinstance(results, list):
                return []

            # Validate question_ids
            valid_ids = {q.id for q in pending_questions}
            validated_results = []
            for r in results:
                if r.get("question_id") in valid_ids:
                    validated_results.append({
                        "question_id": r["question_id"],
                        "answered": bool(r.get("answered", False)),
                        "action": r.get("action", "accept_current"),
                        "new_value": r.get("new_value"),
                        "confidence": min(1.0, max(0.0, float(r.get("confidence", 0.7)))),
                    })

            if validated_results:
                logger.info(f"Analyzed {len(validated_results)} answers from user message")

            return validated_results

        except Exception as e:
            logger.warning(f"Failed to analyze answer to questions: {e}")
            return []

    async def _apply_answer_to_knowledge(
        self,
        answer_analysis: dict,
        pending_question: PendingQuestion,
        session_id: str,
    ) -> None:
        """Apply a user's answer to update the knowledge graph.

        Args:
            answer_analysis: Result from _analyze_answer_to_questions for one question.
            pending_question: The question that was answered.
            session_id: Session ID for tracking.
        """
        if not self._kg_store:
            return

        action = answer_analysis.get("action", "")
        new_value = answer_analysis.get("new_value")
        from datetime import datetime

        try:
            if pending_question.kind in (QuestionKind.CONTRADICTION, QuestionKind.CHANGE):
                # Handle contradiction/change resolution
                fact_id = pending_question.related_fact_id
                if fact_id:
                    if action == "accept_current":
                        # Update the fact to the new value
                        await self._kg_store.update_fact(
                            fact_id=fact_id,
                            new_value=new_value or pending_question.context.split("→")[-1].strip().strip("「」 "),
                            source=f"session:{session_id}",
                        )
                        await self._kg_store.set_fact_status(fact_id, "active")
                        logger.info(f"Updated fact {fact_id}: accepted current value")
                    elif action == "keep_previous":
                        # Keep existing fact, mark as active
                        await self._kg_store.set_fact_status(fact_id, "active")
                        logger.info(f"Kept previous value for fact {fact_id}")
                    elif action == "new_value" and new_value:
                        # Set a completely new value
                        await self._kg_store.update_fact(
                            fact_id=fact_id,
                            new_value=new_value,
                            source=f"session:{session_id}",
                        )
                        await self._kg_store.set_fact_status(fact_id, "active")
                        logger.info(f"Set new value for fact {fact_id}: {new_value}")

            elif pending_question.kind == QuestionKind.MISSING and new_value:
                # Save new insight for missing information
                from el_core.schemas import Insight
                insight = Insight(
                    subject=pending_question.related_entity or "unknown",
                    predicate="has_info",
                    object=new_value,
                    confidence=0.9,
                    domain=Domain.GENERAL,
                )
                insight_id = await self._kg_store.save_insight(insight, session_id=session_id)
                logger.info(f"Saved new insight from missing info answer: {new_value[:50]}")
                # Auto-tag the insight
                try:
                    content = f"{insight.subject} {insight.predicate} {insight.object}"
                    await self.auto_tag_insight(insight_id=insight_id, content=content, max_tags=2)
                except Exception as e:
                    logger.warning(f"Failed to auto-tag missing info insight: {e}")

            elif pending_question.kind == QuestionKind.CLARIFICATION and new_value:
                # Save clarification as insight
                from el_core.schemas import Insight
                insight = Insight(
                    subject=pending_question.related_entity or "unknown",
                    predicate="clarification",
                    object=new_value,
                    confidence=0.85,
                    domain=Domain.GENERAL,
                )
                insight_id = await self._kg_store.save_insight(insight, session_id=session_id)
                logger.info(f"Saved clarification: {new_value[:50]}")
                # Auto-tag the insight
                try:
                    content = f"{insight.subject} {insight.predicate} {insight.object}"
                    await self.auto_tag_insight(insight_id=insight_id, content=content, max_tags=2)
                except Exception as e:
                    logger.warning(f"Failed to auto-tag clarification insight: {e}")

        except Exception as e:
            logger.warning(f"Failed to apply answer to knowledge: {e}")

    async def _generate_all_questions(
        self,
        user_message: str,
        knowledge_used: list[KnowledgeItem],
        extracted_facts: list[ExtractedFact] | None,
        domain: Domain,
        language: str,
        session_id: str | None = None,
    ) -> tuple[list[PendingQuestion], list[ConsistencyIssue]]:
        """Generate all questions (contradictions, changes, missing info) in a unified manner.

        Combines consistency issue detection and missing information detection,
        returning a unified list of PendingQuestion sorted by priority.

        Args:
            user_message: Current user message.
            knowledge_used: Knowledge items retrieved for this turn.
            extracted_facts: Facts extracted from document or message (optional).
            domain: Detected domain.
            language: Session language.
            session_id: Session ID for persistence.

        Returns:
            Tuple of (all PendingQuestions sorted by priority, raw ConsistencyIssues).
        """
        all_questions: list[PendingQuestion] = []
        consistency_issues: list[ConsistencyIssue] = []

        # 1 + 2: Detect consistency issues AND missing information in parallel
        consistency_coro = (
            self._detect_consistency_issues(
                user_message=user_message,
                knowledge_used=knowledge_used,
                language=language,
                session_id=session_id,
            )
            if knowledge_used
            else None
        )

        missing_coro = self._detect_missing_information(
            extracted_facts=extracted_facts,
            user_message=user_message,
            existing_knowledge=knowledge_used,
            domain=domain,
            language=language,
            session_id=session_id,
        )

        if consistency_coro:
            consistency_issues, missing_questions = await asyncio.gather(
                consistency_coro, missing_coro
            )
        else:
            missing_questions = await missing_coro

        # Convert consistency issues to PendingQuestion
        for issue in consistency_issues:
            pq = self._consistency_issue_to_pending_question(issue, session_id)
            all_questions.append(pq)

        all_questions.extend(missing_questions)

        # Sort by priority (highest first)
        all_questions.sort(key=lambda q: q.priority, reverse=True)

        logger.info(
            f"Generated {len(all_questions)} total questions: "
            f"{len(consistency_issues)} consistency + {len(missing_questions)} missing"
        )

        return all_questions, consistency_issues

    # ==================== Document Processing ====================

    async def extract_from_document(
        self,
        content: str,
        filename: str,
        language: str = "Japanese",
    ) -> DocumentExtractionResult:
        """Extract structured information from document content using LLM.

        Args:
            content: The parsed text content of the document.
            filename: Original filename for context.
            language: Language for extraction prompts.

        Returns:
            DocumentExtractionResult with summary, facts, topics, entities.
        """
        # Note: Content should already be truncated by caller (process_document_background)
        # This is a safety limit for direct calls - GPT-5.2 can handle 128k+ tokens
        max_chars = 100000
        if len(content) > max_chars:
            # Keep start 20% + end 80% to prioritize recent content
            start_chars = int(max_chars * 0.2)
            end_chars = max_chars - start_chars - 100
            content = (
                content[:start_chars] 
                + f"\n\n[... 中間 約{len(content) - start_chars - end_chars:,}文字省略 ...]\n\n" 
                + content[-end_chars:]
            )

        if language.lower() in ("english", "en"):
            system_prompt = """You are a document analyzer. Extract key information from the document, paying special attention to temporal (date/time) information.

Output ONLY valid JSON with these fields:
- summary: 2-3 sentence summary of the document
- facts: Array of factual statements as objects with:
  - subject: The subject entity
  - predicate: The relationship/attribute
  - object: The value/content (the SUBSTANTIVE content, NOT dates)
  - source_context: Brief context from document (max 50 chars)
  - event_date: Date when this event occurred (YYYY-MM-DD format, null if unknown)
  - event_date_end: End date for date ranges (YYYY-MM-DD format, null if not a range)
  - date_type: One of "exact", "approximate", "range", "unknown"
- topics: Array of main topics/themes
- entities: Array of mentioned people, organizations, projects, places
- domain: One of "daily_work", "recipe", "postmortem", "creative", "general"

**CRITICAL RULES for dates:**
- Dates should ALWAYS go in the event_date field, NOT in subject/predicate/object.
- DO NOT create facts like "diary date is 1/4" or "entry date is Jan 14". The date is metadata, not content.
- For diary/journal entries: extract WHAT HAPPENED on that date, and put the date in event_date.
- Each date entry in a diary is an independent record. Different dates = different events, not updates.

**Date extraction rules:**
- Exact date (e.g., "May 1, 2024") → date_type: "exact"
- Approximate (e.g., "around May") → date_type: "approximate"
- Range (e.g., "May 1-15") → date_type: "range", set event_date_end
- No date context → date_type: "unknown"

Example for a diary:
{"summary":"Daily journal entries covering exercise routines and meals.","facts":[{"subject":"exercise routine","predicate":"included","object":"push-ups, planks, back bends, twists","source_context":"morning workout log","event_date":"2024-01-14","event_date_end":null,"date_type":"exact"},{"subject":"exercise routine","predicate":"included","object":"toe touches, splits, frog stretch, side planks","source_context":"morning workout log","event_date":"2024-01-04","event_date_end":null,"date_type":"exact"}],"topics":["Exercise","Health"],"entities":[],"domain":"daily_work"}"""
        else:
            system_prompt = """あなたはドキュメント分析者です。ドキュメントから重要な情報を抽出してください。特に日付・時間情報に注意を払ってください。

以下のフィールドを持つ有効なJSONのみを出力してください：
- summary: 2-3文のドキュメント要約
- facts: 事実の配列（各項目は以下の形式）
  - subject: 主語エンティティ
  - predicate: 関係・属性
  - object: 値・内容（**実質的な内容**を記述。日付はここに入れない）
  - source_context: ドキュメントからの簡潔な文脈（最大50文字）
  - event_date: イベントが発生した日付（YYYY-MM-DD形式、不明ならnull）
  - event_date_end: 期間の終了日（YYYY-MM-DD形式、範囲でなければnull）
  - date_type: "exact"（正確）, "approximate"（約）, "range"（期間）, "unknown"（不明）のいずれか
- topics: 主なトピック・テーマの配列
- entities: 言及された人物、組織、プロジェクト、場所の配列
- domain: "daily_work", "recipe", "postmortem", "creative", "general" のいずれか

**日付に関する重要なルール：**
- 日付は必ず event_date フィールドに入れてください。subject/predicate/object には入れないでください。
- 「日記は日付 1/4」「エントリの日付は1月14日」のような事実は絶対に作らないでください。日付はメタデータであり、事実の内容ではありません。
- 日記・日誌の場合：**その日に何があったか**を事実として抽出し、日付は event_date に入れてください。
- 日記の各日付エントリは独立した記録です。異なる日付 = 異なるイベントであり、同じ事実の更新ではありません。

**日付の抽出ルール：**
- 正確な日付がある場合（例：「2024年5月1日」）→ date_type: "exact"
- 曖昧な日付の場合（例：「5月頃」「2024年初め」）→ date_type: "approximate"
- 期間の場合（例：「5月1日〜15日」）→ date_type: "range"、event_date_endを設定
- 日付の文脈がない場合 → date_type: "unknown"

日記の例:
{"summary":"日々の運動メニューと食事の記録。","facts":[{"subject":"運動メニュー","predicate":"の内容","object":"反り・プランク前後・腕立て・捻り等","source_context":"朝の運動記録","event_date":"2024-01-14","event_date_end":null,"date_type":"exact"},{"subject":"運動メニュー","predicate":"の内容","object":"足先・前屈・開脚(左右)・カエル(前)・横プランク(左右)","source_context":"朝の運動記録","event_date":"2024-01-04","event_date_end":null,"date_type":"exact"},{"subject":"朝食","predicate":"のメニュー","object":"トースト、目玉焼き、サラダ","source_context":"食事の記録","event_date":"2024-01-14","event_date_end":null,"date_type":"exact"}],"topics":["運動","健康","食事"],"entities":[],"domain":"daily_work"}"""

        user_prompt = f"""ファイル名: {filename}

【ドキュメント内容】
{content}

上記のドキュメントを分析し、JSON形式で情報を抽出してください。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result_content = ""
        try:
            import json
            # Use higher max_tokens for extraction: reasoning models need budget
            # for both thinking tokens and output tokens
            response = await self._llm.chat(
                messages=messages,
                model=LLMClient.MODEL_MID,
                max_tokens=16384,
            )  # type: ignore
            
            logger.info(
                f"LLM extraction response for {filename}: "
                f"content_length={len(response.content) if response.content else 'None'}, "
                f"refusal={response.refusal if hasattr(response, 'refusal') else 'N/A'}"
            )
            
            result_content = response.content or "{}"

            # Clean up response
            result_content = result_content.strip()
            if result_content.startswith("```"):
                # Handle ```json or ``` prefix
                first_line_end = result_content.find("\n")
                if first_line_end > 0:
                    result_content = result_content[first_line_end + 1:]
                else:
                    result_content = result_content[3:]
            if result_content.endswith("```"):
                result_content = result_content.rsplit("```", 1)[0]
            result_content = result_content.strip()

            logger.debug(f"LLM extraction response for {filename} (first 500 chars): {result_content[:500]}")

            if not result_content or result_content == "{}":
                logger.warning(f"LLM returned empty response for document: {filename}")
                return DocumentExtractionResult(
                    summary=f"LLMが空のレスポンスを返しました: {filename}",
                    facts=[],
                    topics=[],
                    entities=[],
                    domain=Domain.GENERAL,
                )

            data = json.loads(result_content)

            # Validate that we got meaningful data
            raw_facts = data.get("facts", [])
            raw_summary = data.get("summary", "")

            if not raw_facts and not raw_summary:
                logger.warning(
                    f"LLM returned JSON with no facts and no summary for document: {filename}. "
                    f"Keys in response: {list(data.keys())}"
                )

            # Parse facts with date information
            facts = []
            skipped_facts = 0
            for f in raw_facts:
                if isinstance(f, dict) and "subject" in f and "predicate" in f and "object" in f:
                    # Parse event_date
                    event_date = None
                    if f.get("event_date"):
                        try:
                            from datetime import datetime
                            event_date = datetime.strptime(f["event_date"], "%Y-%m-%d")
                        except (ValueError, TypeError):
                            pass
                    
                    # Parse event_date_end
                    event_date_end = None
                    if f.get("event_date_end"):
                        try:
                            from datetime import datetime
                            event_date_end = datetime.strptime(f["event_date_end"], "%Y-%m-%d")
                        except (ValueError, TypeError):
                            pass
                    
                    # Parse date_type
                    date_type_str = f.get("date_type", "unknown")
                    try:
                        date_type = DateType(date_type_str)
                    except ValueError:
                        date_type = DateType.UNKNOWN
                    
                    facts.append(ExtractedFact(
                        subject=f["subject"],
                        predicate=f["predicate"],
                        object=f["object"],
                        source_context=f.get("source_context", ""),
                        event_date=event_date,
                        event_date_end=event_date_end,
                        date_type=date_type,
                    ))
                else:
                    skipped_facts += 1

            if skipped_facts > 0:
                logger.warning(
                    f"Skipped {skipped_facts} malformed facts from document: {filename}. "
                    f"Total valid: {len(facts)}"
                )

            # Parse domain
            domain_str = data.get("domain", "general")
            try:
                domain = Domain(domain_str)
            except ValueError:
                domain = Domain.GENERAL

            # Use meaningful summary, not the placeholder default
            summary = raw_summary if raw_summary else f"ドキュメント '{filename}' を分析しました。"

            result = DocumentExtractionResult(
                summary=summary,
                facts=facts,
                topics=data.get("topics", []),
                entities=data.get("entities", []),
                domain=domain,
            )

            logger.info(
                f"Extracted from document '{filename}': "
                f"{len(facts)} facts, {len(result.topics)} topics, "
                f"{len(result.entities)} entities, domain={domain.value}, "
                f"summary='{summary[:80]}...'"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM JSON response for document '{filename}': {e}. "
                f"Response content (first 300 chars): {result_content[:300]}"
            )
            return DocumentExtractionResult(
                summary=f"{filename} の解析でJSONパースエラーが発生しました",
                facts=[],
                topics=[],
                entities=[],
                domain=Domain.GENERAL,
            )
        except Exception as e:
            logger.error(f"Failed to extract from document '{filename}': {e}", exc_info=True)
            return DocumentExtractionResult(
                summary=f"{filename} の解析でエラーが発生しました",
                facts=[],
                topics=[],
                entities=[],
                domain=Domain.GENERAL,
            )

    async def create_document_review_session(
        self,
        user_id: str,
        document_id: str,
        document_filename: str,
        extracted_facts: list[ExtractedFact],
        language: str | None = None,
    ) -> tuple[str, str, list[ConsistencyIssue]]:
        """Create a session for reviewing uploaded document and checking consistency.
        
        ドキュメントアップロード後に、抽出された事実と過去の知識を照合し、
        矛盾点や不足情報について確認するセッションを作成します。
        
        Args:
            user_id: User identifier.
            document_id: Document ID that was uploaded.
            document_filename: Original filename.
            extracted_facts: Facts extracted from the document.
            language: Language preference. Auto-detected if None.
            
        Returns:
            Tuple of (session_id, opening_message, consistency_issues).
        """
        detected_lang = language or "Japanese"
        session_id = str(uuid.uuid4())
        
        # Create session with document review topic
        topic = f"ドキュメント「{document_filename}」の確認"
        session = Session(
            id=session_id,
            user_id=user_id,
            topic=topic,
            language=detected_lang,
        )
        
        # Check each extracted fact against existing knowledge
        # IMPORTANT: Exclude facts from the same document to avoid self-comparison
        consistency_issues: list[ConsistencyIssue] = []
        related_knowledge: list[KnowledgeItem] = []
        
        # Get IDs of facts from this document to exclude from comparison
        same_doc_fact_ids: set[str] = set()
        if self._kg_store:
            try:
                doc_insights = await self._kg_store.get_insights_for_document(document_id)
                same_doc_fact_ids = {i.id for i in doc_insights if i.id}
                logger.info(f"Excluding {len(same_doc_fact_ids)} facts from same document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to get document facts for exclusion: {e}")
        
        if self._kg_store and extracted_facts:
            for fact in extracted_facts:
                # Search for related knowledge
                search_query = f"{fact.subject} {fact.predicate}"
                related = await self._kg_store.search(
                    search_query,
                    limit=10,  # Fetch more since we'll filter some out
                )
                
                # Filter out facts from the same document
                related = [item for item in related if item.id not in same_doc_fact_ids]
                
                if not related:
                    continue
                
                related_knowledge.extend(related)
                
                # Check for consistency issues
                fact_text = f"{fact.subject}は{fact.predicate}「{fact.object}」"
                if fact.event_date:
                    fact_text += f"（{fact.event_date.strftime('%Y年%m月%d日')}の記録）"
                
                issues = await self._detect_consistency_issues(
                    user_message=fact_text,
                    knowledge_used=related,
                    language=detected_lang,
                    session_id=session_id,
                )
                consistency_issues.extend(issues)
        
        # Store related knowledge in session
        session.prior_knowledge = related_knowledge
        session.prior_context = self._format_prior_knowledge(related_knowledge, detected_lang)
        
        # Store document_id in session for context loading during respond()
        session.review_document_id = document_id
        self._sessions[session_id] = session
        
        # Save session metadata to Neo4j
        if self._kg_store:
            try:
                await self._save_session_metadata(session)
                # Link session to document
                await self._kg_store.link_session_to_document(session_id, document_id)
            except Exception as e:
                logger.warning(f"Failed to save session metadata: {e}")
        
        # Generate opening message with consistency issues
        opening = await self._generate_document_review_opening(
            session=session,
            document_filename=document_filename,
            extracted_facts=extracted_facts,
            consistency_issues=consistency_issues,
            language=detected_lang,
        )

        # Save the initial exchange as the first conversation turn
        # This ensures the document review context is preserved in conversation history
        initial_turn = ConversationTurn(
            user_message=topic,
            assistant_response=opening,
        )
        session.add_turn(initial_turn)

        # Persist the initial turn to Neo4j
        if self._kg_store:
            try:
                await self._kg_store.save_conversation_turn(
                    session_id=session_id,
                    turn_index=0,
                    user_message=topic,
                    assistant_response=opening,
                    timestamp=initial_turn.timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to save initial conversation turn for review session: {e}")
        
        logger.info(
            f"Created document review session {session_id} for document {document_id}: "
            f"{len(extracted_facts)} facts, {len(consistency_issues)} issues"
        )
        
        return session_id, opening, consistency_issues
    
    async def _generate_document_review_opening(
        self,
        session: Session,
        document_filename: str,
        extracted_facts: list[ExtractedFact],
        consistency_issues: list[ConsistencyIssue],
        language: str,
    ) -> str:
        """Generate opening message for document review session.
        
        Args:
            session: The session object.
            document_filename: Name of the uploaded document.
            extracted_facts: Facts extracted from the document.
            consistency_issues: Detected consistency issues.
            language: Session language.
            
        Returns:
            Opening message for the review session.
        """
        if language.lower() in ("english", "en"):
            base_message = (
                f"I've processed the document \"{document_filename}\" and extracted "
                f"{len(extracted_facts)} facts. "
            )
            
            if consistency_issues:
                base_message += (
                    f"I found {len(consistency_issues)} potential inconsistencies "
                    f"with your previous records. Let me ask you about these:\n\n"
                )
                for i, issue in enumerate(consistency_issues[:5], 1):  # Limit to 5
                    base_message += f"{i}. {issue.suggested_question}\n"
                if len(consistency_issues) > 5:
                    base_message += f"\n... and {len(consistency_issues) - 5} more issues.\n"
            else:
                base_message += (
                    "I've checked these facts against your previous records and "
                    "they appear consistent. Is there anything you'd like to clarify or add?"
                )
        else:
            base_message = (
                f"ドキュメント「{document_filename}」を処理し、"
                f"{len(extracted_facts)}件の事実を抽出しました。\n\n"
            )
            
            if consistency_issues:
                base_message += (
                    f"過去の記録と照合したところ、{len(consistency_issues)}件の"
                    f"矛盾点や変更点が見つかりました。確認させてください：\n\n"
                )
                for i, issue in enumerate(consistency_issues[:5], 1):  # 最大5件まで表示
                    base_message += f"{i}. {issue.suggested_question}\n"
                if len(consistency_issues) > 5:
                    base_message += f"\n... 他{len(consistency_issues) - 5}件の確認事項があります。\n"
            else:
                base_message += (
                    "過去の記録と照合しましたが、特に矛盾は見つかりませんでした。"
                    "追加で確認したいことや補足したい情報はありますか？"
                )
        
        return base_message

    async def generate_topic_summary(
        self,
        topic: str,
        facts: list[KnowledgeItem],
        language: str = "Japanese",
    ) -> TopicSummary:
        """Generate a comprehensive summary for a topic based on accumulated facts.

        Args:
            topic: The topic/entity name.
            facts: List of facts related to the topic.
            language: Language for the summary.

        Returns:
            TopicSummary with comprehensive overview.
        """
        if not facts:
            return TopicSummary(
                topic=topic,
                summary=f"{topic}に関する記録はまだありません。",
                key_points=[],
                related_entities=[],
                fact_count=0,
            )

        # Format facts for LLM
        facts_text = "\n".join([
            f"- {f.subject}は{f.predicate}「{f.object}」({f.created_at.strftime('%Y-%m-%d')})"
            for f in facts
        ])

        # Extract unique entities
        all_entities = set()
        for f in facts:
            all_entities.add(f.subject)
            all_entities.add(f.object)
        all_entities.discard(topic)

        # Get time range
        dates = [f.created_at for f in facts]
        time_range = f"{min(dates).strftime('%Y-%m-%d')} ～ {max(dates).strftime('%Y-%m-%d')}"

        if language.lower() in ("english", "en"):
            system_prompt = """You are a knowledge synthesizer. Create a comprehensive summary of the topic based on the facts provided.

Output ONLY valid JSON with these fields:
- summary: 3-5 sentence comprehensive overview
- key_points: Array of the most important points (max 7)
- insights: Array of insights or patterns you noticed
- questions: Array of follow-up questions that might be useful

Example:
{"summary":"Project A has been progressing steadily...","key_points":["Deadline is Feb 28","80% complete","Tanaka leading"],"insights":["Progress accelerated in January","Risk of deadline slip"],"questions":["What are the remaining tasks?","Any blockers?"]}"""
        else:
            system_prompt = """あなたはナレッジ統合者です。提供された事実に基づいて、トピックの包括的な要約を作成してください。

以下のフィールドを持つ有効なJSONのみを出力してください：
- summary: 3-5文の包括的な概要
- key_points: 最も重要なポイントの配列（最大7つ）
- insights: 気づいたパターンや洞察の配列
- questions: 有用と思われるフォローアップ質問の配列

例:
{"summary":"プロジェクトAは順調に進行しており...","key_points":["締め切りは2月28日","完了率80%","田中さんがリード"],"insights":["1月に進捗が加速","締め切り遅延のリスクあり"],"questions":["残りのタスクは何か？","ブロッカーはあるか？"]}"""

        user_prompt = f"""トピック: {topic}
関連する事実数: {len(facts)}
期間: {time_range}

【記録された事実】
{facts_text}

上記の事実を統合し、{topic}についての包括的な要約をJSON形式で作成してください。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import json
            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_MID)  # type: ignore
            result_content = response.content or "{}"

            # Clean up response
            result_content = result_content.strip()
            if result_content.startswith("```"):
                result_content = result_content.split("\n", 1)[-1]
            if result_content.endswith("```"):
                result_content = result_content.rsplit("```", 1)[0]
            result_content = result_content.strip()

            data = json.loads(result_content)

            summary = TopicSummary(
                topic=topic,
                summary=data.get("summary", f"{topic}についての要約"),
                key_points=data.get("key_points", []) + data.get("insights", []),
                related_entities=list(all_entities)[:10],
                fact_count=len(facts),
                time_range=time_range,
            )

            logger.info(f"Generated summary for topic: {topic}")
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate topic summary: {e}")
            return TopicSummary(
                topic=topic,
                summary=f"{topic}に関する{len(facts)}件の事実が記録されています。",
                key_points=[f.object for f in facts[:5]],
                related_entities=list(all_entities)[:10],
                fact_count=len(facts),
                time_range=time_range,
            )

    # ==================== Tag Suggestion Methods ====================

    async def suggest_tags(
        self,
        content: str,
        existing_tags: list[Tag] | None = None,
        max_tags: int = 5,
        language: str | None = None,
    ) -> TagSuggestionResult:
        """Suggest tags for given content using LLM.

        Args:
            content: Content to analyze for tags.
            existing_tags: List of existing tags to consider for reuse.
            max_tags: Maximum number of tags to suggest.
            language: Language for prompts (auto-detect if None).

        Returns:
            TagSuggestionResult with suggested tags.
        """
        import json

        if existing_tags is None:
            existing_tags = await self._kg_store.get_all_tags(limit=100)

        lang = language or detect_language(content)

        # Format existing tags for the prompt
        existing_tags_str = ""
        if existing_tags:
            tag_names = [f"- {t.name}" + (f" (別名: {', '.join(t.aliases[:3])})" if t.aliases else "") 
                        for t in existing_tags[:50]]
            existing_tags_str = "\n".join(tag_names)

        if lang == "Japanese":
            system_prompt = """あなたはコンテンツ分類の専門家です。
与えられたテキストを分析し、適切なタグを提案してください。

タグは以下の特徴を持つべきです：
- 短く簡潔（1〜3語程度）
- 内容の本質を捉えている
- 検索やグルーピングに役立つ
- 既存タグで適切なものがあれば優先的に再利用

JSONフォーマットで出力してください：
{
  "tags": [
    {"name": "タグ名", "relevance": 0.9, "reason": "このタグを選んだ理由", "is_existing": false}
  ],
  "content_summary": "コンテンツの簡潔な要約"
}

relevanceは0.0〜1.0で、コンテンツとの関連度を示します。
is_existingは既存タグリストから選んだ場合true、新規提案の場合falseにしてください。"""

            user_prompt = f"""以下のコンテンツに適切なタグを最大{max_tags}個提案してください。

【コンテンツ】
{content[:2000]}

【既存タグ一覧】
{existing_tags_str if existing_tags_str else "（まだタグがありません）"}

適切な既存タグがあれば優先的に使用し、必要であれば新しいタグを提案してください。"""

        else:
            system_prompt = """You are a content classification expert.
Analyze the given text and suggest appropriate tags.

Tags should be:
- Short and concise (1-3 words)
- Capture the essence of the content
- Useful for search and grouping
- Reuse existing tags when appropriate

Output in JSON format:
{
  "tags": [
    {"name": "tag name", "relevance": 0.9, "reason": "why this tag", "is_existing": false}
  ],
  "content_summary": "brief summary of content"
}

relevance is 0.0-1.0 indicating how related the tag is to the content.
is_existing should be true if selected from existing tags, false if newly suggested."""

            user_prompt = f"""Suggest up to {max_tags} appropriate tags for the following content.

【Content】
{content[:2000]}

【Existing Tags】
{existing_tags_str if existing_tags_str else "(No tags yet)"}

Prioritize existing tags when appropriate, and suggest new ones if needed."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_FAST)  # type: ignore
            result_content = response.content or "{}"

            # Clean up response
            result_content = result_content.strip()
            if result_content.startswith("```"):
                result_content = result_content.split("\n", 1)[-1]
            if result_content.endswith("```"):
                result_content = result_content.rsplit("```", 1)[0]
            result_content = result_content.strip()

            data = json.loads(result_content)

            suggestions: list[TagSuggestion] = []
            existing_matched = 0
            new_suggested = 0

            for tag_data in data.get("tags", [])[:max_tags]:
                tag_name = tag_data.get("name", "").strip()
                if not tag_name:
                    continue

                is_existing = tag_data.get("is_existing", False)
                existing_tag_id = None

                # Check if this matches an existing tag
                if existing_tags:
                    for existing_tag in existing_tags:
                        if (existing_tag.name.lower() == tag_name.lower() or
                            any(alias.lower() == tag_name.lower() for alias in existing_tag.aliases)):
                            existing_tag_id = existing_tag.id
                            is_existing = True
                            tag_name = existing_tag.name  # Use canonical name
                            break

                if is_existing:
                    existing_matched += 1
                else:
                    new_suggested += 1

                suggestions.append(TagSuggestion(
                    name=tag_name,
                    relevance=min(1.0, max(0.0, tag_data.get("relevance", 0.8))),
                    reason=tag_data.get("reason", ""),
                    existing_tag_id=existing_tag_id,
                    is_new=not is_existing,
                ))

            return TagSuggestionResult(
                suggestions=suggestions,
                content_summary=data.get("content_summary", ""),
                existing_tags_matched=existing_matched,
                new_tags_suggested=new_suggested,
            )

        except Exception as e:
            logger.warning(f"Failed to suggest tags: {e}")
            return TagSuggestionResult(
                suggestions=[],
                content_summary="",
                existing_tags_matched=0,
                new_tags_suggested=0,
            )

    async def check_similar_tags(
        self,
        new_tag_name: str,
        existing_tags: list[Tag] | None = None,
        language: str | None = None,
    ) -> tuple[str | None, float]:
        """Check if a new tag is similar to existing tags.

        Args:
            new_tag_name: Name of the potential new tag.
            existing_tags: List of existing tags to compare.
            language: Language for prompts (auto-detect if None).

        Returns:
            Tuple of (existing_tag_id_to_merge_into, similarity_score).
            Returns (None, 0.0) if no similar tag found.
        """
        import json

        if existing_tags is None:
            existing_tags = await self._kg_store.get_all_tags(limit=100)

        if not existing_tags:
            return None, 0.0

        lang = language or detect_language(new_tag_name)

        tag_names_list = [f"{t.name} (ID: {t.id})" for t in existing_tags[:50]]
        tag_names_str = "\n".join(tag_names_list)

        if lang == "Japanese":
            system_prompt = """あなたはタグの類似度判定の専門家です。
新しいタグが既存タグと意味的に同じか判定してください。

JSONフォーマットで出力：
{
  "is_similar": true/false,
  "similar_tag_id": "類似タグのID（なければnull）",
  "similarity_score": 0.9,
  "reason": "判定理由"
}

similarity_scoreは0.0〜1.0で、0.8以上を「同義語」とみなします。
例: 「旅行」と「トラベル」、「プログラミング」と「コーディング」は同義語です。"""

            user_prompt = f"""新規タグ候補: 「{new_tag_name}」

【既存タグ一覧】
{tag_names_str}

この新規タグは既存タグのいずれかと同義ですか？"""

        else:
            system_prompt = """You are an expert in tag similarity assessment.
Determine if a new tag is semantically the same as existing tags.

Output in JSON format:
{
  "is_similar": true/false,
  "similar_tag_id": "ID of similar tag or null",
  "similarity_score": 0.9,
  "reason": "reason for judgment"
}

similarity_score is 0.0-1.0, with 0.8+ considered synonyms.
Example: "travel" and "trip", "programming" and "coding" are synonyms."""

            user_prompt = f"""New tag candidate: "{new_tag_name}"

【Existing Tags】
{tag_names_str}

Is this new tag synonymous with any existing tag?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self._llm.chat(messages=messages, model=LLMClient.MODEL_FAST)  # type: ignore
            result_content = response.content or "{}"

            # Clean up response
            result_content = result_content.strip()
            if result_content.startswith("```"):
                result_content = result_content.split("\n", 1)[-1]
            if result_content.endswith("```"):
                result_content = result_content.rsplit("```", 1)[0]
            result_content = result_content.strip()

            data = json.loads(result_content)

            is_similar = data.get("is_similar", False)
            similar_tag_id = data.get("similar_tag_id")
            similarity_score = data.get("similarity_score", 0.0)

            if is_similar and similar_tag_id and similarity_score >= 0.8:
                # Verify the tag ID exists
                for tag in existing_tags:
                    if tag.id == similar_tag_id:
                        logger.info(f"Tag '{new_tag_name}' is similar to '{tag.name}' (score: {similarity_score})")
                        return similar_tag_id, similarity_score
                # Tag ID not found, try to find by name match in reason
                logger.warning(f"Similar tag ID '{similar_tag_id}' not found in existing tags")

            return None, 0.0

        except Exception as e:
            logger.warning(f"Failed to check tag similarity: {e}")
            return None, 0.0

    async def auto_tag_content(
        self,
        content: str,
        max_tags: int = 5,
        min_relevance: float = 0.6,
        language: str | None = None,
    ) -> list[tuple[Tag, float]]:
        """Automatically tag content by suggesting, checking similarity, and creating tags.

        Args:
            content: Content to tag.
            max_tags: Maximum number of tags.
            min_relevance: Minimum relevance score to include.
            language: Language for prompts.

        Returns:
            List of (Tag, relevance) tuples for the created/matched tags.
        """
        # Get existing tags
        existing_tags = await self._kg_store.get_all_tags(limit=100)

        # Get tag suggestions
        result = await self.suggest_tags(
            content=content,
            existing_tags=existing_tags,
            max_tags=max_tags,
            language=language,
        )

        tags_with_relevance: list[tuple[Tag, float]] = []

        for suggestion in result.suggestions:
            if suggestion.relevance < min_relevance:
                continue

            if suggestion.existing_tag_id:
                # Use existing tag
                tag = await self._kg_store.get_tag(suggestion.existing_tag_id)
                if tag:
                    tags_with_relevance.append((tag, suggestion.relevance))
            else:
                # Check for similar existing tags
                similar_tag_id, similarity = await self.check_similar_tags(
                    new_tag_name=suggestion.name,
                    existing_tags=existing_tags,
                    language=language,
                )

                if similar_tag_id and similarity >= 0.8:
                    # Use similar existing tag
                    tag = await self._kg_store.get_tag(similar_tag_id)
                    if tag:
                        # Add the suggested name as an alias if not already there
                        if (suggestion.name.lower() != tag.name.lower() and
                            suggestion.name.lower() not in [a.lower() for a in tag.aliases]):
                            await self._kg_store.update_tag(
                                tag.id,
                                aliases=tag.aliases + [suggestion.name],
                            )
                            tag = await self._kg_store.get_tag(similar_tag_id)  # Refresh
                        if tag:
                            tags_with_relevance.append((tag, suggestion.relevance))
                else:
                    # Create new tag
                    tag = await self._kg_store.create_tag(name=suggestion.name)
                    tags_with_relevance.append((tag, suggestion.relevance))
                    # Add to existing_tags for subsequent similarity checks
                    existing_tags.append(tag)

        return tags_with_relevance

    async def auto_tag_insight(
        self,
        insight_id: str,
        content: str,
        max_tags: int = 3,
        language: str | None = None,
    ) -> list[tuple[Tag, float]]:
        """Automatically tag an insight.

        Args:
            insight_id: ID of the insight to tag.
            content: Content to analyze for tags (usually subject + predicate + object).
            max_tags: Maximum number of tags.
            language: Language for prompts.

        Returns:
            List of (Tag, relevance) tuples applied to the insight.
        """
        tags_with_relevance = await self.auto_tag_content(
            content=content,
            max_tags=max_tags,
            language=language,
        )

        for tag, relevance in tags_with_relevance:
            await self._kg_store.tag_insight(insight_id, tag.id, relevance)

        logger.info(f"Auto-tagged insight {insight_id} with {len(tags_with_relevance)} tags")
        return tags_with_relevance

    async def auto_tag_document(
        self,
        document_id: str,
        content: str,
        max_tags: int = 5,
        language: str | None = None,
    ) -> list[tuple[Tag, float]]:
        """Automatically tag a document.

        Args:
            document_id: ID of the document to tag.
            content: Content to analyze for tags (usually summary + extracted facts).
            max_tags: Maximum number of tags.
            language: Language for prompts.

        Returns:
            List of (Tag, relevance) tuples applied to the document.
        """
        tags_with_relevance = await self.auto_tag_content(
            content=content,
            max_tags=max_tags,
            language=language,
        )

        for tag, relevance in tags_with_relevance:
            await self._kg_store.tag_document(document_id, tag.id, relevance)

        logger.info(f"Auto-tagged document {document_id} with {len(tags_with_relevance)} tags")
        return tags_with_relevance

    async def auto_tag_insights_batch(
        self,
        insights: list[dict[str, str]],
        max_tags_per_insight: int = 2,
        language: str | None = None,
        existing_tags: list[str] | None = None,
    ) -> dict[str, list[tuple[Tag, float]]]:
        """Automatically tag multiple insights in a single LLM call.

        Args:
            insights: List of dicts with 'id', 'subject', 'predicate', 'object' keys.
            max_tags_per_insight: Maximum tags per insight.
            language: Language for prompts.
            existing_tags: List of existing tag names to prefer for consistency.

        Returns:
            Dict mapping insight_id to list of (Tag, relevance) tuples.
        """
        if not insights:
            return {}

        lang = language or "ja"  # Default to Japanese

        # Build batch prompt
        insights_text = "\n".join([
            f"[{i+1}] {ins['subject']} {ins['predicate']} {ins['object']}"
            for i, ins in enumerate(insights)
        ])

        # Get existing tags for consistency
        existing_tags_text = ""
        if existing_tags:
            existing_tags_text = f"\n既存タグ（優先的に使用）: {', '.join(existing_tags[:30])}\n"

        prompt = f"""以下のファクトに対して、**抽象的なカテゴリタグ**を付与してください。

ファクト一覧:
{insights_text}
{existing_tags_text}
## タグ付けのルール

1. **抽象度を上げる**: 具体的すぎるタグは避ける
   - ❌ 悪い例: 「経過5日」「経過7日」「左かかと」「右かかと」
   - ✅ 良い例: 「経過観察」「かかと」「足」

2. **カテゴリとして機能するタグ**: 複数のファクトをグループ化できるもの
   - ❌ 悪い例: 「ストレッチ未実施」「ストレッチ内容」「ストレッチメニュー」
   - ✅ 良い例: 「ストレッチ」「運動」「トレーニング」

3. **タグの種類**:
   - テーマ: 健康, 仕事, 趣味, 家計, 日常
   - 活動: 運動, 食事, 通院, 勉強
   - 状態: 体調, 気分, 進捗
   - 身体部位: 腰, 足, 肩 （左右は区別しない）

4. **既存タグを優先**: 類似の既存タグがあれば新規作成せず再利用

## 回答形式（JSON配列）:
[
  {{"index": 1, "tags": [{{"name": "タグ名", "relevance": 0.9}}]}},
  ...
]

関連度(relevance)は0.7-1.0の範囲で、そのタグがファクトをどれだけ代表するかを示す。"""

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=LLMClient.MODEL_FAST,
            )

            import json
            import re

            # Parse response
            content = response.content or ""
            content = content.strip()
            # Extract JSON array from response
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                logger.warning(f"Failed to find JSON array in LLM response")
                return {}

            results = json.loads(match.group())
            
            # Process results
            tagged_insights: dict[str, list[tuple[Tag, float]]] = {}
            
            for result in results:
                idx = result.get("index", 0) - 1
                if idx < 0 or idx >= len(insights):
                    continue
                    
                insight_id = insights[idx]["id"]
                insight_tags: list[tuple[Tag, float]] = []
                
                for tag_info in result.get("tags", [])[:max_tags_per_insight]:
                    tag_name = tag_info.get("name", "").strip()
                    relevance = min(1.0, max(0.5, float(tag_info.get("relevance", 0.8))))
                    
                    if not tag_name:
                        continue
                    
                    try:
                        # Get or create tag
                        tag = await self._kg_store.get_or_create_tag(tag_name)
                        
                        # Check for similar tags and merge if needed
                        try:
                            similar = await self._kg_store.find_similar_tags(tag_name, threshold=0.85, limit=1)
                            if similar and similar[0][0].id != tag.id:
                                tag = similar[0][0]
                        except Exception:
                            pass  # Use the created tag if similarity check fails
                        
                        # Tag the insight
                        await self._kg_store.tag_insight(insight_id, tag.id, relevance)
                        insight_tags.append((tag, relevance))
                    except Exception as tag_err:
                        logger.warning(f"Failed to apply tag {tag_name}: {tag_err}")
                
                if insight_tags:
                    tagged_insights[insight_id] = insight_tags
            
            total_tags = sum(len(tags) for tags in tagged_insights.values())
            logger.info(f"Batch-tagged {len(tagged_insights)} insights with {total_tags} total tags")
            return tagged_insights

        except Exception as e:
            logger.error(f"Failed to batch tag insights: {e}")
            return {}
