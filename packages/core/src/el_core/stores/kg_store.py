"""Knowledge Graph Store using Neo4j."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from el_core.schemas import (
    ConsistencyIssue,
    ConsistencyIssueKind,
    ConsistencyIssueStatus,
    DateType,
    Document,
    DocumentChunk,
    DocumentStatus,
    Domain,
    FactStatus,
    FactVersion,
    FactWithHistory,
    Insight,
    KnowledgeGraphData,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    KnowledgeItem,
    KnowledgeStats,
    PendingQuestion,
    QuestionKind,
    QuestionStatus,
    SessionMetadata,
    SessionSummary,
    Tag,
    TaggedItem,
    TaggedItemType,
    TagStats,
    TopicStats,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """Async Knowledge Graph store backed by Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize the KG store.

        Args:
            uri: Neo4j URI. Defaults to NEO4J_URI env var.
            user: Neo4j user. Defaults to NEO4J_USER env var.
            password: Neo4j password. Defaults to NEO4J_PASSWORD env var.
        """
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Connect to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self._uri}")

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        return self._driver

    async def save_insight(self, insight: Insight, session_id: str | None = None) -> str:
        """Save an insight to the knowledge graph.

        Args:
            insight: The insight to save.
            session_id: Optional session ID to associate with this insight.

        Returns:
            The ID of the created insight node.
        """
        insight_id = str(uuid.uuid4())

        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        CREATE (i:Insight {
            id: $id,
            session_id: $session_id,
            predicate: $predicate,
            confidence: $confidence,
            domain: $domain,
            created_at: $created_at,
            event_date: $event_date,
            event_date_end: $event_date_end,
            date_type: $date_type
        })
        CREATE (s)-[:SUBJECT_OF]->(i)
        CREATE (i)-[:HAS_OBJECT]->(o)
        CREATE (s)-[r:RELATES_TO {
            type: $predicate,
            confidence: $confidence,
            insight_id: $id
        }]->(o)
        RETURN i.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                id=insight_id,
                session_id=session_id or "",
                subject=insight.subject,
                predicate=insight.predicate,
                object=insight.object,
                confidence=insight.confidence,
                domain=insight.domain.value,
                created_at=insight.timestamp.isoformat(),
                event_date=insight.event_date.isoformat() if insight.event_date else None,
                event_date_end=insight.event_date_end.isoformat() if insight.event_date_end else None,
                date_type=insight.date_type.value,
            )
            record = await result.single()
            return record["id"] if record else insight_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        domain: Domain | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[KnowledgeItem]:
        """Search the knowledge graph for relevant insights.

        Combines full-text index search (on predicate) with Entity name matching
        (on subject/object) to find all relevant facts.

        Args:
            query: Search query (keywords or concepts).
            limit: Maximum number of results.
            domain: Optional domain filter.
            start_date: Optional start date filter (filters by event_date).
            end_date: Optional end date filter (filters by event_date).

        Returns:
            List of matching knowledge items.
        """
        items: list[KnowledgeItem] = []
        seen_ids: set[str] = set()

        # 1. Full-text search on predicate field
        try:
            ft_cypher = """
            CALL db.index.fulltext.queryNodes('insight_search', $search_term)
            YIELD node, score
            WHERE node:Insight
            """
            if domain:
                ft_cypher += " AND node.domain = $domain_filter"
            if start_date:
                ft_cypher += " AND (node.event_date IS NULL OR node.event_date >= $start_date)"
            if end_date:
                ft_cypher += " AND (node.event_date IS NULL OR node.event_date <= $end_date)"

            ft_cypher += """
            MATCH (s:Entity)-[:SUBJECT_OF]->(node)-[:HAS_OBJECT]->(o:Entity)
            RETURN 
                node.id AS id, s.name AS subject, node.predicate AS predicate,
                o.name AS object, node.confidence AS confidence, node.domain AS domain,
                node.created_at AS created_at, node.event_date AS event_date,
                node.event_date_end AS event_date_end, node.date_type AS date_type,
                score
            ORDER BY score DESC
            LIMIT $max_results
            """

            params: dict[str, Any] = {"search_term": query, "max_results": limit}
            if domain:
                params["domain_filter"] = domain.value
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()

            async with self.driver.session() as session:
                result = await session.run(ft_cypher, **params)
                async for record in result:
                    item_id = record["id"]
                    if item_id not in seen_ids:
                        seen_ids.add(item_id)
                        items.append(
                            KnowledgeItem(
                                id=item_id,
                                subject=record["subject"],
                                predicate=record["predicate"],
                                object=record["object"],
                                confidence=record["confidence"],
                                domain=Domain(record["domain"]),
                                created_at=datetime.fromisoformat(record["created_at"]),
                                event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                                event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                                date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                            )
                        )
        except Exception as e:
            logger.warning(f"Full-text search failed: {e}")

        # 2. Entity-based search: find facts where subject or object name matches
        #    This catches queries like "ON01について" or "岡谷システムの会議室" etc.
        if len(items) < limit:
            try:
                entity_items = await self._basic_search(
                    query, limit - len(items), domain, start_date, end_date
                )
                for item in entity_items:
                    if item.id not in seen_ids:
                        seen_ids.add(item.id)
                        items.append(item)
            except Exception as e:
                logger.warning(f"Entity-based search failed: {e}")

        if items:
            logger.info(f"Knowledge search for '{query[:50]}': found {len(items)} facts")

        return items

    async def _basic_search(
        self,
        query: str,
        limit: int,
        domain: Domain | None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[KnowledgeItem]:
        """Basic search without full-text index."""
        cypher = """
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight)-[:HAS_OBJECT]->(o:Entity)
        WHERE s.name CONTAINS $search_term 
           OR o.name CONTAINS $search_term 
           OR i.predicate CONTAINS $search_term
        """

        if domain:
            cypher += " AND i.domain = $domain_filter"
        
        if start_date:
            cypher += " AND (i.event_date IS NULL OR i.event_date >= $start_date)"
        
        if end_date:
            cypher += " AND (i.event_date IS NULL OR i.event_date <= $end_date)"

        cypher += """
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.created_at AS created_at,
            i.event_date AS event_date,
            i.event_date_end AS event_date_end,
            i.date_type AS date_type
        ORDER BY i.confidence DESC
        LIMIT $max_results
        """

        params: dict[str, Any] = {"search_term": query, "max_results": limit}
        if domain:
            params["domain_filter"] = domain.value
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            async for record in result:
                items.append(
                    KnowledgeItem(
                        id=record["id"],
                        subject=record["subject"],
                        predicate=record["predicate"],
                        object=record["object"],
                        confidence=record["confidence"],
                        domain=Domain(record["domain"]),
                        created_at=datetime.fromisoformat(record["created_at"]),
                        event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                        event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                        date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                    )
                )

        return items

    async def get_related_insights(
        self,
        entity: str,
        limit: int = 10,
    ) -> list[KnowledgeItem]:
        """Get insights related to a specific entity.

        Args:
            entity: Entity name to find related insights for.
            limit: Maximum number of results.

        Returns:
            List of related knowledge items.
        """
        cypher = """
        MATCH (e:Entity {name: $entity})
        OPTIONAL MATCH (e)-[:SUBJECT_OF]->(i1:Insight)-[:HAS_OBJECT]->(o1:Entity)
        OPTIONAL MATCH (s2:Entity)-[:SUBJECT_OF]->(i2:Insight)-[:HAS_OBJECT]->(e)
        WITH collect({
            id: i1.id,
            subject: e.name,
            predicate: i1.predicate,
            object: o1.name,
            confidence: i1.confidence,
            domain: i1.domain,
            created_at: i1.created_at,
            event_date: i1.event_date,
            event_date_end: i1.event_date_end,
            date_type: i1.date_type
        }) + collect({
            id: i2.id,
            subject: s2.name,
            predicate: i2.predicate,
            object: e.name,
            confidence: i2.confidence,
            domain: i2.domain,
            created_at: i2.created_at,
            event_date: i2.event_date,
            event_date_end: i2.event_date_end,
            date_type: i2.date_type
        }) AS all_insights
        UNWIND all_insights AS insight
        WHERE insight.id IS NOT NULL
        RETURN DISTINCT
            insight.id AS id,
            insight.subject AS subject,
            insight.predicate AS predicate,
            insight.object AS object,
            insight.confidence AS confidence,
            insight.domain AS domain,
            insight.created_at AS created_at,
            insight.event_date AS event_date,
            insight.event_date_end AS event_date_end,
            insight.date_type AS date_type
        LIMIT $limit
        """

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, entity=entity, limit=limit)
            async for record in result:
                if record["id"]:
                    items.append(
                        KnowledgeItem(
                            id=record["id"],
                            subject=record["subject"],
                            predicate=record["predicate"],
                            object=record["object"],
                            confidence=record["confidence"],
                            domain=Domain(record["domain"]),
                            created_at=datetime.fromisoformat(record["created_at"]),
                            event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                            event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                            date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                        )
                    )

        return items

    # Session persistence methods

    async def save_session_metadata(
        self,
        session_id: str,
        user_id: str,
        topic: str,
        domain: Domain,
        turn_count: int,
        insights_count: int,
        created_at: datetime,
        updated_at: datetime,
    ) -> str:
        """Save or update session metadata to Neo4j.

        Args:
            session_id: Session ID.
            user_id: User ID.
            topic: Conversation topic.
            domain: Detected domain.
            turn_count: Number of conversation turns.
            insights_count: Number of insights saved.
            created_at: Session creation time.
            updated_at: Session last update time.

        Returns:
            The session ID.
        """
        cypher = """
        MERGE (s:Session {id: $session_id})
        SET s.user_id = $user_id,
            s.topic = $topic,
            s.domain = $domain,
            s.turn_count = $turn_count,
            s.insights_count = $insights_count,
            s.created_at = $created_at,
            s.updated_at = $updated_at
        RETURN s.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                session_id=session_id,
                user_id=user_id,
                topic=topic,
                domain=domain.value,
                turn_count=turn_count,
                insights_count=insights_count,
                created_at=created_at.isoformat(),
                updated_at=updated_at.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved session metadata: {session_id}")
            return record["id"] if record else session_id

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[SessionMetadata]:
        """Get recent sessions for a user.

        Args:
            user_id: User ID.
            limit: Maximum number of sessions to return.

        Returns:
            List of session metadata, ordered by updated_at descending.
        """
        cypher = """
        MATCH (s:Session {user_id: $user_id})
        RETURN 
            s.id AS id,
            s.user_id AS user_id,
            s.topic AS topic,
            s.domain AS domain,
            s.turn_count AS turn_count,
            s.insights_count AS insights_count,
            s.created_at AS created_at,
            s.updated_at AS updated_at
        ORDER BY s.updated_at DESC
        LIMIT $limit
        """

        sessions: list[SessionMetadata] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, user_id=user_id, limit=limit)
            async for record in result:
                sessions.append(
                    SessionMetadata(
                        id=record["id"],
                        user_id=record["user_id"],
                        topic=record["topic"],
                        domain=Domain(record["domain"]),
                        turn_count=record["turn_count"] or 0,
                        insights_count=record["insights_count"] or 0,
                        created_at=datetime.fromisoformat(record["created_at"]),
                        updated_at=datetime.fromisoformat(record["updated_at"]),
                    )
                )

        return sessions

    async def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        """Get session metadata by ID.

        Args:
            session_id: Session ID.

        Returns:
            Session metadata or None if not found.
        """
        cypher = """
        MATCH (s:Session {id: $session_id})
        RETURN 
            s.id AS id,
            s.user_id AS user_id,
            s.topic AS topic,
            s.domain AS domain,
            s.turn_count AS turn_count,
            s.insights_count AS insights_count,
            s.created_at AS created_at,
            s.updated_at AS updated_at
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, session_id=session_id)
            record = await result.single()
            if record:
                return SessionMetadata(
                    id=record["id"],
                    user_id=record["user_id"],
                    topic=record["topic"],
                    domain=Domain(record["domain"]),
                    turn_count=record["turn_count"] or 0,
                    insights_count=record["insights_count"] or 0,
                    created_at=datetime.fromisoformat(record["created_at"]),
                    updated_at=datetime.fromisoformat(record["updated_at"]),
                )
            return None

    async def get_session_insights(self, session_id: str) -> list[KnowledgeItem]:
        """Get all insights saved during a session.

        Args:
            session_id: Session ID.

        Returns:
            List of knowledge items saved in the session.
        """
        cypher = """
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight {session_id: $session_id})-[:HAS_OBJECT]->(o:Entity)
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.created_at AS created_at,
            i.event_date AS event_date,
            i.event_date_end AS event_date_end,
            i.date_type AS date_type
        ORDER BY i.created_at ASC
        """

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, session_id=session_id)
            async for record in result:
                items.append(
                    KnowledgeItem(
                        id=record["id"],
                        subject=record["subject"],
                        predicate=record["predicate"],
                        object=record["object"],
                        confidence=record["confidence"],
                        domain=Domain(record["domain"]),
                        created_at=datetime.fromisoformat(record["created_at"]),
                        event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                        event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                        date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                    )
                )

        return items

    # Session Summary methods

    async def save_session_summary(
        self,
        summary: SessionSummary,
    ) -> str:
        """Save a session summary to Neo4j.

        Args:
            summary: The session summary to save.

        Returns:
            The summary ID.
        """
        cypher = """
        MATCH (s:Session {id: $session_id})
        CREATE (sum:Summary {
            id: $id,
            content: $content,
            key_points: $key_points,
            topics: $topics,
            entities_mentioned: $entities_mentioned,
            turn_start: $turn_start,
            turn_end: $turn_end,
            created_at: $created_at
        })
        CREATE (s)-[:HAS_SUMMARY]->(sum)
        SET s.status = 'ended'
        RETURN sum.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                session_id=summary.session_id,
                id=summary.id,
                content=summary.content,
                key_points=summary.key_points,
                topics=summary.topics,
                entities_mentioned=summary.entities_mentioned,
                turn_start=summary.turn_range[0],
                turn_end=summary.turn_range[1],
                created_at=summary.created_at.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved session summary: {summary.id}")
            return record["id"] if record else summary.id

    async def get_session_summary(self, session_id: str) -> SessionSummary | None:
        """Get the summary for a session.

        Args:
            session_id: Session ID.

        Returns:
            Session summary or None if not found.
        """
        cypher = """
        MATCH (s:Session {id: $session_id})-[:HAS_SUMMARY]->(sum:Summary)
        RETURN 
            sum.id AS id,
            s.id AS session_id,
            sum.content AS content,
            sum.key_points AS key_points,
            sum.topics AS topics,
            sum.entities_mentioned AS entities_mentioned,
            sum.turn_start AS turn_start,
            sum.turn_end AS turn_end,
            sum.created_at AS created_at
        ORDER BY sum.created_at DESC
        LIMIT 1
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, session_id=session_id)
            record = await result.single()
            if record:
                return SessionSummary(
                    id=record["id"],
                    session_id=record["session_id"],
                    content=record["content"],
                    key_points=record["key_points"] or [],
                    topics=record["topics"] or [],
                    entities_mentioned=record["entities_mentioned"] or [],
                    turn_range=(record["turn_start"] or 0, record["turn_end"] or 0),
                    created_at=datetime.fromisoformat(record["created_at"]),
                )
            return None

    async def get_user_session_summaries(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[SessionSummary]:
        """Get recent session summaries for a user.

        Args:
            user_id: User ID.
            limit: Maximum number of summaries to return.

        Returns:
            List of session summaries, ordered by created_at descending.
        """
        cypher = """
        MATCH (s:Session {user_id: $user_id})-[:HAS_SUMMARY]->(sum:Summary)
        RETURN 
            sum.id AS id,
            s.id AS session_id,
            sum.content AS content,
            sum.key_points AS key_points,
            sum.topics AS topics,
            sum.entities_mentioned AS entities_mentioned,
            sum.turn_start AS turn_start,
            sum.turn_end AS turn_end,
            sum.created_at AS created_at
        ORDER BY sum.created_at DESC
        LIMIT $limit
        """

        summaries: list[SessionSummary] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, user_id=user_id, limit=limit)
            async for record in result:
                summaries.append(
                    SessionSummary(
                        id=record["id"],
                        session_id=record["session_id"],
                        content=record["content"],
                        key_points=record["key_points"] or [],
                        topics=record["topics"] or [],
                        entities_mentioned=record["entities_mentioned"] or [],
                        turn_range=(record["turn_start"] or 0, record["turn_end"] or 0),
                        created_at=datetime.fromisoformat(record["created_at"]),
                    )
                )

        return summaries

    async def update_session_status(self, session_id: str, status: str) -> None:
        """Update session status.

        Args:
            session_id: Session ID.
            status: New status (active, ended, archived).
        """
        cypher = """
        MATCH (s:Session {id: $session_id})
        SET s.status = $status, s.updated_at = $updated_at
        """

        async with self.driver.session() as session:
            await session.run(
                cypher,
                session_id=session_id,
                status=status,
                updated_at=datetime.now().isoformat(),
            )
            logger.info(f"Updated session {session_id} status to {status}")

    # Conversation Turn persistence methods

    async def save_conversation_turn(
        self,
        session_id: str,
        turn_index: int,
        user_message: str,
        assistant_response: str,
        timestamp: datetime | None = None,
    ) -> str:
        """Save a conversation turn to Neo4j.

        Args:
            session_id: Session ID.
            turn_index: Index of this turn (0-based).
            user_message: User's message.
            assistant_response: Assistant's response.
            timestamp: When the turn occurred.

        Returns:
            The turn ID.
        """
        turn_id = str(uuid.uuid4())
        now = timestamp or datetime.now()

        cypher = """
        MATCH (s:Session {id: $session_id})
        CREATE (t:ConversationTurn {
            id: $turn_id,
            session_id: $session_id,
            turn_index: $turn_index,
            user_message: $user_message,
            assistant_response: $assistant_response,
            timestamp: $timestamp
        })
        CREATE (s)-[:HAS_TURN]->(t)
        RETURN t.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                turn_id=turn_id,
                session_id=session_id,
                turn_index=turn_index,
                user_message=user_message,
                assistant_response=assistant_response,
                timestamp=now.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved conversation turn {turn_index} for session {session_id}")
            return record["id"] if record else turn_id

    async def get_conversation_history(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get all conversation turns for a session.

        Args:
            session_id: Session ID.

        Returns:
            List of turn dictionaries with user_message, assistant_response, timestamp.
        """
        cypher = """
        MATCH (s:Session {id: $session_id})-[:HAS_TURN]->(t:ConversationTurn)
        RETURN 
            t.turn_index AS turn_index,
            t.user_message AS user_message,
            t.assistant_response AS assistant_response,
            t.timestamp AS timestamp
        ORDER BY t.turn_index ASC
        """

        turns: list[dict[str, Any]] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, session_id=session_id)
            async for record in result:
                turns.append({
                    "turn_index": record["turn_index"],
                    "user_message": record["user_message"],
                    "assistant_response": record["assistant_response"],
                    "timestamp": datetime.fromisoformat(record["timestamp"]) if record["timestamp"] else None,
                })

        return turns

    # Fact versioning methods

    async def get_fact_with_history(self, fact_id: str) -> FactWithHistory | None:
        """Get a fact with its version history.

        Args:
            fact_id: The fact/insight ID.

        Returns:
            FactWithHistory or None if not found.
        """
        cypher = """
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight {id: $fact_id})-[:HAS_OBJECT]->(o:Entity)
        OPTIONAL MATCH (i)-[:HAS_VERSION]->(v:FactVersion)
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.status AS status,
            i.created_at AS created_at,
            i.updated_at AS updated_at,
            collect({
                id: v.id,
                value: v.value,
                source: v.source,
                session_id: v.session_id,
                valid_from: v.valid_from,
                valid_until: v.valid_until,
                created_at: v.created_at
            }) AS versions
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, fact_id=fact_id)
            record = await result.single()
            if not record:
                return None

            versions = []
            for v in record["versions"]:
                if v["id"]:  # Filter out null versions
                    versions.append(FactVersion(
                        id=v["id"],
                        fact_id=fact_id,
                        value=v["value"],
                        source=v["source"] or "会話",
                        session_id=v["session_id"],
                        valid_from=datetime.fromisoformat(v["valid_from"]) if v["valid_from"] else datetime.now(),
                        valid_until=datetime.fromisoformat(v["valid_until"]) if v["valid_until"] else None,
                        created_at=datetime.fromisoformat(v["created_at"]) if v["created_at"] else datetime.now(),
                    ))

            return FactWithHistory(
                id=record["id"],
                subject=record["subject"],
                predicate=record["predicate"],
                current_value=record["object"],
                status=FactStatus(record["status"]) if record["status"] else FactStatus.ACTIVE,
                domain=Domain(record["domain"]),
                confidence=record["confidence"],
                versions=sorted(versions, key=lambda x: x.valid_from, reverse=True),
                created_at=datetime.fromisoformat(record["created_at"]),
                updated_at=datetime.fromisoformat(record["updated_at"]) if record["updated_at"] else datetime.fromisoformat(record["created_at"]),
            )

    async def update_fact(
        self,
        fact_id: str,
        new_value: str,
        source: str = "会話",
        session_id: str | None = None,
    ) -> str:
        """Update a fact with a new value, creating a version history entry.

        Args:
            fact_id: The fact/insight ID to update.
            new_value: The new value.
            source: Source of the update.
            session_id: Session where this update occurred.

        Returns:
            The new version ID.
        """
        version_id = str(uuid.uuid4())
        now = datetime.now()

        cypher = """
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight {id: $fact_id})-[:HAS_OBJECT]->(old_o:Entity)
        
        // Create version for old value
        CREATE (v:FactVersion {
            id: $version_id,
            value: old_o.name,
            source: $source,
            session_id: $session_id,
            valid_from: i.created_at,
            valid_until: $now,
            created_at: $now
        })
        CREATE (i)-[:HAS_VERSION]->(v)
        
        // Update or create new object entity
        MERGE (new_o:Entity {name: $new_value})
        
        // Remove old relationship and create new one
        DELETE (i)-[:HAS_OBJECT]->(old_o)
        CREATE (i)-[:HAS_OBJECT]->(new_o)
        
        // Update insight metadata
        SET i.updated_at = $now,
            i.status = 'active'
        
        RETURN i.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                fact_id=fact_id,
                version_id=version_id,
                new_value=new_value,
                source=source,
                session_id=session_id,
                now=now.isoformat(),
            )
            record = await result.single()
            logger.info(f"Updated fact {fact_id} to '{new_value}', created version {version_id}")
            return version_id

    async def set_fact_status(
        self,
        fact_id: str,
        status: FactStatus,
    ) -> None:
        """Set the status of a fact.

        Args:
            fact_id: The fact/insight ID.
            status: New status.
        """
        cypher = """
        MATCH (i:Insight {id: $fact_id})
        SET i.status = $status, i.updated_at = $updated_at
        """

        async with self.driver.session() as session:
            await session.run(
                cypher,
                fact_id=fact_id,
                status=status.value,
                updated_at=datetime.now().isoformat(),
            )
            logger.info(f"Set fact {fact_id} status to {status.value}")

    async def resolve_consistency_issue(
        self,
        fact_id: str,
        resolution: str,  # "accept_current" | "keep_previous" | "manual"
        new_value: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a consistency issue for a fact.

        Args:
            fact_id: The fact/insight ID.
            resolution: How to resolve - "accept_current", "keep_previous", or "manual".
            new_value: For "manual" resolution, the value to set.
            session_id: Session where resolution occurred.

        Returns:
            Dict with resolution details.
        """
        # Get current fact state
        fact = await self.get_fact_with_history(fact_id)
        if not fact:
            raise ValueError(f"Fact {fact_id} not found")

        result = {
            "fact_id": fact_id,
            "resolution": resolution,
            "previous_value": fact.current_value,
            "new_value": None,
            "version_id": None,
        }

        if resolution == "accept_current":
            # Current value is already what we want, just mark as active
            await self.set_fact_status(fact_id, FactStatus.ACTIVE)
            result["new_value"] = fact.current_value

        elif resolution == "keep_previous":
            # Revert to previous version
            if fact.versions:
                previous_value = fact.versions[0].value
                version_id = await self.update_fact(
                    fact_id=fact_id,
                    new_value=previous_value,
                    source="整合性解決（以前を維持）",
                    session_id=session_id,
                )
                result["new_value"] = previous_value
                result["version_id"] = version_id
            else:
                # No previous version, just mark as active
                await self.set_fact_status(fact_id, FactStatus.ACTIVE)
                result["new_value"] = fact.current_value

        elif resolution == "manual" and new_value:
            # Set to manually specified value
            version_id = await self.update_fact(
                fact_id=fact_id,
                new_value=new_value,
                source="整合性解決（手動）",
                session_id=session_id,
            )
            result["new_value"] = new_value
            result["version_id"] = version_id

        else:
            raise ValueError(f"Invalid resolution: {resolution}")

        logger.info(f"Resolved consistency issue for fact {fact_id}: {resolution}")
        return result

    # ==================== ConsistencyIssue Management ====================

    async def save_consistency_issue(
        self,
        issue: ConsistencyIssue,
        session_id: str | None = None,
    ) -> str:
        """Save a consistency issue to the knowledge graph.

        Args:
            issue: The consistency issue to save.
            session_id: Session where the issue was detected.

        Returns:
            The ID of the created ConsistencyIssue node.
        """
        issue_id = issue.id or str(uuid.uuid4())
        now = datetime.now()

        cypher = """
        CREATE (ci:ConsistencyIssue {
            id: $id,
            kind: $kind,
            status: $status,
            title: $title,
            fact_id: $fact_id,
            previous_text: $previous_text,
            previous_source: $previous_source,
            current_text: $current_text,
            current_source: $current_source,
            suggested_question: $suggested_question,
            confidence: $confidence,
            resolution: $resolution,
            resolved_at: $resolved_at,
            session_id: $session_id,
            created_at: $created_at
        })
        WITH ci
        OPTIONAL MATCH (i:Insight {id: $fact_id})
        FOREACH (_ IN CASE WHEN i IS NOT NULL THEN [1] ELSE [] END |
            CREATE (i)-[:HAS_CONFLICT]->(ci)
        )
        RETURN ci.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=issue_id,
                kind=issue.kind.value,
                status=issue.status.value,
                title=issue.title,
                fact_id=issue.fact_id,
                previous_text=issue.previous_text,
                previous_source=issue.previous_source,
                current_text=issue.current_text,
                current_source=issue.current_source,
                suggested_question=issue.suggested_question,
                confidence=issue.confidence,
                resolution=issue.resolution,
                resolved_at=issue.resolved_at.isoformat() if issue.resolved_at else None,
                session_id=session_id or issue.session_id,
                created_at=now.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved consistency issue {issue_id} for fact {issue.fact_id}")
            return record["id"] if record else issue_id

    async def get_consistency_issue(self, issue_id: str) -> ConsistencyIssue | None:
        """Get a consistency issue by ID.

        Args:
            issue_id: The issue ID.

        Returns:
            The ConsistencyIssue or None if not found.
        """
        cypher = """
        MATCH (ci:ConsistencyIssue {id: $id})
        RETURN ci
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, id=issue_id)
            record = await result.single()

            if not record:
                return None

            node = record["ci"]
            return self._node_to_consistency_issue(node)

    async def get_unresolved_conflicts(
        self,
        fact_id: str | None = None,
        limit: int = 50,
    ) -> list[ConsistencyIssue]:
        """Get all unresolved consistency issues.

        Args:
            fact_id: Optional fact ID to filter by.
            limit: Maximum number of issues to return.

        Returns:
            List of unresolved ConsistencyIssue objects.
        """
        if fact_id:
            cypher = """
            MATCH (ci:ConsistencyIssue {status: 'unresolved', fact_id: $fact_id})
            RETURN ci
            ORDER BY ci.created_at DESC
            LIMIT $limit
            """
            params = {"fact_id": fact_id, "limit": limit}
        else:
            cypher = """
            MATCH (ci:ConsistencyIssue {status: 'unresolved'})
            RETURN ci
            ORDER BY ci.created_at DESC
            LIMIT $limit
            """
            params = {"limit": limit}

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()

            return [
                self._node_to_consistency_issue(record["ci"])
                for record in records
            ]

    async def get_all_conflicts(
        self,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[ConsistencyIssue]:
        """Get all consistency issues.

        Args:
            include_resolved: Whether to include resolved issues.
            limit: Maximum number of issues to return.

        Returns:
            List of ConsistencyIssue objects.
        """
        if include_resolved:
            cypher = """
            MATCH (ci:ConsistencyIssue)
            RETURN ci
            ORDER BY ci.created_at DESC
            LIMIT $limit
            """
        else:
            cypher = """
            MATCH (ci:ConsistencyIssue)
            WHERE ci.status = 'unresolved'
            RETURN ci
            ORDER BY ci.created_at DESC
            LIMIT $limit
            """

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            records = await result.data()

            return [
                self._node_to_consistency_issue(record["ci"])
                for record in records
            ]

    async def mark_conflict_resolved(
        self,
        conflict_id: str,
        resolution: str,
        new_value: str | None = None,
    ) -> bool:
        """Mark a consistency issue as resolved.

        Args:
            conflict_id: The conflict ID to resolve.
            resolution: Resolution action: "accept_current", "keep_previous", "ignore".
            new_value: For manual resolution, the new value to set.

        Returns:
            True if the conflict was found and updated.
        """
        now = datetime.now()

        # First, get the conflict to find the associated fact
        conflict = await self.get_consistency_issue(conflict_id)
        if not conflict:
            logger.warning(f"Conflict {conflict_id} not found")
            return False

        # Update the conflict status
        cypher = """
        MATCH (ci:ConsistencyIssue {id: $id})
        SET ci.status = $status,
            ci.resolution = $resolution,
            ci.resolved_at = $resolved_at
        RETURN ci
        """

        status = ConsistencyIssueStatus.IGNORED if resolution == "ignore" else ConsistencyIssueStatus.RESOLVED

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=conflict_id,
                status=status.value,
                resolution=resolution,
                resolved_at=now.isoformat(),
            )
            record = await result.single()

            if not record:
                return False

            # If there's an associated fact and resolution is not "ignore", update the fact
            if conflict.fact_id and resolution != "ignore":
                try:
                    await self.resolve_consistency_issue(
                        fact_id=conflict.fact_id,
                        resolution=resolution,
                        new_value=new_value,
                        session_id=conflict.session_id,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update fact {conflict.fact_id}: {e}")

            logger.info(f"Resolved conflict {conflict_id} with action: {resolution}")
            return True

    async def get_conflicts_for_fact(self, fact_id: str) -> list[ConsistencyIssue]:
        """Get all conflicts associated with a fact.

        Args:
            fact_id: The fact ID.

        Returns:
            List of ConsistencyIssue objects for this fact.
        """
        cypher = """
        MATCH (ci:ConsistencyIssue {fact_id: $fact_id})
        RETURN ci
        ORDER BY ci.created_at DESC
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, fact_id=fact_id)
            records = await result.data()

            return [
                self._node_to_consistency_issue(record["ci"])
                for record in records
            ]

    def _node_to_consistency_issue(self, node: dict) -> ConsistencyIssue:
        """Convert a Neo4j node to a ConsistencyIssue object.

        Args:
            node: Neo4j node dict.

        Returns:
            ConsistencyIssue object.
        """
        return ConsistencyIssue(
            id=node.get("id"),
            kind=ConsistencyIssueKind(node.get("kind", "change")),
            status=ConsistencyIssueStatus(node.get("status", "unresolved")),
            title=node.get("title", ""),
            fact_id=node.get("fact_id"),
            previous_text=node.get("previous_text", ""),
            previous_source=node.get("previous_source", ""),
            current_text=node.get("current_text", ""),
            current_source=node.get("current_source", "現在の会話"),
            suggested_question=node.get("suggested_question", ""),
            confidence=node.get("confidence", 0.7),
            resolution=node.get("resolution"),
            resolved_at=datetime.fromisoformat(node["resolved_at"]) if node.get("resolved_at") else None,
            session_id=node.get("session_id"),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
        )

    # ==================== Document Management ====================

    async def save_document(self, document: Document) -> str:
        """Save document metadata to the knowledge graph.

        Args:
            document: Document metadata to save.

        Returns:
            The document ID.
        """
        cypher = """
        CREATE (d:Document {
            id: $id,
            filename: $filename,
            content_type: $content_type,
            size_bytes: $size_bytes,
            extracted_summary: $extracted_summary,
            extracted_facts_count: $extracted_facts_count,
            raw_content_preview: $raw_content_preview,
            topics: $topics,
            entities: $entities,
            domain: $domain,
            status: $status,
            error_message: $error_message,
            page_count: $page_count,
            created_at: $created_at,
            processed_at: $processed_at
        })
        RETURN d.id AS id
        """

        async with self.driver.session() as session:
            await session.run(
                cypher,
                id=document.id,
                filename=document.filename,
                content_type=document.content_type,
                size_bytes=document.size_bytes,
                extracted_summary=document.extracted_summary,
                extracted_facts_count=document.extracted_facts_count,
                raw_content_preview=document.raw_content_preview,
                topics=document.topics,
                entities=document.entities,
                domain=document.domain.value,
                status=document.status.value,
                error_message=document.error_message,
                page_count=document.page_count,
                created_at=document.created_at.isoformat(),
                processed_at=document.processed_at.isoformat() if document.processed_at else None,
            )
            logger.info(f"Saved document: {document.id} ({document.filename})")
            return document.id

    async def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        extraction_result: dict[str, Any] | None = None,
    ) -> None:
        """Update document processing status.

        Args:
            document_id: Document ID.
            status: New status.
            error_message: Error message if failed.
            extraction_result: Extraction result to update fields.
        """
        set_clauses = ["d.status = $status", "d.error_message = $error_message"]
        params: dict[str, Any] = {
            "document_id": document_id,
            "status": status.value,
            "error_message": error_message,
        }

        if status == DocumentStatus.COMPLETED:
            set_clauses.append("d.processed_at = $processed_at")
            params["processed_at"] = datetime.now().isoformat()

        if extraction_result:
            if "summary" in extraction_result:
                set_clauses.append("d.extracted_summary = $summary")
                params["summary"] = extraction_result["summary"]
            if "facts_count" in extraction_result:
                set_clauses.append("d.extracted_facts_count = $facts_count")
                params["facts_count"] = extraction_result["facts_count"]
            if "topics" in extraction_result:
                set_clauses.append("d.topics = $topics")
                params["topics"] = extraction_result["topics"]
            if "entities" in extraction_result:
                set_clauses.append("d.entities = $entities")
                params["entities"] = extraction_result["entities"]
            if "domain" in extraction_result:
                set_clauses.append("d.domain = $domain")
                params["domain"] = extraction_result["domain"]

        cypher = f"""
        MATCH (d:Document {{id: $document_id}})
        SET {', '.join(set_clauses)}
        """

        async with self.driver.session() as session:
            await session.run(cypher, **params)
            logger.info(f"Updated document {document_id} status to {status.value}")

    async def get_document(self, document_id: str) -> Document | None:
        """Get document by ID.

        Args:
            document_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})
        RETURN d
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            record = await result.single()

            if not record:
                return None

            node = record["d"]
            return Document(
                id=node["id"],
                filename=node["filename"],
                content_type=node["content_type"],
                size_bytes=node["size_bytes"],
                extracted_summary=node.get("extracted_summary", ""),
                extracted_facts_count=node.get("extracted_facts_count", 0),
                raw_content_preview=node.get("raw_content_preview", ""),
                topics=list(node.get("topics", [])),
                entities=list(node.get("entities", [])),
                domain=Domain(node.get("domain", "general")),
                status=DocumentStatus(node["status"]),
                error_message=node.get("error_message"),
                page_count=node.get("page_count", 1),
                created_at=datetime.fromisoformat(node["created_at"]),
                processed_at=datetime.fromisoformat(node["processed_at"]) if node.get("processed_at") else None,
            )

    async def get_documents(self, limit: int = 50) -> list[Document]:
        """Get all documents.

        Args:
            limit: Maximum number of documents.

        Returns:
            List of documents.
        """
        cypher = """
        MATCH (d:Document)
        RETURN d
        ORDER BY d.created_at DESC
        LIMIT $limit
        """

        documents: list[Document] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                node = record["d"]
                documents.append(Document(
                    id=node["id"],
                    filename=node["filename"],
                    content_type=node["content_type"],
                    size_bytes=node["size_bytes"],
                    extracted_summary=node.get("extracted_summary", ""),
                    extracted_facts_count=node.get("extracted_facts_count", 0),
                    raw_content_preview=node.get("raw_content_preview", ""),
                    topics=list(node.get("topics", [])),
                    entities=list(node.get("entities", [])),
                    domain=Domain(node.get("domain", "general")),
                    status=DocumentStatus(node["status"]),
                    error_message=node.get("error_message"),
                    page_count=node.get("page_count", 1),
                    created_at=datetime.fromisoformat(node["created_at"]),
                    processed_at=datetime.fromisoformat(node["processed_at"]) if node.get("processed_at") else None,
                ))

        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its linked facts.

        Args:
            document_id: Document ID.

        Returns:
            True if deleted, False if not found.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[:EXTRACTED]->(i:Insight)
        DETACH DELETE d, i
        RETURN count(d) AS deleted
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            record = await result.single()
            deleted = record["deleted"] > 0
            if deleted:
                logger.info(f"Deleted document: {document_id}")
            return deleted

    async def link_session_to_document(self, session_id: str, document_id: str) -> None:
        """Link a session to a document for review purposes.
        
        Args:
            session_id: Session ID.
            document_id: Document ID.
        """
        cypher = """
        MATCH (s:Session {id: $session_id})
        MATCH (d:Document {id: $document_id})
        MERGE (s)-[:REVIEWS_DOCUMENT]->(d)
        """
        
        async with self.driver.session() as session:
            await session.run(cypher, session_id=session_id, document_id=document_id)
            logger.info(f"Linked session {session_id} to document {document_id}")

    # ==================== PendingQuestion Persistence ====================

    async def save_pending_question(
        self,
        question: PendingQuestion,
        session_id: str,
    ) -> str:
        """Save a pending question to the knowledge graph.

        Args:
            question: The PendingQuestion to save.
            session_id: Session ID this question belongs to.

        Returns:
            The ID of the saved PendingQuestion node.
        """
        question_id = question.id or str(uuid.uuid4())
        now = datetime.now()

        cypher = """
        CREATE (pq:PendingQuestion {
            id: $id,
            kind: $kind,
            question: $question,
            context: $context,
            related_fact_id: $related_fact_id,
            related_entity: $related_entity,
            priority: $priority,
            status: $status,
            answer: $answer,
            session_id: $session_id,
            asked_at: $asked_at,
            answered_at: $answered_at,
            created_at: $created_at
        })
        WITH pq
        OPTIONAL MATCH (s:Session {id: $session_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            CREATE (s)-[:HAS_QUESTION]->(pq)
        )
        RETURN pq.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=question_id,
                kind=question.kind.value,
                question=question.question,
                context=question.context,
                related_fact_id=question.related_fact_id,
                related_entity=question.related_entity,
                priority=question.priority,
                status=question.status.value,
                answer=question.answer,
                session_id=session_id,
                asked_at=question.asked_at.isoformat() if question.asked_at else None,
                answered_at=question.answered_at.isoformat() if question.answered_at else None,
                created_at=now.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved pending question {question_id} for session {session_id}")
            return record["id"] if record else question_id

    async def get_pending_questions(
        self,
        session_id: str,
        status: str | None = None,
    ) -> list[PendingQuestion]:
        """Get pending questions for a session.

        Args:
            session_id: Session ID to query.
            status: Optional status filter (e.g., "pending", "answered").

        Returns:
            List of PendingQuestion objects.
        """
        if status:
            cypher = """
            MATCH (pq:PendingQuestion {session_id: $session_id, status: $status})
            RETURN pq
            ORDER BY pq.priority DESC, pq.created_at ASC
            """
            params: dict[str, Any] = {"session_id": session_id, "status": status}
        else:
            cypher = """
            MATCH (pq:PendingQuestion {session_id: $session_id})
            RETURN pq
            ORDER BY pq.priority DESC, pq.created_at ASC
            """
            params = {"session_id": session_id}

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()

            questions: list[PendingQuestion] = []
            for record in records:
                node = record["pq"]
                try:
                    q = PendingQuestion(
                        id=node["id"],
                        kind=QuestionKind(node["kind"]),
                        question=node["question"],
                        context=node.get("context", ""),
                        related_fact_id=node.get("related_fact_id"),
                        related_entity=node.get("related_entity"),
                        priority=node.get("priority", 0),
                        status=QuestionStatus(node.get("status", "pending")),
                        answer=node.get("answer"),
                        session_id=node.get("session_id"),
                        asked_at=datetime.fromisoformat(node["asked_at"]) if node.get("asked_at") else None,
                        answered_at=datetime.fromisoformat(node["answered_at"]) if node.get("answered_at") else None,
                        created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
                    )
                    questions.append(q)
                except Exception as e:
                    logger.warning(f"Failed to parse PendingQuestion node: {e}")
                    continue

            return questions

    async def get_unresolved_questions_for_session(
        self,
        session_id: str,
    ) -> list[PendingQuestion]:
        """Get unresolved (pending) questions for a session.

        Args:
            session_id: Session ID to query.

        Returns:
            List of PendingQuestion objects with status 'pending'.
        """
        return await self.get_pending_questions(session_id, status="pending")

    async def update_question_status(
        self,
        question_id: str,
        status: str,
        answer: str | None = None,
    ) -> None:
        """Update the status and answer of a pending question.

        Args:
            question_id: ID of the question to update.
            status: New status value (pending, answered, skipped, expired).
            answer: Optional answer text.
        """
        now = datetime.now()

        cypher = """
        MATCH (pq:PendingQuestion {id: $id})
        SET pq.status = $status,
            pq.answer = CASE WHEN $answer IS NOT NULL THEN $answer ELSE pq.answer END,
            pq.answered_at = CASE WHEN $status IN ['answered', 'skipped'] THEN $now ELSE pq.answered_at END
        RETURN pq.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=question_id,
                status=status,
                answer=answer,
                now=now.isoformat(),
            )
            record = await result.single()
            if record:
                logger.info(f"Updated question {question_id} status to {status}")
            else:
                logger.warning(f"Question {question_id} not found for status update")

    async def save_pending_questions_batch(
        self,
        questions: list[PendingQuestion],
        session_id: str,
    ) -> None:
        """Save multiple pending questions in a batch operation.

        Args:
            questions: List of PendingQuestion objects to save.
            session_id: Session ID they belong to.
        """
        if not questions:
            return

        cypher = """
        UNWIND $questions AS q
        CREATE (pq:PendingQuestion {
            id: q.id,
            kind: q.kind,
            question: q.question,
            context: q.context,
            related_fact_id: q.related_fact_id,
            related_entity: q.related_entity,
            priority: q.priority,
            status: q.status,
            answer: q.answer,
            session_id: $session_id,
            asked_at: q.asked_at,
            answered_at: q.answered_at,
            created_at: q.created_at
        })
        WITH pq
        OPTIONAL MATCH (s:Session {id: $session_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            CREATE (s)-[:HAS_QUESTION]->(pq)
        )
        """

        now = datetime.now()
        questions_data = []
        for q in questions:
            questions_data.append({
                "id": q.id or str(uuid.uuid4()),
                "kind": q.kind.value,
                "question": q.question,
                "context": q.context,
                "related_fact_id": q.related_fact_id,
                "related_entity": q.related_entity,
                "priority": q.priority,
                "status": q.status.value,
                "answer": q.answer,
                "asked_at": q.asked_at.isoformat() if q.asked_at else None,
                "answered_at": q.answered_at.isoformat() if q.answered_at else None,
                "created_at": now.isoformat(),
            })

        async with self.driver.session() as session:
            await session.run(cypher, questions=questions_data, session_id=session_id)
            logger.info(f"Batch saved {len(questions)} pending questions for session {session_id}")

    async def link_fact_to_document(self, fact_id: str, document_id: str) -> None:
        """Link a fact to its source document.

        Args:
            fact_id: The fact/insight ID.
            document_id: The source document ID.
        """
        cypher = """
        MATCH (i:Insight {id: $fact_id})
        MATCH (d:Document {id: $document_id})
        MERGE (d)-[:EXTRACTED]->(i)
        """

        async with self.driver.session() as session:
            await session.run(cypher, fact_id=fact_id, document_id=document_id)

    async def get_insights_for_document(self, document_id: str) -> list[Insight]:
        """Get all insights/facts extracted from a document.

        Args:
            document_id: The document ID.

        Returns:
            List of insights linked to the document.
        """
        # Insight subject/object are stored as Entity relationships
        cypher = """
        MATCH (d:Document {id: $document_id})-[:EXTRACTED]->(i:Insight)
        OPTIONAL MATCH (s:Entity)-[:SUBJECT_OF]->(i)
        OPTIONAL MATCH (i)-[:HAS_OBJECT]->(o:Entity)
        RETURN i, s.name AS subject, o.name AS object
        ORDER BY i.created_at DESC
        """

        insights: list[Insight] = []
        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            async for record in result:
                node = record["i"]
                subject = record["subject"]
                obj = record["object"]
                
                # Skip insights with missing required fields
                if not subject or not obj:
                    continue
                try:
                    insights.append(Insight(
                        id=node["id"],
                        subject=subject,
                        predicate=node.get("predicate", ""),
                        object=obj,
                        confidence=node.get("confidence", 1.0),
                        domain=Domain(node.get("domain", "general")),
                        event_date=datetime.fromisoformat(node["event_date"]) if node.get("event_date") else None,
                        event_date_end=datetime.fromisoformat(node["event_date_end"]) if node.get("event_date_end") else None,
                        date_type=DateType(node["date_type"]) if node.get("date_type") else None,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse insight {node.get('id')}: {e}")
                    continue
        return insights

    async def delete_document_insights(self, document_id: str) -> int:
        """Delete all insights/facts linked to a document (without deleting the document itself).

        Args:
            document_id: The document ID.

        Returns:
            Number of insights deleted.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})-[:EXTRACTED]->(i:Insight)
        DETACH DELETE i
        RETURN count(i) AS deleted
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            record = await result.single()
            deleted = record["deleted"] if record else 0
            if deleted:
                logger.info(f"Deleted {deleted} insights for document {document_id}")
            return deleted

    async def get_all_recent_insights(self, limit: int = 50) -> list[dict]:
        """Get recent insights from all sources (documents and sessions).

        Args:
            limit: Maximum number of insights to return.

        Returns:
            List of insight dicts with source information.
        """
        cypher = """
        MATCH (i:Insight)
        OPTIONAL MATCH (s:Entity)-[:SUBJECT_OF]->(i)
        OPTIONAL MATCH (i)-[:HAS_OBJECT]->(o:Entity)
        OPTIONAL MATCH (d:Document)-[:EXTRACTED]->(i)
        OPTIONAL MATCH (i)-[:HAS_TAG]->(t:Tag)
        WITH i, s.name AS subject, o.name AS object, 
             d.id AS document_id, d.filename AS document_filename,
             collect(DISTINCT t.name) AS tags
        WHERE subject IS NOT NULL AND object IS NOT NULL
        RETURN i, subject, object, document_id, document_filename, tags
        ORDER BY i.created_at DESC
        LIMIT $limit
        """

        insights: list[dict] = []
        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                node = record["i"]
                try:
                    insights.append({
                        "id": node["id"],
                        "subject": record["subject"],
                        "predicate": node.get("predicate", ""),
                        "object": record["object"],
                        "confidence": node.get("confidence", 1.0),
                        "domain": node.get("domain", "general"),
                        "verified": node.get("status", "pending") == "verified",
                        "event_date": node.get("event_date"),
                        "document_id": record["document_id"],
                        "document_filename": record["document_filename"],
                        "tags": record["tags"] or [],
                        "created_at": node.get("created_at"),
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse insight {node.get('id')}: {e}")
                    continue
        return insights

    # ==================== Knowledge Aggregation ====================

    async def get_all_topics(self, limit: int = 50) -> list[TopicStats]:
        """Get all topics with their fact counts.

        Args:
            limit: Maximum number of topics.

        Returns:
            List of topic statistics.
        """
        # Topics come from insight domains and explicit topic tags
        cypher = """
        MATCH (i:Insight)
        WHERE i.domain IS NOT NULL
        WITH i.domain AS topic, count(i) AS fact_count, max(i.created_at) AS last_updated
        RETURN topic AS name, fact_count, 0 AS document_count, last_updated
        ORDER BY fact_count DESC
        LIMIT $limit
        """

        topics: list[TopicStats] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                topics.append(TopicStats(
                    name=record["name"],
                    fact_count=record["fact_count"],
                    document_count=record["document_count"],
                    last_updated=datetime.fromisoformat(record["last_updated"]) if record["last_updated"] else None,
                ))

        return topics

    async def get_all_entities(self, limit: int = 50) -> list[TopicStats]:
        """Get all entities with their relationship counts.

        Args:
            limit: Maximum number of entities.

        Returns:
            List of entity statistics.
        """
        cypher = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:SUBJECT_OF]->(i1:Insight)
        OPTIONAL MATCH (i2:Insight)-[:HAS_OBJECT]->(e)
        WITH e.name AS name, 
             count(DISTINCT i1) + count(DISTINCT i2) AS fact_count,
             max(coalesce(i1.created_at, i2.created_at)) AS last_updated
        WHERE fact_count > 0
        RETURN name, fact_count, 0 AS document_count, last_updated
        ORDER BY fact_count DESC
        LIMIT $limit
        """

        entities: list[TopicStats] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                entities.append(TopicStats(
                    name=record["name"],
                    fact_count=record["fact_count"],
                    document_count=record["document_count"],
                    last_updated=datetime.fromisoformat(record["last_updated"]) if record["last_updated"] else None,
                ))

        return entities

    async def get_facts_by_topic(self, topic: str, limit: int = 100) -> list[KnowledgeItem]:
        """Get all facts for a specific topic/domain.

        Args:
            topic: Topic name (domain value).
            limit: Maximum number of facts.

        Returns:
            List of knowledge items.
        """
        cypher = """
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight)-[:HAS_OBJECT]->(o:Entity)
        WHERE i.domain = $topic
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.created_at AS created_at,
            coalesce(i.status, 'active') AS status,
            i.event_date AS event_date,
            i.event_date_end AS event_date_end,
            i.date_type AS date_type
        ORDER BY i.created_at DESC
        LIMIT $limit
        """

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, topic=topic, limit=limit)
            async for record in result:
                items.append(KnowledgeItem(
                    id=record["id"],
                    subject=record["subject"],
                    predicate=record["predicate"],
                    object=record["object"],
                    confidence=record["confidence"],
                    domain=Domain(record["domain"]),
                    created_at=datetime.fromisoformat(record["created_at"]),
                    status=FactStatus(record["status"]),
                    event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                    event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                    date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                ))

        return items

    async def get_facts_by_entity(self, entity: str, limit: int = 100) -> list[KnowledgeItem]:
        """Get all facts related to a specific entity.

        Args:
            entity: Entity name.
            limit: Maximum number of facts.

        Returns:
            List of knowledge items.
        """
        cypher = """
        MATCH (e:Entity {name: $entity})
        OPTIONAL MATCH (e)-[:SUBJECT_OF]->(i1:Insight)-[:HAS_OBJECT]->(o1:Entity)
        OPTIONAL MATCH (s2:Entity)-[:SUBJECT_OF]->(i2:Insight)-[:HAS_OBJECT]->(e)
        WITH collect({
            id: i1.id, subject: e.name, predicate: i1.predicate, object: o1.name,
            confidence: i1.confidence, domain: i1.domain, created_at: i1.created_at,
            status: coalesce(i1.status, 'active'),
            event_date: i1.event_date, event_date_end: i1.event_date_end, date_type: i1.date_type
        }) + collect({
            id: i2.id, subject: s2.name, predicate: i2.predicate, object: e.name,
            confidence: i2.confidence, domain: i2.domain, created_at: i2.created_at,
            status: coalesce(i2.status, 'active'),
            event_date: i2.event_date, event_date_end: i2.event_date_end, date_type: i2.date_type
        }) AS all_facts
        UNWIND all_facts AS f
        WHERE f.id IS NOT NULL
        RETURN DISTINCT f.id AS id, f.subject AS subject, f.predicate AS predicate,
               f.object AS object, f.confidence AS confidence, f.domain AS domain,
               f.created_at AS created_at, f.status AS status,
               f.event_date AS event_date, f.event_date_end AS event_date_end, f.date_type AS date_type
        ORDER BY f.created_at DESC
        LIMIT $limit
        """

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, entity=entity, limit=limit)
            async for record in result:
                if record["id"]:
                    items.append(KnowledgeItem(
                        id=record["id"],
                        subject=record["subject"],
                        predicate=record["predicate"],
                        object=record["object"],
                        confidence=record["confidence"],
                        domain=Domain(record["domain"]),
                        created_at=datetime.fromisoformat(record["created_at"]),
                        status=FactStatus(record["status"]),
                        event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                        event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                        date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                    ))

        return items

    async def get_knowledge_stats(self) -> KnowledgeStats:
        """Get overall knowledge base statistics.

        Returns:
            Knowledge base statistics.
        """
        cypher = """
        MATCH (i:Insight)
        WITH count(i) AS total_facts
        OPTIONAL MATCH (d:Document)
        WITH total_facts, count(d) AS total_documents
        OPTIONAL MATCH (s:Session)
        RETURN total_facts, total_documents, count(s) AS total_sessions
        """

        async with self.driver.session() as session:
            result = await session.run(cypher)
            record = await result.single()

            topics = await self.get_all_topics(limit=10)
            entities = await self.get_all_entities(limit=10)

            return KnowledgeStats(
                total_facts=record["total_facts"] if record else 0,
                total_documents=record["total_documents"] if record else 0,
                total_sessions=record["total_sessions"] if record else 0,
                topics=topics,
                entities=entities,
            )

    # ==================== Document Chunk Management ====================

    async def save_chunk(self, chunk: DocumentChunk) -> str:
        """Save a document chunk to the knowledge graph.

        Args:
            chunk: The document chunk to save.

        Returns:
            The chunk ID.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})
        CREATE (c:Chunk {
            id: $id,
            content: $content,
            chunk_index: $chunk_index,
            chunk_date: $chunk_date,
            chunk_date_end: $chunk_date_end,
            heading: $heading,
            char_count: $char_count,
            created_at: $created_at
        })
        CREATE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                chunk_date=chunk.chunk_date.isoformat() if chunk.chunk_date else None,
                chunk_date_end=chunk.chunk_date_end.isoformat() if chunk.chunk_date_end else None,
                heading=chunk.heading,
                char_count=chunk.char_count,
                created_at=chunk.created_at.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved chunk: {chunk.id} (doc: {chunk.document_id})")
            return record["id"] if record else chunk.id

    async def save_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        """Save multiple document chunks.

        Args:
            chunks: List of document chunks to save.

        Returns:
            List of chunk IDs.
        """
        chunk_ids: list[str] = []
        for chunk in chunks:
            chunk_id = await self.save_chunk(chunk)
            chunk_ids.append(chunk_id)
        return chunk_ids

    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Get a document chunk by ID.

        Args:
            chunk_id: Chunk ID.

        Returns:
            DocumentChunk if found, None otherwise.
        """
        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
        RETURN c, d.id AS document_id
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, chunk_id=chunk_id)
            record = await result.single()

            if not record:
                return None

            node = record["c"]
            return DocumentChunk(
                id=node["id"],
                document_id=record["document_id"],
                content=node["content"],
                chunk_index=node["chunk_index"],
                chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                heading=node.get("heading", ""),
                char_count=node.get("char_count", 0),
                created_at=datetime.fromisoformat(node["created_at"]),
            )

    async def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID.

        Returns:
            List of document chunks ordered by chunk_index.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c, d.id AS document_id
        ORDER BY c.chunk_index ASC
        """

        chunks: list[DocumentChunk] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            async for record in result:
                node = record["c"]
                chunks.append(DocumentChunk(
                    id=node["id"],
                    document_id=record["document_id"],
                    content=node["content"],
                    chunk_index=node["chunk_index"],
                    chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                    chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                    heading=node.get("heading", ""),
                    char_count=node.get("char_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                ))

        return chunks

    async def get_chunks_by_date(
        self,
        target_date: datetime,
        tolerance_days: int = 0,
    ) -> list[DocumentChunk]:
        """Get document chunks matching a specific date.

        Args:
            target_date: The date to search for.
            tolerance_days: Number of days tolerance (0 for exact match).

        Returns:
            List of matching document chunks.
        """
        if tolerance_days == 0:
            # Exact date match (same day)
            start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            from datetime import timedelta
            start_date = target_date - timedelta(days=tolerance_days)
            end_date = target_date + timedelta(days=tolerance_days)

        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.chunk_date IS NOT NULL
          AND c.chunk_date >= $start_date
          AND c.chunk_date <= $end_date
        RETURN c, d.id AS document_id
        ORDER BY c.chunk_date ASC
        """

        chunks: list[DocumentChunk] = []

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            async for record in result:
                node = record["c"]
                chunks.append(DocumentChunk(
                    id=node["id"],
                    document_id=record["document_id"],
                    content=node["content"],
                    chunk_index=node["chunk_index"],
                    chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                    chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                    heading=node.get("heading", ""),
                    char_count=node.get("char_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                ))

        return chunks

    async def get_chunks_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DocumentChunk]:
        """Get document chunks within a date range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            List of document chunks ordered by date.
        """
        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.chunk_date IS NOT NULL
          AND c.chunk_date >= $start_date
          AND c.chunk_date <= $end_date
        RETURN c, d.id AS document_id
        ORDER BY c.chunk_date ASC
        """

        chunks: list[DocumentChunk] = []

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            async for record in result:
                node = record["c"]
                chunks.append(DocumentChunk(
                    id=node["id"],
                    document_id=record["document_id"],
                    content=node["content"],
                    chunk_index=node["chunk_index"],
                    chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                    chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                    heading=node.get("heading", ""),
                    char_count=node.get("char_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                ))

        return chunks

    async def search_chunks(
        self,
        query: str,
        limit: int = 10,
    ) -> list[DocumentChunk]:
        """Search chunks by content.

        Args:
            query: Search query (keywords).
            limit: Maximum number of results.

        Returns:
            List of matching document chunks.
        """
        cypher = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.content CONTAINS $query OR c.heading CONTAINS $query
        RETURN c, d.id AS document_id
        ORDER BY c.created_at DESC
        LIMIT $limit
        """

        chunks: list[DocumentChunk] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, query=query, limit=limit)
            async for record in result:
                node = record["c"]
                chunks.append(DocumentChunk(
                    id=node["id"],
                    document_id=record["document_id"],
                    content=node["content"],
                    chunk_index=node["chunk_index"],
                    chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                    chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                    heading=node.get("heading", ""),
                    char_count=node.get("char_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                ))

        return chunks

    async def link_insight_to_chunk(self, insight_id: str, chunk_id: str) -> None:
        """Link an insight to its source chunk.

        This allows tracing insights back to the original document content.

        Args:
            insight_id: The insight ID.
            chunk_id: The source chunk ID.
        """
        cypher = """
        MATCH (i:Insight {id: $insight_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (i)-[:EXTRACTED_FROM]->(c)
        """

        async with self.driver.session() as session:
            await session.run(cypher, insight_id=insight_id, chunk_id=chunk_id)
            logger.debug(f"Linked insight {insight_id} to chunk {chunk_id}")

    async def get_chunk_for_insight(self, insight_id: str) -> DocumentChunk | None:
        """Get the source chunk for an insight.

        Args:
            insight_id: The insight ID.

        Returns:
            Source chunk if found, None otherwise.
        """
        cypher = """
        MATCH (i:Insight {id: $insight_id})-[:EXTRACTED_FROM]->(c:Chunk)
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        RETURN c, d.id AS document_id
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, insight_id=insight_id)
            record = await result.single()

            if not record:
                return None

            node = record["c"]
            return DocumentChunk(
                id=node["id"],
                document_id=record["document_id"],
                content=node["content"],
                chunk_index=node["chunk_index"],
                chunk_date=datetime.fromisoformat(node["chunk_date"]) if node.get("chunk_date") else None,
                chunk_date_end=datetime.fromisoformat(node["chunk_date_end"]) if node.get("chunk_date_end") else None,
                heading=node.get("heading", ""),
                char_count=node.get("char_count", 0),
                created_at=datetime.fromisoformat(node["created_at"]),
            )

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID.

        Returns:
            Number of chunks deleted.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        WITH c, count(c) AS chunk_count
        DETACH DELETE c
        RETURN chunk_count
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            record = await result.single()
            deleted = record["chunk_count"] if record else 0
            if deleted:
                logger.info(f"Deleted {deleted} chunks for document {document_id}")
            return deleted

    async def setup_indexes(self) -> None:
        """Set up required indexes for the knowledge graph."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX insight_id IF NOT EXISTS FOR (i:Insight) ON (i.id)",
            "CREATE INDEX insight_domain IF NOT EXISTS FOR (i:Insight) ON (i.domain)",
            "CREATE INDEX insight_session IF NOT EXISTS FOR (i:Insight) ON (i.session_id)",
            "CREATE INDEX insight_status IF NOT EXISTS FOR (i:Insight) ON (i.status)",
            # Temporal indexes for date-based queries
            "CREATE INDEX insight_event_date IF NOT EXISTS FOR (i:Insight) ON (i.event_date)",
            "CREATE INDEX insight_date_type IF NOT EXISTS FOR (i:Insight) ON (i.date_type)",
            "CREATE INDEX session_id IF NOT EXISTS FOR (s:Session) ON (s.id)",
            "CREATE INDEX session_user IF NOT EXISTS FOR (s:Session) ON (s.user_id)",
            "CREATE INDEX session_status IF NOT EXISTS FOR (s:Session) ON (s.status)",
            "CREATE INDEX summary_id IF NOT EXISTS FOR (sum:Summary) ON (sum.id)",
            "CREATE INDEX fact_version_id IF NOT EXISTS FOR (v:FactVersion) ON (v.id)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX document_status IF NOT EXISTS FOR (d:Document) ON (d.status)",
            # Conversation turn indexes
            "CREATE INDEX turn_id IF NOT EXISTS FOR (t:ConversationTurn) ON (t.id)",
            "CREATE INDEX turn_session IF NOT EXISTS FOR (t:ConversationTurn) ON (t.session_id)",
            "CREATE INDEX turn_index IF NOT EXISTS FOR (t:ConversationTurn) ON (t.turn_index)",
            # Chunk indexes
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX chunk_date IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_date)",
            "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            # Tag indexes
            "CREATE INDEX tag_id IF NOT EXISTS FOR (t:Tag) ON (t.id)",
            "CREATE INDEX tag_name IF NOT EXISTS FOR (t:Tag) ON (t.name)",
            "CREATE INDEX tag_usage IF NOT EXISTS FOR (t:Tag) ON (t.usage_count)",
            # ConsistencyIssue indexes
            "CREATE INDEX conflict_id IF NOT EXISTS FOR (c:ConsistencyIssue) ON (c.id)",
            "CREATE INDEX conflict_status IF NOT EXISTS FOR (c:ConsistencyIssue) ON (c.status)",
            "CREATE INDEX conflict_fact IF NOT EXISTS FOR (c:ConsistencyIssue) ON (c.fact_id)",
            "CREATE INDEX conflict_session IF NOT EXISTS FOR (c:ConsistencyIssue) ON (c.session_id)",
            """
            CREATE FULLTEXT INDEX insight_search IF NOT EXISTS 
            FOR (i:Insight) ON EACH [i.predicate]
            """,
            """
            CREATE FULLTEXT INDEX chunk_content_search IF NOT EXISTS 
            FOR (c:Chunk) ON EACH [c.content, c.heading]
            """,
            """
            CREATE FULLTEXT INDEX summary_search IF NOT EXISTS 
            FOR (sum:Summary) ON EACH [sum.content, sum.topics]
            """,
            """
            CREATE FULLTEXT INDEX tag_search IF NOT EXISTS
            FOR (t:Tag) ON EACH [t.name, t.aliases]
            """,
        ]

        async with self.driver.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                    logger.info(f"Created index: {index_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")

    async def search_by_date_range(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        query: str | None = None,
        domain: Domain | None = None,
        limit: int = 20,
    ) -> list[KnowledgeItem]:
        """Search knowledge items by date range and optional keyword.

        Args:
            start_date: Start date for filtering (inclusive).
            end_date: End date for filtering (inclusive).
            query: Optional keyword search term.
            domain: Optional domain filter.
            limit: Maximum number of results.

        Returns:
            List of matching knowledge items ordered by event_date.
        """
        conditions = ["i.event_date IS NOT NULL"]
        params: dict[str, Any] = {"limit": limit}

        if start_date:
            conditions.append("i.event_date >= $start_date")
            params["start_date"] = start_date.isoformat()

        if end_date:
            conditions.append("i.event_date <= $end_date")
            params["end_date"] = end_date.isoformat()

        if domain:
            conditions.append("i.domain = $domain")
            params["domain"] = domain.value

        if query:
            conditions.append(
                "(s.name CONTAINS $query OR o.name CONTAINS $query OR i.predicate CONTAINS $query)"
            )
            params["query"] = query

        where_clause = " AND ".join(conditions)

        cypher = f"""
        MATCH (s:Entity)-[:SUBJECT_OF]->(i:Insight)-[:HAS_OBJECT]->(o:Entity)
        WHERE {where_clause}
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.created_at AS created_at,
            i.event_date AS event_date,
            i.event_date_end AS event_date_end,
            i.date_type AS date_type,
            coalesce(i.status, 'active') AS status
        ORDER BY i.event_date ASC
        LIMIT $limit
        """

        items: list[KnowledgeItem] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            async for record in result:
                items.append(KnowledgeItem(
                    id=record["id"],
                    subject=record["subject"],
                    predicate=record["predicate"],
                    object=record["object"],
                    confidence=record["confidence"],
                    domain=Domain(record["domain"]),
                    created_at=datetime.fromisoformat(record["created_at"]),
                    event_date=datetime.fromisoformat(record["event_date"]) if record["event_date"] else None,
                    event_date_end=datetime.fromisoformat(record["event_date_end"]) if record["event_date_end"] else None,
                    date_type=DateType(record["date_type"]) if record["date_type"] else DateType.UNKNOWN,
                    status=FactStatus(record["status"]),
                ))

        return items

    # ==================== Tag Management ====================

    async def save_tag(self, tag: Tag) -> str:
        """Save a tag to the knowledge graph.

        Args:
            tag: The tag to save.

        Returns:
            The tag ID.
        """
        cypher = """
        MERGE (t:Tag {id: $id})
        SET t.name = $name,
            t.aliases = $aliases,
            t.color = $color,
            t.description = $description,
            t.usage_count = $usage_count,
            t.created_at = $created_at,
            t.updated_at = $updated_at
        RETURN t.id AS id
        """

        async with self.driver.session() as session:
            result = await session.run(
                cypher,
                id=tag.id,
                name=tag.name,
                aliases=tag.aliases,
                color=tag.color,
                description=tag.description,
                usage_count=tag.usage_count,
                created_at=tag.created_at.isoformat(),
                updated_at=tag.updated_at.isoformat(),
            )
            record = await result.single()
            logger.info(f"Saved tag: {tag.name} ({tag.id})")
            return record["id"] if record else tag.id

    async def create_tag(self, name: str, color: str | None = None, description: str = "") -> Tag:
        """Create a new tag with auto-generated ID.

        Args:
            name: Tag name.
            color: Optional color for UI display.
            description: Optional description.

        Returns:
            The created Tag object.
        """
        tag = Tag(
            id=str(uuid.uuid4()),
            name=name.strip(),
            aliases=[],
            color=color,
            description=description,
            usage_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        await self.save_tag(tag)
        return tag

    async def get_tag(self, tag_id: str) -> Tag | None:
        """Get a tag by ID.

        Args:
            tag_id: Tag ID.

        Returns:
            Tag if found, None otherwise.
        """
        cypher = """
        MATCH (t:Tag {id: $tag_id})
        RETURN t
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, tag_id=tag_id)
            record = await result.single()

            if not record:
                return None

            node = record["t"]
            return Tag(
                id=node["id"],
                name=node["name"],
                aliases=list(node.get("aliases", [])),
                color=node.get("color"),
                description=node.get("description", ""),
                usage_count=node.get("usage_count", 0),
                created_at=datetime.fromisoformat(node["created_at"]),
                updated_at=datetime.fromisoformat(node["updated_at"]),
            )

    async def get_tag_by_name(self, name: str) -> Tag | None:
        """Get a tag by name (case-insensitive) or alias.

        Args:
            name: Tag name or alias to search for.

        Returns:
            Tag if found, None otherwise.
        """
        normalized_name = name.strip().lower()
        cypher = """
        MATCH (t:Tag)
        WHERE toLower(t.name) = $name 
           OR any(alias IN t.aliases WHERE toLower(alias) = $name)
        RETURN t
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, name=normalized_name)
            record = await result.single()

            if not record:
                return None

            node = record["t"]
            return Tag(
                id=node["id"],
                name=node["name"],
                aliases=list(node.get("aliases", [])),
                color=node.get("color"),
                description=node.get("description", ""),
                usage_count=node.get("usage_count", 0),
                created_at=datetime.fromisoformat(node["created_at"]),
                updated_at=datetime.fromisoformat(node["updated_at"]),
            )

    async def get_all_tags(self, limit: int = 100) -> list[Tag]:
        """Get all tags ordered by usage count.

        Args:
            limit: Maximum number of tags.

        Returns:
            List of tags.
        """
        cypher = """
        MATCH (t:Tag)
        RETURN t
        ORDER BY t.usage_count DESC
        LIMIT $limit
        """

        tags: list[Tag] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                node = record["t"]
                tags.append(Tag(
                    id=node["id"],
                    name=node["name"],
                    aliases=list(node.get("aliases", [])),
                    color=node.get("color"),
                    description=node.get("description", ""),
                    usage_count=node.get("usage_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                    updated_at=datetime.fromisoformat(node["updated_at"]),
                ))

        return tags

    async def get_tag_stats(self, tag_id: str) -> TagStats | None:
        """Get statistics for a tag.

        Args:
            tag_id: Tag ID.

        Returns:
            TagStats if found, None otherwise.
        """
        cypher = """
        MATCH (t:Tag {id: $tag_id})
        OPTIONAL MATCH (i:Insight)-[ri:HAS_TAG]->(t)
        OPTIONAL MATCH (d:Document)-[rd:HAS_TAG]->(t)
        WITH t, 
             count(DISTINCT i) AS insight_count,
             count(DISTINCT d) AS document_count,
             avg(coalesce(ri.relevance, 0.5)) AS avg_insight_relevance,
             avg(coalesce(rd.relevance, 0.5)) AS avg_doc_relevance,
             max(coalesce(ri.created_at, rd.created_at)) AS last_used
        RETURN t, insight_count, document_count,
               (avg_insight_relevance + avg_doc_relevance) / 2 AS avg_relevance,
               last_used
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, tag_id=tag_id)
            record = await result.single()

            if not record:
                return None

            node = record["t"]
            return TagStats(
                id=node["id"],
                name=node["name"],
                aliases=list(node.get("aliases", [])),
                color=node.get("color"),
                insight_count=record["insight_count"],
                document_count=record["document_count"],
                total_count=record["insight_count"] + record["document_count"],
                avg_relevance=record["avg_relevance"] or 0.0,
                last_used=datetime.fromisoformat(record["last_used"]) if record["last_used"] else None,
                created_at=datetime.fromisoformat(node["created_at"]),
            )

    async def get_all_tag_stats(self, limit: int = 100) -> list[TagStats]:
        """Get statistics for all tags.

        Args:
            limit: Maximum number of tags.

        Returns:
            List of tag statistics.
        """
        cypher = """
        MATCH (t:Tag)
        OPTIONAL MATCH (i:Insight)-[ri:HAS_TAG]->(t)
        OPTIONAL MATCH (d:Document)-[rd:HAS_TAG]->(t)
        WITH t, 
             count(DISTINCT i) AS insight_count,
             count(DISTINCT d) AS document_count,
             avg(coalesce(ri.relevance, 0.5)) AS avg_insight_relevance,
             avg(coalesce(rd.relevance, 0.5)) AS avg_doc_relevance,
             max(coalesce(ri.created_at, rd.created_at)) AS last_used
        RETURN t, insight_count, document_count,
               CASE WHEN insight_count + document_count > 0 
                    THEN (coalesce(avg_insight_relevance, 0) + coalesce(avg_doc_relevance, 0)) / 2
                    ELSE 0 END AS avg_relevance,
               last_used
        ORDER BY insight_count + document_count DESC
        LIMIT $limit
        """

        stats: list[TagStats] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, limit=limit)
            async for record in result:
                node = record["t"]
                stats.append(TagStats(
                    id=node["id"],
                    name=node["name"],
                    aliases=list(node.get("aliases", [])),
                    color=node.get("color"),
                    insight_count=record["insight_count"],
                    document_count=record["document_count"],
                    total_count=record["insight_count"] + record["document_count"],
                    avg_relevance=record["avg_relevance"] or 0.0,
                    last_used=datetime.fromisoformat(record["last_used"]) if record["last_used"] else None,
                    created_at=datetime.fromisoformat(node["created_at"]),
                ))

        return stats

    async def update_tag(
        self,
        tag_id: str,
        name: str | None = None,
        color: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
    ) -> Tag | None:
        """Update a tag.

        Args:
            tag_id: Tag ID.
            name: New name (optional).
            color: New color (optional).
            description: New description (optional).
            aliases: New aliases list (optional).

        Returns:
            Updated Tag if found, None otherwise.
        """
        set_clauses = ["t.updated_at = $updated_at"]
        params: dict[str, Any] = {
            "tag_id": tag_id,
            "updated_at": datetime.now().isoformat(),
        }

        if name is not None:
            set_clauses.append("t.name = $name")
            params["name"] = name.strip()

        if color is not None:
            set_clauses.append("t.color = $color")
            params["color"] = color

        if description is not None:
            set_clauses.append("t.description = $description")
            params["description"] = description

        if aliases is not None:
            set_clauses.append("t.aliases = $aliases")
            params["aliases"] = aliases

        cypher = f"""
        MATCH (t:Tag {{id: $tag_id}})
        SET {', '.join(set_clauses)}
        RETURN t
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()

            if not record:
                return None

            node = record["t"]
            logger.info(f"Updated tag: {tag_id}")
            return Tag(
                id=node["id"],
                name=node["name"],
                aliases=list(node.get("aliases", [])),
                color=node.get("color"),
                description=node.get("description", ""),
                usage_count=node.get("usage_count", 0),
                created_at=datetime.fromisoformat(node["created_at"]),
                updated_at=datetime.fromisoformat(node["updated_at"]),
            )

    async def delete_tag(self, tag_id: str) -> bool:
        """Delete a tag and its relationships.

        Args:
            tag_id: Tag ID.

        Returns:
            True if deleted, False if not found.
        """
        cypher = """
        MATCH (t:Tag {id: $tag_id})
        DETACH DELETE t
        RETURN count(t) AS deleted
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, tag_id=tag_id)
            record = await result.single()
            deleted = record["deleted"] > 0
            if deleted:
                logger.info(f"Deleted tag: {tag_id}")
            return deleted

    async def tag_insight(
        self,
        insight_id: str,
        tag_id: str,
        relevance: float = 0.8,
    ) -> None:
        """Add a tag to an insight with relevance score.

        Args:
            insight_id: Insight ID.
            tag_id: Tag ID.
            relevance: Relevance score (0.0-1.0).
        """
        cypher = """
        MATCH (i:Insight {id: $insight_id})
        MATCH (t:Tag {id: $tag_id})
        MERGE (i)-[r:HAS_TAG]->(t)
        SET r.relevance = $relevance,
            r.created_at = $created_at
        WITH t
        SET t.usage_count = t.usage_count + 1,
            t.updated_at = $created_at
        """

        async with self.driver.session() as session:
            await session.run(
                cypher,
                insight_id=insight_id,
                tag_id=tag_id,
                relevance=relevance,
                created_at=datetime.now().isoformat(),
            )
            logger.debug(f"Tagged insight {insight_id} with tag {tag_id} (relevance: {relevance})")

    async def tag_document(
        self,
        document_id: str,
        tag_id: str,
        relevance: float = 0.8,
    ) -> None:
        """Add a tag to a document with relevance score.

        Args:
            document_id: Document ID.
            tag_id: Tag ID.
            relevance: Relevance score (0.0-1.0).
        """
        cypher = """
        MATCH (d:Document {id: $document_id})
        MATCH (t:Tag {id: $tag_id})
        MERGE (d)-[r:HAS_TAG]->(t)
        SET r.relevance = $relevance,
            r.created_at = $created_at
        WITH t
        SET t.usage_count = t.usage_count + 1,
            t.updated_at = $created_at
        """

        async with self.driver.session() as session:
            await session.run(
                cypher,
                document_id=document_id,
                tag_id=tag_id,
                relevance=relevance,
                created_at=datetime.now().isoformat(),
            )
            logger.debug(f"Tagged document {document_id} with tag {tag_id} (relevance: {relevance})")

    async def untag_insight(self, insight_id: str, tag_id: str) -> bool:
        """Remove a tag from an insight.

        Args:
            insight_id: Insight ID.
            tag_id: Tag ID.

        Returns:
            True if relationship was removed, False otherwise.
        """
        cypher = """
        MATCH (i:Insight {id: $insight_id})-[r:HAS_TAG]->(t:Tag {id: $tag_id})
        DELETE r
        WITH t
        SET t.usage_count = CASE WHEN t.usage_count > 0 THEN t.usage_count - 1 ELSE 0 END
        RETURN 1 AS removed
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, insight_id=insight_id, tag_id=tag_id)
            record = await result.single()
            return record is not None

    async def untag_document(self, document_id: str, tag_id: str) -> bool:
        """Remove a tag from a document.

        Args:
            document_id: Document ID.
            tag_id: Tag ID.

        Returns:
            True if relationship was removed, False otherwise.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})-[r:HAS_TAG]->(t:Tag {id: $tag_id})
        DELETE r
        WITH t
        SET t.usage_count = CASE WHEN t.usage_count > 0 THEN t.usage_count - 1 ELSE 0 END
        RETURN 1 AS removed
        """

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id, tag_id=tag_id)
            record = await result.single()
            return record is not None

    async def get_items_by_tag(
        self,
        tag_id: str,
        item_type: TaggedItemType | None = None,
        limit: int = 50,
    ) -> list[TaggedItem]:
        """Get all items tagged with a specific tag.

        Args:
            tag_id: Tag ID.
            item_type: Filter by item type (optional).
            limit: Maximum number of items.

        Returns:
            List of tagged items with relevance scores.
        """
        items: list[TaggedItem] = []

        # Get insights
        if item_type is None or item_type == TaggedItemType.INSIGHT:
            insight_cypher = """
            MATCH (t:Tag {id: $tag_id})<-[r:HAS_TAG]-(i:Insight)
            MATCH (s:Entity)-[:SUBJECT_OF]->(i)-[:HAS_OBJECT]->(o:Entity)
            RETURN i.id AS id, r.relevance AS relevance, r.created_at AS tagged_at,
                   s.name AS subject, i.predicate AS predicate, o.name AS object,
                   i.created_at AS created_at
            ORDER BY r.relevance DESC
            LIMIT $limit
            """

            async with self.driver.session() as session:
                result = await session.run(insight_cypher, tag_id=tag_id, limit=limit)
                async for record in result:
                    items.append(TaggedItem(
                        item_id=record["id"],
                        item_type=TaggedItemType.INSIGHT,
                        relevance=record["relevance"] or 0.8,
                        title=f"{record['subject']} {record['predicate']}",
                        summary=record["object"],
                        created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else None,
                        subject=record["subject"],
                        predicate=record["predicate"],
                        object=record["object"],
                    ))

        # Get documents
        if item_type is None or item_type == TaggedItemType.DOCUMENT:
            doc_cypher = """
            MATCH (t:Tag {id: $tag_id})<-[r:HAS_TAG]-(d:Document)
            RETURN d.id AS id, r.relevance AS relevance, r.created_at AS tagged_at,
                   d.filename AS filename, d.extracted_summary AS summary,
                   d.created_at AS created_at
            ORDER BY r.relevance DESC
            LIMIT $limit
            """

            async with self.driver.session() as session:
                result = await session.run(doc_cypher, tag_id=tag_id, limit=limit)
                async for record in result:
                    items.append(TaggedItem(
                        item_id=record["id"],
                        item_type=TaggedItemType.DOCUMENT,
                        relevance=record["relevance"] or 0.8,
                        title=record["filename"],
                        summary=record["summary"] or "",
                        created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else None,
                        filename=record["filename"],
                    ))

        # Sort by relevance
        items.sort(key=lambda x: x.relevance, reverse=True)
        return items[:limit]

    async def get_tags_for_insight(self, insight_id: str) -> list[tuple[Tag, float]]:
        """Get all tags for an insight with their relevance scores.

        Args:
            insight_id: Insight ID.

        Returns:
            List of (Tag, relevance) tuples.
        """
        cypher = """
        MATCH (i:Insight {id: $insight_id})-[r:HAS_TAG]->(t:Tag)
        RETURN t, r.relevance AS relevance
        ORDER BY r.relevance DESC
        """

        tags: list[tuple[Tag, float]] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, insight_id=insight_id)
            async for record in result:
                node = record["t"]
                tag = Tag(
                    id=node["id"],
                    name=node["name"],
                    aliases=list(node.get("aliases", [])),
                    color=node.get("color"),
                    description=node.get("description", ""),
                    usage_count=node.get("usage_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                    updated_at=datetime.fromisoformat(node["updated_at"]),
                )
                tags.append((tag, record["relevance"] or 0.8))

        return tags

    async def get_tags_for_document(self, document_id: str) -> list[tuple[Tag, float]]:
        """Get all tags for a document with their relevance scores.

        Args:
            document_id: Document ID.

        Returns:
            List of (Tag, relevance) tuples.
        """
        cypher = """
        MATCH (d:Document {id: $document_id})-[r:HAS_TAG]->(t:Tag)
        RETURN t, r.relevance AS relevance
        ORDER BY r.relevance DESC
        """

        tags: list[tuple[Tag, float]] = []

        async with self.driver.session() as session:
            result = await session.run(cypher, document_id=document_id)
            async for record in result:
                node = record["t"]
                tag = Tag(
                    id=node["id"],
                    name=node["name"],
                    aliases=list(node.get("aliases", [])),
                    color=node.get("color"),
                    description=node.get("description", ""),
                    usage_count=node.get("usage_count", 0),
                    created_at=datetime.fromisoformat(node["created_at"]),
                    updated_at=datetime.fromisoformat(node["updated_at"]),
                )
                tags.append((tag, record["relevance"] or 0.8))

        return tags

    async def merge_tags(
        self,
        source_tag_ids: list[str],
        target_tag_id: str,
        add_as_aliases: bool = True,
    ) -> tuple[Tag | None, int]:
        """Merge multiple tags into a single tag.

        Args:
            source_tag_ids: IDs of tags to merge (will be deleted).
            target_tag_id: ID of target tag (will be kept).
            add_as_aliases: Whether to add source names as aliases.

        Returns:
            Tuple of (merged Tag, number of relationships moved).
        """
        if not source_tag_ids:
            return await self.get_tag(target_tag_id), 0

        # First, get source tag names for aliases
        source_names: list[str] = []
        if add_as_aliases:
            for source_id in source_tag_ids:
                source_tag = await self.get_tag(source_id)
                if source_tag:
                    source_names.append(source_tag.name)
                    source_names.extend(source_tag.aliases)

        # Move all HAS_TAG relationships from source tags to target
        move_cypher = """
        MATCH (source:Tag)
        WHERE source.id IN $source_ids
        MATCH (item)-[r:HAS_TAG]->(source)
        MATCH (target:Tag {id: $target_id})
        MERGE (item)-[new_r:HAS_TAG]->(target)
        SET new_r.relevance = coalesce(new_r.relevance, r.relevance),
            new_r.created_at = coalesce(new_r.created_at, r.created_at)
        DELETE r
        RETURN count(r) AS moved
        """

        relationships_moved = 0
        async with self.driver.session() as session:
            result = await session.run(
                move_cypher,
                source_ids=source_tag_ids,
                target_id=target_tag_id,
            )
            record = await result.single()
            relationships_moved = record["moved"] if record else 0

        # Update target tag with new aliases
        if add_as_aliases and source_names:
            target_tag = await self.get_tag(target_tag_id)
            if target_tag:
                # Merge aliases, avoiding duplicates
                existing_aliases = set(target_tag.aliases)
                existing_aliases.add(target_tag.name.lower())
                new_aliases = [
                    name for name in source_names
                    if name.lower() not in existing_aliases
                ]
                await self.update_tag(
                    target_tag_id,
                    aliases=target_tag.aliases + new_aliases,
                )

        # Delete source tags
        for source_id in source_tag_ids:
            await self.delete_tag(source_id)

        # Recalculate usage count for target
        count_cypher = """
        MATCH (t:Tag {id: $target_id})<-[:HAS_TAG]-(item)
        WITH t, count(item) AS usage
        SET t.usage_count = usage
        """
        async with self.driver.session() as session:
            await session.run(count_cypher, target_id=target_tag_id)

        merged_tag = await self.get_tag(target_tag_id)
        logger.info(f"Merged {len(source_tag_ids)} tags into {target_tag_id}, moved {relationships_moved} relationships")

        return merged_tag, relationships_moved

    async def search_tags(self, query: str, limit: int = 10) -> list[Tag]:
        """Search tags by name or alias.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching tags.
        """
        # Try fulltext search first
        try:
            cypher = """
            CALL db.index.fulltext.queryNodes('tag_search', $query)
            YIELD node, score
            WHERE node:Tag
            RETURN node AS t, score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            tags: list[Tag] = []
            async with self.driver.session() as session:
                result = await session.run(cypher, query=query, limit=limit)
                async for record in result:
                    node = record["t"]
                    tags.append(Tag(
                        id=node["id"],
                        name=node["name"],
                        aliases=list(node.get("aliases", [])),
                        color=node.get("color"),
                        description=node.get("description", ""),
                        usage_count=node.get("usage_count", 0),
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node["updated_at"]),
                    ))
            return tags

        except Exception as fulltext_err:
            logger.debug(f"Fulltext search failed, falling back to basic search: {fulltext_err}")
            # Fallback to basic search
            try:
                cypher = """
                MATCH (t:Tag)
                WHERE toLower(t.name) CONTAINS toLower($query)
                   OR any(alias IN coalesce(t.aliases, []) WHERE toLower(alias) CONTAINS toLower($query))
                RETURN t
                ORDER BY t.usage_count DESC
                LIMIT $limit
                """

                tags = []
                async with self.driver.session() as session:
                    result = await session.run(cypher, query=query, limit=limit)
                    async for record in result:
                        node = record["t"]
                        tags.append(Tag(
                            id=node["id"],
                            name=node["name"],
                            aliases=list(node.get("aliases", [])),
                            color=node.get("color"),
                            description=node.get("description", ""),
                            usage_count=node.get("usage_count", 0),
                            created_at=datetime.fromisoformat(node["created_at"]),
                            updated_at=datetime.fromisoformat(node["updated_at"]),
                        ))
                return tags
            except Exception as basic_err:
                logger.error(f"Basic tag search also failed: {basic_err}")
                raise

    async def get_or_create_tag(self, name: str) -> Tag:
        """Get an existing tag by name or create a new one.

        Args:
            name: Tag name.

        Returns:
            Existing or newly created Tag.
        """
        existing = await self.get_tag_by_name(name)
        if existing:
            return existing
        return await self.create_tag(name)

    # ==================== Knowledge Graph Visualization ====================

    async def get_knowledge_graph_data(
        self,
        min_usage_count: int = 0,
        limit: int = 100,
    ) -> KnowledgeGraphData:
        """Get knowledge graph data for visualization.

        Returns nodes (tags) and edges (co-occurrence relationships between tags).
        Tags that appear on the same Insight/Document are considered connected.

        Args:
            min_usage_count: Minimum usage count to include a tag.
            limit: Maximum number of nodes.

        Returns:
            KnowledgeGraphData with nodes and edges.
        """
        # Get all tags as nodes with their stats
        node_cypher = """
        MATCH (t:Tag)
        WHERE t.usage_count >= $min_usage_count
        OPTIONAL MATCH (i:Insight)-[:HAS_TAG]->(t)
        OPTIONAL MATCH (d:Document)-[:HAS_TAG]->(t)
        WITH t, count(DISTINCT i) AS insight_count, count(DISTINCT d) AS document_count
        RETURN t.id AS id,
               t.name AS name,
               t.color AS color,
               t.usage_count AS usage_count,
               insight_count,
               document_count
        ORDER BY t.usage_count DESC
        LIMIT $limit
        """

        nodes: list[KnowledgeGraphNode] = []
        node_ids: set[str] = set()

        async with self.driver.session() as session:
            result = await session.run(
                node_cypher,
                min_usage_count=min_usage_count,
                limit=limit,
            )
            async for record in result:
                node_id = record["id"]
                usage = record["usage_count"] or 0
                # Calculate weight based on usage (log scale for better visualization)
                weight = 1.0 + (usage ** 0.5) * 0.5 if usage > 0 else 1.0

                nodes.append(KnowledgeGraphNode(
                    id=node_id,
                    name=record["name"],
                    weight=weight,
                    color=record["color"],
                    usage_count=usage,
                    insight_count=record["insight_count"] or 0,
                    document_count=record["document_count"] or 0,
                ))
                node_ids.add(node_id)

        # Get co-occurrence edges (tags that appear on the same item)
        edge_cypher = """
        MATCH (t1:Tag)<-[:HAS_TAG]-(item)-[:HAS_TAG]->(t2:Tag)
        WHERE t1.id < t2.id
          AND t1.id IN $node_ids
          AND t2.id IN $node_ids
        WITH t1.id AS source, t2.id AS target, count(item) AS co_count
        WHERE co_count > 0
        RETURN source, target, co_count
        ORDER BY co_count DESC
        """

        edges: list[KnowledgeGraphEdge] = []

        if node_ids:
            async with self.driver.session() as session:
                result = await session.run(edge_cypher, node_ids=list(node_ids))
                async for record in result:
                    co_count = record["co_count"]
                    # Weight based on co-occurrence count
                    weight = 1.0 + (co_count ** 0.5) * 0.3

                    edges.append(KnowledgeGraphEdge(
                        source=record["source"],
                        target=record["target"],
                        weight=weight,
                        co_occurrence_count=co_count,
                    ))

        logger.info(f"Knowledge graph: {len(nodes)} nodes, {len(edges)} edges")

        return KnowledgeGraphData(
            nodes=nodes,
            edges=edges,
            total_tags=len(nodes),
            total_connections=len(edges),
        )
