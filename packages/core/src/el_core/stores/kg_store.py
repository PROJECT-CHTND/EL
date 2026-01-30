"""Knowledge Graph Store using Neo4j."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from el_core.schemas import (
    DateType,
    Document,
    DocumentChunk,
    DocumentStatus,
    Domain,
    FactStatus,
    FactVersion,
    FactWithHistory,
    Insight,
    KnowledgeItem,
    KnowledgeStats,
    SessionMetadata,
    SessionSummary,
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

        Args:
            query: Search query (keywords or concepts).
            limit: Maximum number of results.
            domain: Optional domain filter.
            start_date: Optional start date filter (filters by event_date).
            end_date: Optional end date filter (filters by event_date).

        Returns:
            List of matching knowledge items.
        """
        # Full-text search across subject, predicate, and object
        cypher = """
        CALL db.index.fulltext.queryNodes('insight_search', $search_term)
        YIELD node, score
        WHERE node:Insight
        """

        if domain:
            cypher += " AND node.domain = $domain_filter"
        
        if start_date:
            cypher += " AND (node.event_date IS NULL OR node.event_date >= $start_date)"
        
        if end_date:
            cypher += " AND (node.event_date IS NULL OR node.event_date <= $end_date)"

        cypher += """
        MATCH (s:Entity)-[:SUBJECT_OF]->(node)-[:HAS_OBJECT]->(o:Entity)
        RETURN 
            node.id AS id,
            s.name AS subject,
            node.predicate AS predicate,
            o.name AS object,
            node.confidence AS confidence,
            node.domain AS domain,
            node.created_at AS created_at,
            node.event_date AS event_date,
            node.event_date_end AS event_date_end,
            node.date_type AS date_type,
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

        items: list[KnowledgeItem] = []

        try:
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
        except Exception as e:
            # If full-text index doesn't exist, fall back to basic search
            logger.warning(f"Full-text search failed, using basic search: {e}")
            items = await self._basic_search(query, limit, domain, start_date, end_date)

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
