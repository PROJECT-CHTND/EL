"""Knowledge Graph Store using Neo4j."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from el_core.schemas import (
    Domain,
    FactStatus,
    FactVersion,
    FactWithHistory,
    Insight,
    KnowledgeItem,
    SessionMetadata,
    SessionSummary,
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
            created_at: $created_at
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
            )
            record = await result.single()
            return record["id"] if record else insight_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        domain: Domain | None = None,
    ) -> list[KnowledgeItem]:
        """Search the knowledge graph for relevant insights.

        Args:
            query: Search query (keywords or concepts).
            limit: Maximum number of results.
            domain: Optional domain filter.

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
            score
        ORDER BY score DESC
        LIMIT $max_results
        """

        params: dict[str, Any] = {"search_term": query, "max_results": limit}
        if domain:
            params["domain_filter"] = domain.value

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
                        )
                    )
        except Exception as e:
            # If full-text index doesn't exist, fall back to basic search
            logger.warning(f"Full-text search failed, using basic search: {e}")
            items = await self._basic_search(query, limit, domain)

        return items

    async def _basic_search(
        self,
        query: str,
        limit: int,
        domain: Domain | None,
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

        cypher += """
        RETURN 
            i.id AS id,
            s.name AS subject,
            i.predicate AS predicate,
            o.name AS object,
            i.confidence AS confidence,
            i.domain AS domain,
            i.created_at AS created_at
        ORDER BY i.confidence DESC
        LIMIT $max_results
        """

        params: dict[str, Any] = {"search_term": query, "max_results": limit}
        if domain:
            params["domain_filter"] = domain.value

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
            created_at: i1.created_at
        }) + collect({
            id: i2.id,
            subject: s2.name,
            predicate: i2.predicate,
            object: e.name,
            confidence: i2.confidence,
            domain: i2.domain,
            created_at: i2.created_at
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
            insight.created_at AS created_at
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
            i.created_at AS created_at
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

    async def setup_indexes(self) -> None:
        """Set up required indexes for the knowledge graph."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX insight_id IF NOT EXISTS FOR (i:Insight) ON (i.id)",
            "CREATE INDEX insight_domain IF NOT EXISTS FOR (i:Insight) ON (i.domain)",
            "CREATE INDEX insight_session IF NOT EXISTS FOR (i:Insight) ON (i.session_id)",
            "CREATE INDEX insight_status IF NOT EXISTS FOR (i:Insight) ON (i.status)",
            "CREATE INDEX session_id IF NOT EXISTS FOR (s:Session) ON (s.id)",
            "CREATE INDEX session_user IF NOT EXISTS FOR (s:Session) ON (s.user_id)",
            "CREATE INDEX session_status IF NOT EXISTS FOR (s:Session) ON (s.status)",
            "CREATE INDEX summary_id IF NOT EXISTS FOR (sum:Summary) ON (sum.id)",
            "CREATE INDEX fact_version_id IF NOT EXISTS FOR (v:FactVersion) ON (v.id)",
            """
            CREATE FULLTEXT INDEX insight_search IF NOT EXISTS 
            FOR (i:Insight) ON EACH [i.predicate]
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
