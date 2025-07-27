from __future__ import annotations

import os
from typing import Iterable

from neo4j import GraphDatabase, Driver, Transaction  # type: ignore

from agent.models.kg import Entity, Relation, KGPayload


class Neo4jClient:
    """Minimal Neo4j driver wrapper for KG insertion."""

    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD environment variable is required")

        try:
            self._driver: Driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                encrypted=os.getenv("NEO4J_ENCRYPTED", "true").lower() == "true"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    @staticmethod
    def _merge_entities(tx: Transaction, entities: Iterable[Entity]) -> None:
        entities_list = list(entities)
        if not entities_list:
            return

        query = (
            "UNWIND $entities AS entity "
            "MERGE (e:Entity {id: entity.id}) "
            "SET e.label = entity.label, e.type = entity.type, e.confidence = entity.confidence"
        )

        entities_data = [
            {
                "id": ent.id,
                "label": ent.label,
                "type": ent.type,
                "confidence": ent.confidence,
            }
            for ent in entities_list
        ]

        tx.run(query, entities=entities_data)
    @staticmethod
    def _merge_relations(tx: Transaction, relations: Iterable[Relation]) -> None:
        relations_list = list(relations)
        if not relations_list:
            return

        query = (
            "UNWIND $relations AS rel "
            "MATCH (s:Entity {id: rel.source}), (t:Entity {id: rel.target}) "
            "MERGE (s)-[r:RELATION {type: rel.type}]->(t) "
            "SET r.confidence = rel.confidence"
        )

        relations_data = [
            {
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "confidence": rel.confidence,
            }
            for rel in relations_list
        ]

        tx.run(query, relations=relations_data)
                type=ent.type,
                confidence=ent.confidence,
            )

    @staticmethod
    def _merge_relations(tx: Transaction, relations: Iterable[Relation]) -> None:
        query = (
            "MATCH (s:Entity {id: $source}), (t:Entity {id: $target}) "
            "MERGE (s)-[r:RELATION {type: $type}]->(t) "
            "SET r.confidence = $confidence"
        )
        for rel in relations:
            tx.run(
                query,
                source=rel.source,
                target=rel.target,
                type=rel.type,
                confidence=rel.confidence,
            ) 