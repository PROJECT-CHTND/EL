from __future__ import annotations

import os
from typing import Iterable

from neo4j import GraphDatabase, Driver, ManagedTransaction  # type: ignore

from agent.models.kg import Entity, Relation, KGPayload
from agent.models.fact import Fact, FactIn, Approval


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
    def _merge_entities(tx: ManagedTransaction, entities: Iterable[Entity]) -> None:
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
    def _merge_relations(tx: ManagedTransaction, relations: Iterable[Relation]) -> None:
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

    def ingest(self, payload: KGPayload) -> None:
        """Ingest a KGPayload into Neo4j."""

        with self._driver.session() as session:
            session.execute_write(self._merge_entities, payload.entities)  # type: ignore[arg-type]
            session.execute_write(self._merge_relations, payload.relations)  # type: ignore[arg-type] 

    # --- Fact approval workflow ---
    @staticmethod
    def _create_tentative_fact(tx: ManagedTransaction, fact: FactIn, assigned_id: str) -> Fact:
        query = (
            "MERGE (f:Fact {id: $id}) "
            "SET f.text = $text, f.belief = $belief, f.impact = $impact, f.status = $status "
            "WITH f, $status AS st "
            "FOREACH (_ IN CASE WHEN st = 'confirmed' THEN [1] ELSE [] END | SET f:ConfirmedFact) "
            "RETURN f.id AS id, f.text AS text, f.belief AS belief, f.impact AS impact, f.status AS status"
        )
        rec = tx.run(
            query,
            id=assigned_id,
            text=fact.text,
            belief=fact.belief,
            impact=fact.impact,
            status="pending" if fact.impact == "critical" else "confirmed",
        ).single()
        if rec is None:  # pragma: no cover
            raise RuntimeError("Failed to create Fact")
        return Fact(
            id=rec["id"],
            text=rec["text"],
            belief=float(rec["belief"]),
            impact=rec["impact"],
            status=rec["status"],
        )

    @staticmethod
    def _create_approval_node(tx: ManagedTransaction, fact_id: str, approval_id: str) -> Approval:
        query = (
            "MATCH (f:Fact {id: $fact_id}) "
            "MERGE (a:Approval {id: $approval_id}) "
            "SET a.fact_id = $fact_id, a.decision = 'pending', a.ts = datetime().toString() "
            "MERGE (f)-[:REQUIRES_APPROVAL]->(a) "
            "RETURN a.id AS id, a.fact_id AS fact_id, a.decision AS decision, a.ts AS ts"
        )
        rec = tx.run(query, fact_id=fact_id, approval_id=approval_id).single()
        if rec is None:  # pragma: no cover
            raise RuntimeError("Failed to create Approval")
        return Approval(id=rec["id"], fact_id=rec["fact_id"], approver=None, ts=rec["ts"], decision=rec["decision"])  # type: ignore[arg-type]

    def submit_fact(self, fact: FactIn) -> Fact:
        """Submit a fact. Critical facts become pending with an Approval node; normal facts confirm immediately."""
        import uuid

        assigned_id = fact.id or str(uuid.uuid4())
        with self._driver.session() as session:
            created = session.execute_write(self._create_tentative_fact, fact, assigned_id)
            if created.impact == "critical":
                _ = session.execute_write(self._create_approval_node, created.id, str(uuid.uuid4()))
            # If confirmed immediately, also commit to confirmed KG graph projection if applicable
            return created

    @staticmethod
    def _apply_approval(tx: ManagedTransaction, fact_id: str, decision: str, approver: str | None) -> Fact:
        # Update approval node and fact status
        query = (
            "MATCH (f:Fact {id: $fact_id}) "
            "OPTIONAL MATCH (f)-[:REQUIRES_APPROVAL]->(a:Approval) "
            "WITH f, a, $decision AS dec "
            "SET f.status = CASE dec WHEN 'approve' THEN 'confirmed' WHEN 'reject' THEN 'rejected' ELSE f.status END "
            "FOREACH (_ IN CASE WHEN dec = 'approve' THEN [1] ELSE [] END | SET f:ConfirmedFact) "
            "FOREACH (_ IN CASE WHEN dec = 'reject' THEN [1] ELSE [] END | REMOVE f:ConfirmedFact) "
            "FOREACH (_ IN CASE WHEN a IS NULL THEN [] ELSE [1] END | SET a.decision = dec, a.ts = datetime().toString(), a.approver = $approver) "
            "RETURN f.id AS id, f.text AS text, f.belief AS belief, f.impact AS impact, f.status AS status"
        )
        rec = tx.run(query, fact_id=fact_id, decision=decision, approver=approver).single()
        if rec is None:
            raise ValueError("Fact not found")
        return Fact(
            id=rec["id"],
            text=rec["text"],
            belief=float(rec["belief"]),
            impact=rec["impact"],
            status=rec["status"],
        )

    def approve_fact(self, fact_id: str, decision: str, approver: str | None = None) -> Fact:
        with self._driver.session() as session:
            updated = session.execute_write(self._apply_approval, fact_id, decision, approver)
            return updated