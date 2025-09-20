from __future__ import annotations

from typing import Any, Dict, Optional

from neo4j import GraphDatabase, Driver


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def add_node(self, label: str, properties: Dict[str, Any]) -> None:
        query = f"CREATE (n:{label} $props)"
        with self._driver.session() as session:
            session.run(query, props=properties)


