from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path
import json

from .evaluator import Evaluator
from .knowledge_integrator import KnowledgeIntegrator
from .strategist import Strategist
from ..schemas import Evidence, Hypothesis
from ..utils.pii import mask_structure, user_id_hash


class Orchestrator:
    def __init__(self) -> None:
        self.strategist = Strategist()
        self.integrator = KnowledgeIntegrator()
        self.evaluator = Evaluator()

    def run(self, goal: str) -> float:
        _plan = self.strategist.plan(goal)
        write_wal_line("orchestrator.plan", {"goal": goal, "steps": len(_plan)})
        hypothesis = Hypothesis(id="h-1", text=goal)
        evidences: List[Evidence] = []
        hypothesis = self.integrator.integrate(hypothesis, evidences)
        score = self.evaluator.score(hypothesis, evidences)
        write_wal_line(
            "orchestrator.result",
            {
                "goal": goal,
                "hypothesis": hypothesis.model_dump(),
                "evidence_count": len(evidences),
                "score": score,
            },
        )
        return score


def _wal_dir() -> Path:
    here = Path(__file__).resolve()
    # repo root (EL/) if possible; fallback to current dir
    repo_root = here.parents[4] if len(here.parents) >= 5 else here.parents[0]
    wal_dir = repo_root / "logs" / "wal"
    wal_dir.mkdir(parents=True, exist_ok=True)
    return wal_dir


def write_wal_line(event: str, data: Dict[str, Any] | None = None) -> None:
    try:
        ts = datetime.utcnow().isoformat() + "Z"
        fname = datetime.utcnow().strftime("%Y-%m-%d.log")
        path = _wal_dir() / fname
        data_in: Dict[str, Any] = dict(data or {})
        # Derive user_id_hash from session_id if present, then drop raw session_id
        sid = data_in.pop("session_id", None)
        uid_hash = user_id_hash(str(sid)) if sid is not None else None
        masked_data = mask_structure(data_in)
        record: Dict[str, Any] = {"ts": ts, "event": str(event), "data": masked_data}
        if uid_hash:
            record["user_id_hash"] = uid_hash
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # WAL は観測目的のため、本体処理を阻害しない
        pass


