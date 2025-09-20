from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..schemas import Hypothesis
from . import prompts
from ..monitoring.metrics import LLM_CALLS


class LLM:
    def hypothesis_candidates(self, context: str, k: int) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_evidence(self, sentences: List[str], hypothesis: str) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def generate_question(self, hypothesis: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def synthesize_doc(self, confirmed: List[Hypothesis], kg_delta: Dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAILLM(LLM):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return res.choices[0].message.content or ""

    def _json_payload(self, payload: Any) -> str:
        # ensure_ascii=False to preserve Japanese
        return json.dumps(payload, ensure_ascii=False)

    def _parse_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            # Try to extract first JSON object if accidental wrappers exist
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None

    def hypothesis_candidates(self, context: str, k: int) -> List[str]:
        user = self._json_payload({"k": int(k), "context": context})
        LLM_CALLS.labels(kind="hypothesis_candidates").inc()
        out = self._chat(prompts.HYPOTHESIS_CANDIDATES_SYS, user, temperature=0.7)
        data = self._parse_json(out)
        if isinstance(data, dict):
            cands = data.get("candidates", [])
            return [str(x) for x in cands][:k]
        # Fallback: naive line split
        lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
        return lines[:k]

    def extract_evidence(self, sentences: List[str], hypothesis: str) -> Dict[str, Any]:
        user = self._json_payload({"hypothesis": hypothesis, "sentences": sentences})
        LLM_CALLS.labels(kind="extract_evidence").inc()
        out = self._chat(prompts.EXTRACT_EVIDENCE_SYS, user, temperature=0.2)
        data = self._parse_json(out)
        if isinstance(data, dict):
            return data
        return {"entities": [], "relations": [], "supports": [], "feature_vector": {}}

    def generate_question(self, hypothesis: str) -> str:
        user = self._json_payload({"hypothesis": hypothesis})
        LLM_CALLS.labels(kind="generate_question").inc()
        out = self._chat(prompts.GENERATE_QUESTION_SYS, user, temperature=0.3)
        data = self._parse_json(out)
        if isinstance(data, dict) and isinstance(data.get("question"), str):
            return data["question"].strip()
        return (out or "").strip().splitlines()[0] if out else ""

    def synthesize_doc(self, confirmed: List[Hypothesis], kg_delta: Dict[str, Any]) -> str:
        payload = {
            "confirmed": [h.model_dump() for h in confirmed],
            "kg_delta": kg_delta,
        }
        LLM_CALLS.labels(kind="synthesize_doc").inc()
        out = self._chat(prompts.SYNTHESIZE_DOC_SYS, self._json_payload(payload), temperature=0.2)
        data = self._parse_json(out)
        if isinstance(data, dict) and isinstance(data.get("markdown"), str):
            return data["markdown"].strip()
        return (out or "").strip()

    # New helpers
    def question_strategist(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        payload = {"hypothesis": hypothesis.model_dump()}
        LLM_CALLS.labels(kind="question_strategist").inc()
        out = self._chat(prompts.QUESTION_STRATEGIST_SYS, self._json_payload(payload), temperature=0.2)
        data = self._parse_json(out)
        if isinstance(data, dict):
            return data
        return {"action": "none", "question": None, "expected_gain": 0.0, "estimated_cost": 0.0, "stop_rule_hit": True}

    def qa_refine(self, question: str, hypothesis: str) -> Dict[str, Any]:
        payload = {"question": question, "hypothesis": hypothesis}
        LLM_CALLS.labels(kind="qa_refine").inc()
        out = self._chat(prompts.QA_REFINE_SYS, self._json_payload(payload), temperature=0.2)
        data = self._parse_json(out)
        if isinstance(data, dict):
            return data
        return {"question": question, "reasons": [], "checks": {"is_binary": False, "has_assumptions": False, "is_minimal": False}}


