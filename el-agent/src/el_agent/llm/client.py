from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..schemas import Hypothesis
from . import prompts


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

    def hypothesis_candidates(self, context: str, k: int) -> List[str]:
        user = json.dumps({"k": int(k), "context": context})
        out = self._chat(prompts.HYPOTHESIS_CANDIDATES_SYS, user, temperature=0.7)
        try:
            data = json.loads(out)
            cands = data.get("candidates", [])
            return [str(x) for x in cands][:k]
        except Exception:
            # Fallback: naive line split
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return lines[:k]

    def extract_evidence(self, sentences: List[str], hypothesis: str) -> Dict[str, Any]:
        user = json.dumps({"hypothesis": hypothesis, "sentences": sentences})
        out = self._chat(prompts.EXTRACT_EVIDENCE_SYS, user, temperature=0.2)
        try:
            data = json.loads(out)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {"entities": [], "relations": [], "supports": [], "feature_vector": {}}

    def generate_question(self, hypothesis: str) -> str:
        user = f"仮説: {hypothesis}"
        out = self._chat(prompts.GENERATE_QUESTION_SYS, user, temperature=0.3)
        # one sentence JA
        return out.strip().splitlines()[0] if out else ""

    def synthesize_doc(self, confirmed: List[Hypothesis], kg_delta: Dict[str, Any]) -> str:
        payload = {
            "confirmed": [h.model_dump() for h in confirmed],
            "kg_delta": kg_delta,
        }
        out = self._chat(prompts.SYNTHESIZE_DOC_SYS, json.dumps(payload), temperature=0.2)
        return out.strip()


