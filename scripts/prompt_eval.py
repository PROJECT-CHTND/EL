#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# Ensure we can import el_agent from the monorepo layout (el-agent/src)
REPO_ROOT = Path(__file__).resolve().parents[1]
EL_AGENT_SRC = REPO_ROOT / "el-agent" / "src"
if str(EL_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(EL_AGENT_SRC))

from el_agent.llm.client import OpenAILLM  # type: ignore  # noqa: E402
from el_agent.schemas import Hypothesis  # type: ignore  # noqa: E402


@dataclass
class EvalResult:
    stamp: str
    prompt: str
    case_path: str
    latency_sec: float
    payload: Dict[str, Any]


class DummyLLM:
    """Fallback LLM used when OpenAI is not configured.

    Implements the same surface used here to avoid external calls.
    """

    def hypothesis_candidates(self, context: str, k: int) -> List[str]:
        base = [s for s in context.replace("\n", " ").split("。") if s.strip()][:k]
        if not base:
            base = [f"仮説{i+1}" for i in range(max(1, k))]
        return base[:k]

    def extract_evidence(self, sentences: List[str], hypothesis: str) -> Dict[str, Any]:
        supports = []
        for idx, s in enumerate(sentences[:3]):
            supports.append(
                {
                    "hypothesis": hypothesis,
                    "polarity": "+",
                    "span": s[:120],
                    "score_raw": 0.5,
                    "source": f"dummy:{idx}",
                    "time": 0,
                }
            )
        return {
            "entities": [],
            "relations": [],
            "supports": supports,
            "feature_vector": {"cosine": 0.5, "source_trust": 0.5, "recency": 0.5, "logic_ok": 1},
        }

    def generate_question(self, hypothesis: str) -> str:
        h = hypothesis.strip().rstrip("。")
        return f"{h}か確認できますか？"

    def synthesize_doc(self, confirmed: List[Hypothesis], kg_delta: Dict[str, Any]) -> str:
        lines = ["# 合成レポート", "", "## 確定仮説"]
        for h in confirmed:
            lines.append(f"- {h.id}: {h.text} (belief={h.belief:.2f})")
        lines.append("\n## KG差分")
        lines.append(json.dumps(kg_delta, ensure_ascii=False))
        return "\n".join(lines)

    def qa_refine(self, question: str, hypothesis: str) -> Dict[str, Any]:
        q = question.strip()
        if not q.endswith("？") and not q.endswith("?"):
            q = q.rstrip("。") + "？"
        return {
            "question": q,
            "reasons": ["フォールバックで形式を整えました"],
            "checks": {"is_binary": False, "has_assumptions": False, "is_minimal": True},
        }


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _init_llm() -> Any:
    try:
        # Use OpenAI if API key is provided
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAILLM(api_key=api_key)
        # Fallback when not provided
        return DummyLLM()
    except Exception:
        return DummyLLM()


def _run_prompt(prompt: str, case: Dict[str, Any]) -> Dict[str, Any]:
    llm = _init_llm()
    if prompt == "hypothesis":
        context = str(case.get("context", ""))
        k = int(case.get("k", 5))
        cands = llm.hypothesis_candidates(context=context, k=k)
        return {"candidates": cands}
    if prompt == "evidence":
        sentences = list(case.get("sentences", []))
        hypothesis = str(case.get("hypothesis", ""))
        ev = llm.extract_evidence(sentences=sentences, hypothesis=hypothesis)
        return ev
    if prompt == "question":
        hypothesis = str(case.get("hypothesis", ""))
        q = llm.generate_question(hypothesis=hypothesis)
        return {"question": q}
    if prompt == "question_qc":
        question = str(case.get("question", ""))
        hypothesis = str(case.get("hypothesis", ""))
        out = llm.qa_refine(question=question, hypothesis=hypothesis)
        return out
    if prompt == "doc_synth":
        confirmed_raw = list(case.get("confirmed", []))
        kg_delta = dict(case.get("kg_delta", {}))
        confirmed: List[Hypothesis] = []
        for item in confirmed_raw:
            try:
                confirmed.append(Hypothesis(**item))
            except Exception:
                # Minimal repair: attempt to coerce
                minimal = {"id": str(item.get("id", "h")), "text": str(item.get("text", ""))}
                confirmed.append(Hypothesis(**minimal))
        md = llm.synthesize_doc(confirmed=confirmed, kg_delta=kg_delta)
        return {"markdown": md}
    raise ValueError(f"Unknown prompt: {prompt}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single prompt E2E and append a scoring row.")
    parser.add_argument("--prompt", required=True, choices=[
        "hypothesis",
        "evidence",
        "question",
        "question_qc",
        "doc_synth",
    ])
    parser.add_argument("--case", required=True, help="Path to case JSON")
    parser.add_argument("--out", default="reports/prompt_runs", help="Output directory (default: reports/prompt_runs)")

    args = parser.parse_args(argv)
    case_path = Path(args.case)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    with case_path.open("r", encoding="utf-8") as f:
        case_data = json.load(f)

    stamp = _now_stamp()
    t0 = time.perf_counter()
    payload: Dict[str, Any] = {}
    try:
        payload = _run_prompt(args.prompt, case_data)
        ok = True
    except Exception as e:
        # Capture error in payload for debugging, but still write files
        payload = {"error": str(e)}
        ok = False
    latency = time.perf_counter() - t0

    # Save JSON output
    out_json = out_dir / f"{args.prompt}_{stamp}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stamp": stamp,
                "prompt": args.prompt,
                "case": str(case_path),
                "latency_sec": round(latency, 3),
                "ok": ok,
                "output": payload,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Append scoring CSV row
    scores_csv = out_dir / "scores.csv"
    new_file = not scores_csv.exists()
    with scores_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["stamp", "prompt", "case", "latency", "ok", "form_ng", "meaning_ng", "notes"])
        writer.writerow([stamp, args.prompt, str(case_path), f"{latency:.3f}", int(ok), "", "", ""])  # placeholders

    # Print minimal status to stdout
    print(f"saved: {out_json}")
    print(f"updated: {scores_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


