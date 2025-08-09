#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# Ensure we can import el_agent from repo layout (el-agent/src)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
EL_AGENT_SRC = REPO_ROOT / "el-agent" / "src"
if str(EL_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(EL_AGENT_SRC))


from el_agent.llm import OpenAILLM, prompts  # type: ignore  # noqa: E402
from el_agent.schemas import Hypothesis  # type: ignore  # noqa: E402


@dataclass
class RunResult:
    variant: str
    latency_s: float
    output: Any


def _read_case(case_path: Path) -> Any:
    text = case_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        return text


def _run_prompt(llm: OpenAILLM, prompt_name: str, case_data: Any) -> Any:
    name = prompt_name.strip().lower()

    # question → generate_question(hypothesis)
    if name == "question":
        if isinstance(case_data, dict) and isinstance(case_data.get("hypothesis"), str):
            hyp = case_data["hypothesis"]
        elif isinstance(case_data, str):
            hyp = case_data
        else:
            raise ValueError("case must provide 'hypothesis' (str) for prompt 'question'")
        return llm.generate_question(hypothesis=hyp)

    # hypothesis → hypothesis_candidates(context, k)
    if name in ("hypothesis", "hypothesis_candidates"):
        if not isinstance(case_data, dict):
            raise ValueError("case must be a JSON object with 'context' and optional 'k'")
        context = str(case_data.get("context", ""))
        k = int(case_data.get("k", 3))
        return llm.hypothesis_candidates(context=context, k=k)

    # extract → extract_evidence(sentences, hypothesis)
    if name in ("extract", "extract_evidence"):
        if not isinstance(case_data, dict):
            raise ValueError("case must be a JSON object with 'sentences' (list[str]) and 'hypothesis' (str)")
        sentences = list(case_data.get("sentences", []))
        hypothesis = str(case_data.get("hypothesis", ""))
        return llm.extract_evidence(sentences=sentences, hypothesis=hypothesis)

    # synthesize → synthesize_doc(confirmed, kg_delta)
    if name in ("synthesize", "synthesize_doc"):
        if not isinstance(case_data, dict):
            raise ValueError("case must be a JSON object with 'confirmed' (list[Hypothesis-like]) and 'kg_delta' (dict)")
        confirmed_raw = list(case_data.get("confirmed", []))
        confirmed = []
        for obj in confirmed_raw:
            try:
                confirmed.append(Hypothesis.model_validate(obj))
            except Exception:
                # fallback minimal
                confirmed.append(Hypothesis(id=obj.get("id", "h"), text=obj.get("text", "")))
        kg_delta = dict(case_data.get("kg_delta", {}))
        return llm.synthesize_doc(confirmed=confirmed, kg_delta=kg_delta)

    # question_strategist → question_strategist(Hypothesis)
    if name in ("question_strategist", "strategist"):
        if not isinstance(case_data, dict):
            raise ValueError("case must be a JSON object compatible with Hypothesis")
        hyp = Hypothesis.model_validate(case_data)
        return llm.question_strategist(hypothesis=hyp)

    # qa_refine → qa_refine(question, hypothesis)
    if name in ("qa_refine", "refine"):
        if not isinstance(case_data, dict):
            raise ValueError("case must be a JSON object with 'question' and 'hypothesis'")
        q = str(case_data.get("question", ""))
        h = str(case_data.get("hypothesis", ""))
        return llm.qa_refine(question=q, hypothesis=h)

    raise ValueError(f"Unsupported prompt name: {prompt_name}")


def run_variant(prompt_name: str, variant: str, case_data: Any) -> RunResult:
    # Switch prompt variant
    prompts.set_variant(prompt_name, variant)

    # Single LLM client (prompt constants are read at call-time)
    llm = OpenAILLM()

    t0 = time.perf_counter()
    output = _run_prompt(llm=llm, prompt_name=prompt_name, case_data=case_data)
    latency = time.perf_counter() - t0
    return RunResult(variant=variant, latency_s=latency, output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt A/B test runner")
    parser.add_argument("--prompt", required=True, help="prompt name (e.g., question)")
    parser.add_argument("--case", required=True, help="path to case JSON/text file")
    parser.add_argument("--A", dest="variant_a", required=True, help="variant name for A (e.g., v1)")
    parser.add_argument("--B", dest="variant_b", required=True, help="variant name for B (e.g., v2)")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "ab"), help="output directory")

    args = parser.parse_args()

    prompt_name: str = args.prompt
    case_path = Path(args.case)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_data = _read_case(case_path)

    # Run A and B sequentially
    res_a = run_variant(prompt_name=prompt_name, variant=args.variant_a, case_data=case_data)
    res_b = run_variant(prompt_name=prompt_name, variant=args.variant_b, case_data=case_data)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"ab_{prompt_name}_{ts}.json"

    payload: Dict[str, Any] = {
        "prompt": prompt_name,
        "case_path": str(case_path),
        "timestamp": ts,
        "A": {
            "variant": res_a.variant,
            "latency_s": round(res_a.latency_s, 6),
            "output": res_a.output,
        },
        "B": {
            "variant": res_b.variant,
            "latency_s": round(res_b.latency_s, 6),
            "output": res_b.output,
        },
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()


