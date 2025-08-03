#!/usr/bin/env python
"""Proof-of-Concept runner for EL pipeline.

Usage::

    python scripts/pipeline_cli.py input.txt --focus "AI" --temp 0.3

Requires OPENAI_API_KEY (and optionally Neo4j / Redis envs) to be set.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agent.pipeline.stage01_context import generate_context
from agent.pipeline.stage02_extract import extract_knowledge
from agent.pipeline.stage04_slots import propose_slots
from agent.pipeline.stage06_qgen import generate_questions
from agent.pipeline.stage07_qcheck import return_validated_questions


async def run_pipeline(text: str, focus: str | None, temperature: float):  # noqa: D401
    print("===== Stage01: Hierarchical Context =====")
    ctx = await generate_context(text)
    print(json.dumps(ctx.model_dump(), ensure_ascii=False, indent=2))

    print("\n===== Stage02: Knowledge Extraction =====")
    kg = await extract_knowledge(text, focus=focus, temperature=temperature)
    print(json.dumps(kg.model_dump(), ensure_ascii=False, indent=2))

    print("\n===== Stage04: Slot Discovery =====")
    slots = await propose_slots(kg)
    for s in slots:
        print(f"- {s.name}: {s.description}")

    print("\n===== Stage06/07: Question Generation & QA =====")
    if slots:
        qs = await generate_questions(slots)
        qs = await return_validated_questions(qs)
        for q in qs:
            print(f"Q: {q.text} (spec={q.specificity:.2f}, tacit={q.tacit_power:.2f})")
    else:
        print("No slots proposed – skipping question generation.")


def main() -> None:  # noqa: D401
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Input text file")
    ap.add_argument("--focus", default=None, help="Extraction focus keyword")
    ap.add_argument("--temp", type=float, default=0.0, help="LLM temperature")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    asyncio.run(run_pipeline(text, args.focus, args.temp))


if __name__ == "__main__":
    main() 