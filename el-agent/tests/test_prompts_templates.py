from __future__ import annotations

import re

from el_agent.llm import prompts as P  # type: ignore


def assert_json_only(template: str) -> None:
    # No triple backticks or language fences
    assert "```" not in template
    # Should mention 出力 or スキーマ and not include example code fences
    # Ensure no prefix like "Please respond with" English boilerplate is required
    # Core acceptance: templates must instruct JSON-only and avoid non-JSON wrappers
    required_phrases = ["出力はJSONのみ", "スキーマ"]
    for phrase in required_phrases:
        assert phrase in template


def test_hypothesis_candidates_template_schema_keys():
    t = P.HYPOTHESIS_CANDIDATES_SYS
    assert_json_only(t)
    # Must mention keys
    assert '"k"' in t or 'k' in t
    assert '"context"' in t or 'context' in t
    assert '"candidates"' in t or 'candidates' in t


def test_extract_evidence_template_schema_keys():
    t = P.EXTRACT_EVIDENCE_SYS
    assert_json_only(t)
    for key in ["entities", "relations", "supports", "feature_vector", "polarity", "cosine", "source_trust", "recency", "logic_ok"]:
        assert key in t


def test_generate_question_template_schema_keys():
    t = P.GENERATE_QUESTION_SYS
    assert_json_only(t)
    assert "hypothesis" in t
    assert "question" in t


def test_synthesize_doc_template_schema_keys():
    t = P.SYNTHESIZE_DOC_SYS
    assert_json_only(t)
    for key in ["confirmed", "kg_delta", "markdown"]:
        assert key in t


def test_question_strategist_template_schema_keys():
    t = P.QUESTION_STRATEGIST_SYS
    assert_json_only(t)
    for key in ["action", "question", "expected_gain", "estimated_cost", "stop_rule_hit", "ask", "search", "none"]:
        assert key in t


def test_qa_refine_template_schema_keys():
    t = P.QA_REFINE_SYS
    assert_json_only(t)
    for key in ["question", "reasons", "checks", "is_binary", "has_assumptions", "is_minimal", "hypothesis"]:
        assert key in t


