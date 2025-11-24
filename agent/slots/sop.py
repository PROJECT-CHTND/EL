from __future__ import annotations

from typing import Dict, List

from agent.slots import Slot, SlotRegistry


_SOP_SLOT_DEFINITIONS: List[Dict[str, object]] = [
    {
        "name": "objective",
        "description": "Overall objective of the SOP / runbook and the desired end state once completed",
        "type": "sop_objective",
        "importance": 1.0,
    },
    {
        "name": "prerequisites",
        "description": "Prerequisites and dependencies such as permissions, environments, materials, credentials",
        "type": "sop_prerequisites",
        "importance": 0.95,
    },
    {
        "name": "environment",
        "description": "Target environment and scope (prod/stg, blast radius, rollback feasibility)",
        "type": "sop_environment",
        "importance": 0.9,
    },
    {
        "name": "steps",
        "description": "High-level sequence of steps, later to be detailed with commands and expected results",
        "type": "sop_steps",
        "importance": 0.98,
    },
    {
        "name": "branches",
        "description": "Branches or exception paths with decision criteria for each path",
        "type": "sop_branches",
        "importance": 0.9,
    },
    {
        "name": "validation",
        "description": "Validation methods and expected results for key steps, including observables and pass criteria",
        "type": "sop_validation",
        "importance": 0.96,
    },
    {
        "name": "rollback",
        "description": "Rollback / recovery steps, including conditions and time needed to safely revert",
        "type": "sop_rollback",
        "importance": 0.97,
    },
    {
        "name": "hazards",
        "description": "Hazards and cautions (e.g., data loss, outage risk) and mitigation strategies",
        "type": "sop_hazards",
        "importance": 0.9,
    },
    {
        "name": "checklist",
        "description": "Final checklist items (prereqs, core steps, validation, cleanup)",
        "type": "sop_checklist",
        "importance": 0.85,
    },
]


_FALLBACK_QUESTIONS: Dict[str, Dict[str, str]] = {
    "objective": {
        "Japanese": "この手順の目的は何ですか？完了後に達しているべき状態を一文で教えてください。",
        "English": "What is the objective of this procedure? In one sentence, what state should be achieved when it is done?",
    },
    "prerequisites": {
        "Japanese": "この手順を実行する前提条件や依存関係（権限/環境/資材/認証情報など）は何がありますか？足りないものはありますか？",
        "English": "What prerequisites and dependencies (permissions, environments, materials, credentials) are required? Is anything currently missing?",
    },
    "environment": {
        "Japanese": "対象となる環境や範囲（本番/検証、影響範囲、ロールバック可否など）を具体的に教えてください。",
        "English": "Please clarify the target environment and scope (prod/staging, blast radius, rollback feasibility).",
    },
    "steps": {
        "Japanese": "大まかな手順の流れを番号付きで列挙してください（後で詳細化します）。",
        "English": "List the high-level sequence of steps as a numbered list (we will detail them later).",
    },
    "branches": {
        "Japanese": "分岐条件や例外パスはありますか？それぞれどのような条件でどちらのパスを選びますか？",
        "English": "Are there any branches or exception paths? For each, what conditions decide which path to take?",
    },
    "validation": {
        "Japanese": "各重要ステップの検証方法や期待結果を教えてください。どのような観測ポイントと合格基準がありますか？",
        "English": "For each key step, what are the validation methods and expected results? What observables and pass criteria do you use?",
    },
    "rollback": {
        "Japanese": "失敗時のロールバックやリカバリー手順はありますか？安全に戻す条件や想定所要時間も含めて教えてください。",
        "English": "What are the rollback or recovery steps if something goes wrong, including conditions and time needed to safely revert?",
    },
    "hazards": {
        "Japanese": "この手順における危険事項や注意点（データ消失/停止リスクなど）はありますか？それぞれの回避策・緩和策を教えてください。",
        "English": "What hazards or cautions (e.g., data loss or outage risk) exist in this procedure, and how do you avoid or mitigate them?",
    },
    "checklist": {
        "Japanese": "最終チェックリストに入れるべき項目（前提/主要手順/検証/後片付け）を列挙してください。",
        "English": "Please list the items that should go into the final checklist (prereqs, core steps, validation, cleanup).",
    },
}


def build_sop_registry() -> SlotRegistry:
    """Create a SlotRegistry pre-populated with critical SOP slots."""

    registry = SlotRegistry()
    for spec in _SOP_SLOT_DEFINITIONS:
        registry.add(Slot(**spec))
    return registry


def fallback_question(slot_name: str, language: str) -> str:
    """Return a deterministic fallback question for the given SOP slot and language."""

    lang = "Japanese" if language.lower().startswith("ja") else "English"
    questions = _FALLBACK_QUESTIONS.get(slot_name)
    if not questions:
        return "Could you share more details about this procedure?"
    return questions.get(lang) or questions.get("English") or "Could you share more details about this procedure?"


