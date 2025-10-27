from __future__ import annotations

from typing import Dict, List

from agent.slots import Slot, SlotRegistry

_POSTMORTEM_SLOT_DEFINITIONS: List[Dict[str, object]] = [
    {
        "name": "summary",
        "description": "Incident summary including when, where, what failed, and who was impacted",
        "type": "postmortem_summary",
        "importance": 1.0,
    },
    {
        "name": "impact",
        "description": "Quantified and qualitative impact on users, revenue, or systems",
        "type": "postmortem_impact",
        "importance": 1.0,
    },
    {
        "name": "detection_ttd",
        "description": "Detection method and time to detect (TTD)",
        "type": "postmortem_detection",
        "importance": 0.9,
    },
    {
        "name": "timeline",
        "description": "Chronological sequence of key events during the incident",
        "type": "postmortem_timeline",
        "importance": 0.98,
    },
    {
        "name": "capa",
        "description": "Corrective and preventive actions including owner, due date, and success criteria",
        "type": "postmortem_capa",
        "importance": 0.95,
    },
]

_FALLBACK_QUESTIONS: Dict[str, Dict[str, str]] = {
    "summary": {
        "Japanese": "障害の概要を一文で教えてください。いつどこで何が起こり、誰に影響しましたか？",
        "English": "Please give a one-sentence incident summary: when and where it happened, what failed, and who was affected?",
    },
    "impact": {
        "Japanese": "障害の影響範囲を教えてください。利用者数や失敗率など、定量・定性の両面で教えてください。",
        "English": "What was the impact? Share quantitative and qualitative signals such as affected users or failure rate.",
    },
    "detection_ttd": {
        "Japanese": "どのように検知し、検知までどのくらい時間がかかりましたか？",
        "English": "How was the incident detected and what was the time to detect (TTD)?",
    },
    "timeline": {
        "Japanese": "主な出来事を時系列で整理しましょう。時刻・アクション・結果を3つほど挙げてください。",
        "English": "Please outline the key timeline: list 3 events with time, action, and outcome.",
    },
    "capa": {
        "Japanese": "再発防止・是正策を具体化しましょう。オーナー、期限、成功基準を含めて教えてください。",
        "English": "Let us capture corrective/preventive actions with owner, due date, and success criteria.",
    },
}


def build_postmortem_registry() -> SlotRegistry:
    """Create a SlotRegistry pre-populated with critical postmortem slots."""

    registry = SlotRegistry()
    for spec in _POSTMORTEM_SLOT_DEFINITIONS:
        registry.add(Slot(**spec))
    return registry


def fallback_question(slot_name: str, language: str) -> str:
    """Return a deterministic fallback question for the given slot and language."""

    lang = "Japanese" if language.lower().startswith("ja") else "English"
    questions = _FALLBACK_QUESTIONS.get(slot_name)
    if not questions:
        return "Could you share more details?"
    return questions.get(lang) or questions.get("English") or "Could you share more details?"
