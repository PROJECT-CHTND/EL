from __future__ import annotations

from typing import Dict, List

from agent.slots import Slot, SlotRegistry


_DAILY_WORK_SLOT_DEFINITIONS: List[Dict[str, object]] = [
    {
        "name": "subject",
        "description": "High-level subject of today's work (project, product, or initiative)",
        "type": "daily_subject",
        "importance": 0.9,
    },
    {
        "name": "projects",
        "description": "Projects you worked on today",
        "type": "daily_projects",
        "importance": 0.9,
    },
    {
        "name": "tasks",
        "description": "Concrete tasks you executed today, ideally with time spent and links",
        "type": "daily_tasks",
        "importance": 0.98,
    },
    {
        "name": "artifacts",
        "description": "Artifacts produced today (PRs, documents, deployments, etc.)",
        "type": "daily_artifacts",
        "importance": 0.9,
    },
    {
        "name": "blockers",
        "description": "Blockers or issues you encountered, their causes and how you addressed them",
        "type": "daily_blockers",
        "importance": 0.9,
    },
    {
        "name": "next_step",
        "description": "Next concrete step you plan to take (e.g., tomorrow's first task)",
        "type": "daily_next_step",
        "importance": 0.95,
    },
]


_FALLBACK_QUESTIONS: Dict[str, Dict[str, str]] = {
    "subject": {
        "Japanese": "今日の業務の主な対象（プロダクトやプロジェクトなど）を、一言で教えてください。",
        "English": "In one phrase, what was the main subject of your work today (product, project, or initiative)?",
    },
    "projects": {
        "Japanese": "今日はどのプロジェクトに取り組みましたか？（例：機能A、改善B、調査C など）",
        "English": "Which projects did you work on today? (e.g., Feature A, Improvement B, Investigation C)",
    },
    "tasks": {
        "Japanese": "具体的にどのタスクを実施しましたか？できれば所要時間や関連リンク（PR/チケットなど）も含めて教えてください。",
        "English": "Which concrete tasks did you complete today? If possible, include time spent and related links (PRs, tickets, etc.).",
    },
    "artifacts": {
        "Japanese": "今日生まれた成果物（PR、ドキュメント、デプロイなど）があれば教えてください。",
        "English": "What tangible artifacts did you produce today (PRs, documents, deployments, etc.), if any?",
    },
    "blockers": {
        "Japanese": "今日、詰まりやブロッカーはありましたか？原因と、どのように対処した/する予定かを教えてください。",
        "English": "Did you encounter any blockers today? Please describe the causes and how you addressed or plan to address them.",
    },
    "next_step": {
        "Japanese": "次の一歩として、明日あるいは次回にやるべき具体的な一つのタスクを挙げてください。",
        "English": "As your next step, what is one concrete task you plan to do tomorrow or next time?",
    },
}


def build_daily_work_registry() -> SlotRegistry:
    """Create a SlotRegistry pre-populated with daily work report slots."""

    registry = SlotRegistry()
    for spec in _DAILY_WORK_SLOT_DEFINITIONS:
        registry.add(Slot(**spec))
    return registry


def fallback_question(slot_name: str, language: str) -> str:
    """Return a deterministic fallback question for the given daily work slot and language."""

    lang = "Japanese" if language.lower().startswith("ja") else "English"
    questions = _FALLBACK_QUESTIONS.get(slot_name)
    if not questions:
        return "Could you share more details about today's work?"
    return questions.get(lang) or questions.get("English") or "Could you share more details about today's work?"


