from __future__ import annotations

from typing import Dict, List

from agent.slots import Slot, SlotRegistry


_RECIPE_SLOT_DEFINITIONS: List[Dict[str, object]] = [
    {
        "name": "basic",
        "description": "Basic information about the dish: name and number of servings",
        "type": "recipe_basic",
        "importance": 1.0,
    },
    {
        "name": "ingredients",
        "description": "List of ingredients with quantities and units (g/ml/piece/tsp, etc.)",
        "type": "recipe_ingredients",
        "importance": 0.98,
    },
    {
        "name": "tools",
        "description": "Required tools and equipment (e.g., pot, oven, thermometer)",
        "type": "recipe_tools",
        "importance": 0.9,
    },
    {
        "name": "prep",
        "description": "Preparation steps before cooking (chopping, marinating, tempering, etc.)",
        "type": "recipe_prep",
        "importance": 0.9,
    },
    {
        "name": "steps",
        "description": "Detailed numbered steps with temperature, time, and heat level",
        "type": "recipe_steps",
        "importance": 0.99,
    },
    {
        "name": "substitutions",
        "description": "Possible ingredient substitutions and dietary constraints (vegetarian/low-carb/gluten-free)",
        "type": "recipe_substitutions",
        "importance": 0.85,
    },
    {
        "name": "pitfalls",
        "description": "Common pitfalls, how to avoid them, and visual cues to judge progress",
        "type": "recipe_pitfalls",
        "importance": 0.9,
    },
    {
        "name": "storage",
        "description": "Storage guidance and reheating tips (how long, how to reheat safely)",
        "type": "recipe_storage",
        "importance": 0.85,
    },
]


_FALLBACK_QUESTIONS: Dict[str, Dict[str, str]] = {
    "basic": {
        "Japanese": "料理名と、何人前を想定しているかを教えてください。",
        "English": "What is the name of the dish, and how many servings is it for?",
    },
    "ingredients": {
        "Japanese": "材料と分量を、単位付き（g/ml/個/小さじなど）で列挙してください。",
        "English": "Please list all ingredients with quantities and units (g/ml/piece/tsp, etc.).",
    },
    "tools": {
        "Japanese": "必要な器具・道具（例：鍋、フライパン、オーブン、温度計など）を列挙してください。",
        "English": "List the tools and equipment required (e.g., pot, pan, oven, thermometer).",
    },
    "prep": {
        "Japanese": "調理前の下ごしらえがあれば、切る・漬ける・常温に戻すなど、具体的に教えてください。",
        "English": "Describe any prep steps before cooking (chopping, marinating, bringing to room temperature, etc.).",
    },
    "steps": {
        "Japanese": "調理の工程を番号付きで詳しく書いてください。各工程の火加減・温度・時間もできるだけ明記してください。",
        "English": "Detail the cooking steps as a numbered list, including heat level, temperature, and time for each step as much as possible.",
    },
    "substitutions": {
        "Japanese": "代替できる材料や、アレルギー・制約（菜食/低糖/グルテンなど）への対応があれば教えてください。",
        "English": "If there are ingredient substitutions or ways to adapt the recipe for dietary constraints (vegetarian/low-carb/gluten-free), please describe them.",
    },
    "pitfalls": {
        "Japanese": "このレシピで失敗しやすいポイントと、その回避策、目で見て判断できるサインを教えてください。",
        "English": "What are common pitfalls in this recipe, how can you avoid them, and what visual cues help judge if things are going well or not?",
    },
    "storage": {
        "Japanese": "作った料理の保存方法と保存可能期間、再加熱のコツがあれば教えてください。",
        "English": "How should the dish be stored, for how long, and do you have any tips for reheating it?",
    },
}


def build_recipe_registry() -> SlotRegistry:
    """Create a SlotRegistry pre-populated with critical recipe slots."""

    registry = SlotRegistry()
    for spec in _RECIPE_SLOT_DEFINITIONS:
        registry.add(Slot(**spec))
    return registry


def fallback_question(slot_name: str, language: str) -> str:
    """Return a deterministic fallback question for the given recipe slot and language."""

    lang = "Japanese" if language.lower().startswith("ja") else "English"
    questions = _FALLBACK_QUESTIONS.get(slot_name)
    if not questions:
        return "Could you share more details about this recipe?"
    return questions.get(lang) or questions.get("English") or "Could you share more details about this recipe?"


