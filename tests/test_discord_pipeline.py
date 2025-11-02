import asyncio
import types

import pytest


@pytest.mark.asyncio
async def test_runner_run_turn_with_mocks(monkeypatch):
    from agent.pipeline import runner as _runner

    class _KG:
        def __init__(self):
            self.entities = []
            self.relations = []

        def model_dump_json(self):
            return "{}"

    class _Slot:
        def __init__(self, name: str):
            self.name = name
            self.description = "desc"
            self.type = None
            self.importance = 1.0
            self.filled = False
            self.last_filled_ts = None

    class _Question:
        def __init__(self, text: str):
            self.text = text
            self.specificity = 0.9
            self.tacit_power = 0.8

        def model_dump(self, exclude_none=True):
            return {"text": self.text}

    async def _extract_knowledge(answer_text: str, focus=None, temperature=0.0):
        return _KG()

    def _merge_and_persist(payload):
        return payload

    async def _propose_slots(kg, topic_meta=None, registry=None, max_slots=3):
        return [_Slot("summary")]

    async def _generate_questions(slots, strategy_path=None, max_questions=10):
        return [_Question("テスト質問ですか？")]

    async def _return_validated_questions(questions):
        return questions

    monkeypatch.setattr(_runner, "extract_knowledge", _extract_knowledge)
    monkeypatch.setattr(_runner, "merge_and_persist", _merge_and_persist)
    monkeypatch.setattr(_runner, "propose_slots", _propose_slots)
    monkeypatch.setattr(_runner, "generate_questions", _generate_questions)
    monkeypatch.setattr(_runner, "return_validated_questions", _return_validated_questions)

    res = await _runner.run_turn("answer text", topic_meta="topic")
    assert res and res[0].text

