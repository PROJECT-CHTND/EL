import types
import pytest
import asyncio

from nani_bot import generate_opening_question, analyze_and_respond, ThinkingSession

from agent.slots import Slot
from agent.models.question import Question
from agent.models.kg import KGPayload
from agent.slots.postmortem import fallback_question

# -----------------------------
# helper mocks
# -----------------------------
class DummyChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)

class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]

@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    """Patch nani_bot.client.chat.completions.create to avoid real API calls."""
    import nani_bot as nb  # local import to access the singleton client

    async def _fake_create(*args, **kwargs):  # noqa: D401
        return DummyResponse("{\"dummy\":true}")

    monkeypatch.setattr(nb.client.chat.completions, "create", _fake_create, raising=True)
    yield

@pytest.mark.asyncio
async def test_generate_opening_question(monkeypatch):
    """generate_opening_question should return non-empty string using mocked LLM."""
    # monkeypatching is auto via fixture
    from nani_bot import generate_opening_question

    result = await generate_opening_question("テスト", "Japanese")
    assert isinstance(result, str)
    assert result  # non-empty

@pytest.mark.asyncio
async def test_analyze_and_respond(monkeypatch):
    """analyze_and_respond should produce at least one next question with mocks."""
    import nani_bot as nb

    # patch pipeline functions to deterministic mocks
    async def _mock_extract(text: str, **kwargs):  # noqa: ANN001
        class _KG:  # minimal stub with model_dump_json attribute used in prompts
            entities: list[str] = []
            relations: list[str] = []

            def model_dump_json(self):
                return "{}"
        return _KG()

    async def _mock_propose(kg, **kwargs):  # noqa: ANN001
        return [Slot(name="slot1", description="dummy")]

    async def _mock_generate(slots, **kwargs):  # noqa: ANN001
        return [Question(slot_name="slot1", text="質問A", specificity=0.9, tacit_power=0.9)]

    async def _mock_validate(questions):  # noqa: ANN001
        return questions  # return as-is

    async def _mock_run_turn(answer_text: str, **kwargs):  # noqa: ANN001
        await _mock_extract(answer_text, **kwargs)
        slots = await _mock_propose(None)
        questions = await _mock_generate(slots)
        return await _mock_validate(questions)

    monkeypatch.setattr(nb, "extract_knowledge", _mock_extract, raising=True)
    monkeypatch.setattr(nb, "propose_slots", _mock_propose, raising=True)
    monkeypatch.setattr(nb, "generate_questions", _mock_generate, raising=True)
    monkeypatch.setattr(nb, "return_validated_questions", _mock_validate, raising=True)
    monkeypatch.setattr(nb, "run_turn", _mock_run_turn, raising=True)

    # construct session
    session = ThinkingSession("user1", "テストトピック", 123, "Japanese")
    session.last_question = "最初の質問"
    session.add_exchange("最初の質問", "最初の回答")

    result = await analyze_and_respond(session)
    assert "next_questions" in result
    assert result["next_questions"]
    assert isinstance(result["next_questions"][0]["question"], str)


@pytest.mark.asyncio
async def test_postmortem_gap_loop(monkeypatch):
    """Postmortem sessions should progress through critical slots."""
    import nani_bot as nb

    session = ThinkingSession("user2", "昨日の障害を振り返りたい", 456, "Japanese")
    assert session.goal_kind == "postmortem"

    summary_question = fallback_question("summary", session.language)
    summary_answer = "昨日14:30の決済障害で約2時間停止し、ユーザーの35%が失敗しました。"
    session.pending_slot = "summary"
    session.last_question = summary_question
    session.add_exchange(summary_question, summary_answer)
    session.slot_registry.update("summary", value=summary_answer, filled_ratio=1.0)
    session.goal_state.setdefault("filled", {})["summary"] = summary_answer
    session.slot_answers.setdefault("summary", []).append(summary_answer)
    session.pending_slot = None

    async def _mock_extract(text: str, **kwargs):  # noqa: ANN001
        return KGPayload(entities=[], relations=[])

    def _mock_merge(payload, **kwargs):  # noqa: ANN001
        return payload

    async def _mock_generate(slots, **kwargs):  # noqa: ANN001
        slot = slots[0]
        return [Question(slot_name=slot.name, text="影響はどのくらいでしたか？", specificity=0.9, tacit_power=0.9)]

    async def _mock_validate(questions):  # noqa: ANN001
        return questions

    monkeypatch.setattr(nb, "extract_knowledge", _mock_extract, raising=True)
    monkeypatch.setattr(nb, "merge_and_persist", _mock_merge, raising=True)
    monkeypatch.setattr(nb, "generate_questions", _mock_generate, raising=True)
    monkeypatch.setattr(nb, "return_validated_questions", _mock_validate, raising=True)

    result = await analyze_and_respond(session)

    assert result["status"] == "continue"
    assert result["next_questions"]
    assert result["next_questions"][0]["slot_name"] == "impact"
    assert session.pending_slot == "impact"


@pytest.mark.asyncio
async def test_postmortem_gap_loop_mismatched_slot(monkeypatch):
    """Fallback question should be used when qgen returns mismatched slot."""
    import nani_bot as nb

    session = ThinkingSession("user3", "障害", 789, "Japanese")
    assert session.goal_kind == "postmortem"

    summary_question = fallback_question("summary", session.language)
    summary_answer = "14:30 に決済APIで障害が発生し、2時間停止しました。"
    session.pending_slot = "summary"
    session.last_question = summary_question
    session.add_exchange(summary_question, summary_answer)
    session.slot_registry.update("summary", value=summary_answer, filled_ratio=1.0)
    session.goal_state.setdefault("filled", {})["summary"] = summary_answer
    session.slot_answers.setdefault("summary", []).append(summary_answer)
    session.pending_slot = None

    async def _mock_extract(text: str, **kwargs):  # noqa: ANN001
        return KGPayload(entities=[], relations=[])

    def _mock_merge(payload, **kwargs):  # noqa: ANN001
        return payload

    async def _mock_generate(slots, **kwargs):  # noqa: ANN001
        return [Question(slot_name="unrelated", text="別の質問", specificity=0.9, tacit_power=0.9)]

    async def _mock_validate(questions):  # noqa: ANN001
        return questions

    monkeypatch.setattr(nb, "extract_knowledge", _mock_extract, raising=True)
    monkeypatch.setattr(nb, "merge_and_persist", _mock_merge, raising=True)
    monkeypatch.setattr(nb, "generate_questions", _mock_generate, raising=True)
    monkeypatch.setattr(nb, "return_validated_questions", _mock_validate, raising=True)

    result = await analyze_and_respond(session)

    assert result["status"] == "continue"
    assert result["next_questions"]
    next_q = result["next_questions"][0]
    assert next_q["slot_name"] == "impact"
    assert next_q["question"] == fallback_question("impact", session.language)
    assert session.pending_slot == "impact"
