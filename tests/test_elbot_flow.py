import types
import pytest
import asyncio

from nani_bot import generate_opening_question, analyze_and_respond, ThinkingSession

from agent.slots import Slot
from agent.models.question import Question

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

    monkeypatch.setattr(nb, "extract_knowledge", _mock_extract, raising=True)
    monkeypatch.setattr(nb, "propose_slots", _mock_propose, raising=True)
    monkeypatch.setattr(nb, "generate_questions", _mock_generate, raising=True)
    monkeypatch.setattr(nb, "return_validated_questions", _mock_validate, raising=True)

    # construct session
    session = ThinkingSession("user1", "テストトピック", 123, "Japanese")
    session.last_question = "最初の質問"
    session.add_exchange("最初の質問", "最初の回答")

    result = await analyze_and_respond(session)
    assert "next_questions" in result
    assert result["next_questions"]
    assert isinstance(result["next_questions"][0]["question"], str)
