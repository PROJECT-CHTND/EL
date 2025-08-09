import pytest

from el_agent.schemas import Evidence, Hypothesis, StrategistAction


def test_hypothesis_clamps_and_ci_order():
    h = Hypothesis(
        id="h1",
        text="test",
        slots=["A"],
        belief=1.5,            # clamp -> 1.0
        belief_ci=(0.9, -0.2), # clamp and reorder -> (0.0, 0.9)
        novelty=-1.0,          # clamp -> 0.0
        contradictions=["c1"],
        last_update=123,
        provenance=[{"src": "x", "type": "y", "weight": 2.0}],
        action_cost={"ask": 2.0, "search": -1.0},
        status="open",
    )
    assert h.belief == 1.0
    assert h.belief_ci == (0.0, 0.9)
    assert h.novelty == 0.0
    assert h.action_cost == {"ask": 1.0, "search": 0.0}


def test_hypothesis_status_validation():
    with pytest.raises(ValueError):
        Hypothesis(id="h2", text="x", status="invalid")


def test_evidence_feature_vector_clamp():
    ev = Evidence(
        entities=[],
        relations=[],
        supports=[],
        feature_vector={"cosine": 1.2, "source_trust": -0.4, "recency": 0.5, "logic_ok": 1},
    )
    assert ev.feature_vector["cosine"] == 1.0
    assert ev.feature_vector["source_trust"] == 0.0
    assert ev.feature_vector["recency"] == 0.5
    assert ev.feature_vector["logic_ok"] == 1


def test_strategist_action_clamp():
    act = StrategistAction(
        target_hypothesis="h1",
        action="ask",
        question=None,
        expected_gain=2.0,
        estimated_cost=-0.3,
        stop_rule_hit=False,
    )
    assert act.expected_gain == 1.0
    assert act.estimated_cost == 0.0


