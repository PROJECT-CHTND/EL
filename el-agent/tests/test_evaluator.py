from __future__ import annotations

from el_agent.core.evaluator import ConfidenceEvaluator, update_belief, logit, sigmoid
from el_agent.schemas import Evidence, Hypothesis


def test_monotonic_positive_delta_increases_belief():
    h = Hypothesis(id="h1", text="t", belief=0.4, belief_ci=(0.2, 0.6))
    # Strong positive evidence
    ev = Evidence(
        entities=[],
        relations=[],
        supports=[{"polarity": "+"}],
        feature_vector={"cosine": 0.9, "source_trust": 0.9, "recency": 0.9, "logic_ok": 1},
    )
    conf = ConfidenceEvaluator()
    delta = conf.score(ev)["logit_delta"]
    h2 = update_belief(h, delta)
    assert h2.belief > h.belief


def test_negative_delta_decreases_belief():
    h = Hypothesis(id="h1", text="t", belief=0.6, belief_ci=(0.4, 0.8))
    ev = Evidence(
        entities=[],
        relations=[],
        supports=[{"polarity": "-"}, {"polarity": "-"}],
        feature_vector={"cosine": 0.8, "source_trust": 0.5, "recency": 0.5, "logic_ok": 0},
    )
    conf = ConfidenceEvaluator()
    delta = conf.score(ev)["logit_delta"]
    h2 = update_belief(h, delta)
    assert h2.belief < h.belief


