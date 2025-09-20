from __future__ import annotations

from el_agent.core.strategist import Strategist, entropy
from el_agent.schemas import Hypothesis


def test_entropy_monotonic_around_half():
    assert entropy(0.5) > entropy(0.4)
    assert entropy(0.5) > entropy(0.6)


def test_higher_uncertainty_increases_action_value():
    s = Strategist(tau_stop=0.0)
    h_low = Hypothesis(id="h1", text="t", belief=0.1, slots=["a"], action_cost={"ask": 1.0, "search": 1.0})
    h_high = Hypothesis(id="h2", text="t", belief=0.5, slots=["a"], action_cost={"ask": 1.0, "search": 1.0})
    a_low = s.pick_action(h_low)
    a_high = s.pick_action(h_high)
    assert a_high.expected_gain >= a_low.expected_gain


def test_tau_stop_halts():
    s = Strategist(tau_stop=10.0)
    h = Hypothesis(id="h1", text="t", belief=0.5, slots=["a", "b"], action_cost={"ask": 1.0, "search": 1.0})
    a = s.pick_action(h)
    assert a.action == "none"
    assert a.stop_rule_hit is True


