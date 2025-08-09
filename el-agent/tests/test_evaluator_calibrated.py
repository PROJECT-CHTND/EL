from __future__ import annotations

import json
import os
from pathlib import Path


def test_evaluator_uses_calibrated_weights(tmp_path: Path, monkeypatch):
    from el_agent.core.evaluator import ConfidenceEvaluator  # type: ignore[import-not-found]
    from el_agent.schemas import Evidence  # type: ignore[import-not-found]
    # Prepare a minimal weights file that doubles cosine and adds intercept
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps({"intercept": 0.5, "coef": {"cosine": 1.0}}), encoding="utf-8")

    monkeypatch.setenv("EL_EVAL_WEIGHTS", str(weights_path))

    ev = Evidence(entities=[], relations=[], supports=[{"polarity": "+"}], feature_vector={"cosine": 0.8})
    conf = ConfidenceEvaluator()
    out = conf.score(ev)

    # linear_sum = 0.5 + 1.0 * 0.8 = 1.3
    assert abs(out["logit_delta"] - 1.3) < 1e-6

