#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


def read_labels(path: Path) -> Tuple[List[Dict[str, float]], List[int]]:
    X: List[Dict[str, float]] = []
    y: List[int] = []
    if not path.exists():
        return X, y
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            features = obj.get("features", {}) or {}
            label = obj.get("label", None)
            if not isinstance(features, dict):
                continue
            if label not in (0, 1):
                continue
            row: Dict[str, float] = {}
            for k, v in features.items():
                try:
                    row[str(k)] = float(v)
                except Exception:
                    continue
            # Keep only numeric rows
            X.append(row)
            y.append(int(label))
    return X, y


def to_matrix(X_dicts: List[Dict[str, float]], feature_names: List[str]) -> List[List[float]]:
    mat: List[List[float]] = []
    for row in X_dicts:
        mat.append([float(row.get(name, 0.0)) for name in feature_names])
    return mat


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Calibrate evaluator logistic regression from labels.jsonl")
    parser.add_argument(
        "--labels",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "labels.jsonl"),
        help="Path to labels.jsonl (default: reports/labels.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "config" / "weights" / "weights.json"),
        help="Output path for weights.json (default: config/weights/weights.json)",
    )
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum labeled rows required to fit")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    out_path = Path(args.out)

    X_dicts, y = read_labels(labels_path)
    n = len(y)
    if n < args.min_samples:
        logging.warning("Not enough labels: %d < %d; skip calibration", n, args.min_samples)
        return 0

    # Collect feature names as union of keys, stable order sorted
    feature_names = sorted({k for row in X_dicts for k in row.keys()})
    if not feature_names:
        logging.error("No usable features found in %s", labels_path)
        return 1

    X = to_matrix(X_dicts, feature_names)

    # Fit logistic regression
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception as e:
        logging.error("scikit-learn is required: pip install scikit-learn (%s)", e)
        return 1

    clf = LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced")
    clf.fit(X, y)

    intercept = float(clf.intercept_[0])
    coefs = [float(v) for v in clf.coef_[0]]
    coef_map: Dict[str, float] = {name: weight for name, weight in zip(feature_names, coefs)}

    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"intercept": intercept, "coef": coef_map}, f, ensure_ascii=False, indent=2)
    logging.info("Wrote calibrated weights -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


