from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

EPS = 1e-12
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_710


def triangular_ratio_score(model_ratio: float, oracle_ratio: float) -> float:
    """Score a model ratio against the oracle, with a peak of one at equality."""
    if not np.isfinite(model_ratio):
        return 0.0
    if not np.isfinite(oracle_ratio) or not 0.0 <= oracle_ratio < 1.0:
        raise ValueError(f"oracle_ratio must be finite and in [0, 1), got {oracle_ratio}")

    if oracle_ratio <= EPS:
        score = 1.0 - model_ratio
    elif oracle_ratio >= 1.0 - EPS:
        score = model_ratio
    elif model_ratio <= oracle_ratio:
        score = model_ratio / oracle_ratio
    else:
        score = (1.0 - model_ratio) / (1.0 - oracle_ratio)
    return float(np.clip(score, 0.0, 1.0))


def score_task1_oracle_ratio(
    *,
    small_mse: float,
    large_mse: float,
    simulator_small_mse: float | None,
    simulator_large_mse: float | None,
) -> dict[str, Any]:
    """Compute the official Task 1 oracle-ratio fields for one triplet."""
    small = float(small_mse)
    large = float(large_mse)
    if not np.isfinite(small) or not np.isfinite(large) or small < 0.0 or large < 0.0:
        raise ValueError("model MSE values must be finite and non-negative")

    try:
        if simulator_small_mse is None or simulator_large_mse is None:
            raise ValueError("missing simulator MSE")
        simulator_small = float(simulator_small_mse)
        simulator_large = float(simulator_large_mse)
    except (TypeError, ValueError):
        simulator_small = np.nan
        simulator_large = np.nan

    simulator_values_valid = (
        np.isfinite(simulator_small)
        and np.isfinite(simulator_large)
        and simulator_small >= 0.0
        and simulator_large > EPS
    )
    oracle_ratio = float(simulator_small / simulator_large) if simulator_values_valid else None
    oracle_valid = bool(simulator_values_valid and simulator_large > simulator_small)

    if large <= EPS:
        model_ratio = 0.0 if small <= EPS else None
        score = 0.0 if oracle_valid else None
    else:
        model_ratio = float(small / large)
        score = (
            triangular_ratio_score(model_ratio, oracle_ratio)
            if oracle_valid and oracle_ratio is not None
            else None
        )

    return {
        "oracle_ratio": oracle_ratio,
        "model_ratio": model_ratio,
        "oracle_valid": oracle_valid,
        "direction_pass": bool(large > small),
        "oracle_ratio_score": score,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return None
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if len(array) == 1:
        value = float(array[0])
        return [value, value]

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def summarize_task1_oracle_ratio(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("oracle_valid")
        and row.get("oracle_ratio_score") is not None
        and np.isfinite(row["oracle_ratio_score"])
    ]
    if not valid:
        return {
            "oracle_valid_rows": 0,
            "oracle_ratio_score_mean": None,
            "oracle_ratio_score_percent": None,
            "oracle_ratio_score_median": None,
            "oracle_ratio_score_mean_ci95": None,
            "direction_pass_rate": None,
            "model_ratio_median": None,
            "oracle_ratio_median": None,
            "zero_large_response_rows": 0,
        }

    scores = np.asarray([row["oracle_ratio_score"] for row in valid], dtype=float)
    model_ratios = np.asarray(
        [row["model_ratio"] for row in valid if row.get("model_ratio") is not None],
        dtype=float,
    )
    oracle_ratios = np.asarray([row["oracle_ratio"] for row in valid], dtype=float)
    mean_score = float(scores.mean())
    return {
        "oracle_valid_rows": len(valid),
        "oracle_ratio_score_mean": mean_score,
        "oracle_ratio_score_percent": 100.0 * mean_score,
        "oracle_ratio_score_median": float(np.median(scores)),
        "oracle_ratio_score_mean_ci95": bootstrap_mean_ci(
            scores,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "direction_pass_rate": float(np.mean([row["direction_pass"] for row in valid])),
        "model_ratio_median": float(np.median(model_ratios)) if len(model_ratios) else None,
        "oracle_ratio_median": float(np.median(oracle_ratios)),
        "zero_large_response_rows": int(
            sum(
                row.get("large_mse") is not None
                and np.isfinite(row["large_mse"])
                and row["large_mse"] <= EPS
                for row in valid
            )
        ),
    }
