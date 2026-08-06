from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from worldsimprobe.common.video import time_aligned_video_mse
from worldsimprobe.evaluation.task1_action_calibration.oracle_ratio import (
    score_task1_oracle_ratio,
    summarize_task1_oracle_ratio,
)

VARIANTS = ("original", "small", "large")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def evaluate_task1_row(row: dict[str, Any], root: Path) -> dict[str, Any]:
    videos = row.get("videos") or row.get("candidate_videos") or {}
    missing = [variant for variant in VARIANTS if not videos.get(variant)]
    if missing:
        raise ValueError(f"Task 1 row {row.get('sample_id')} is missing videos: {missing}")
    timing = row.get("video_timing") or row.get("candidate_video_timing") or {}
    paths = {variant: _resolve(root, str(videos[variant])) for variant in VARIANTS}

    small_metric = time_aligned_video_mse(
        paths["original"],
        paths["small"],
        timing.get("original"),
        timing.get("small"),
    )
    large_metric = time_aligned_video_mse(
        paths["original"],
        paths["large"],
        timing.get("original"),
        timing.get("large"),
    )
    score = score_task1_oracle_ratio(
        small_mse=float(small_metric["mse"]),
        large_mse=float(large_metric["mse"]),
        simulator_small_mse=row.get("simulator_small_mse"),
        simulator_large_mse=row.get("simulator_large_mse"),
    )
    return {
        "sample_id": row.get("sample_id") or row.get("row_id"),
        "small_mse": float(small_metric["mse"]),
        "large_mse": float(large_metric["mse"]),
        "small_video_metric": small_metric,
        "large_video_metric": large_metric,
        **score,
    }


def evaluate_task1_rows(rows: Iterable[dict[str, Any]], root: Path) -> dict[str, Any]:
    results = []
    errors = []
    for row in rows:
        try:
            results.append(evaluate_task1_row(row, root))
        except Exception as exc:
            error = {
                "sample_id": row.get("sample_id") or row.get("row_id"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            failure = score_task1_oracle_ratio(
                small_mse=0.0,
                large_mse=1.0,
                simulator_small_mse=row.get("simulator_small_mse"),
                simulator_large_mse=row.get("simulator_large_mse"),
            )
            failure.update(
                {
                    **error,
                    "evaluation_failed": 1,
                    "small_mse": None,
                    "large_mse": None,
                    "model_ratio": None,
                    "direction_pass": False,
                    "oracle_ratio_score": 0.0,
                }
            )
            results.append(failure)
    summary = summarize_task1_oracle_ratio(results)
    summary.update(
        {
            "rows_requested": len(results),
            "rows_scored": len(results),
            "rows_evaluated_successfully": sum(
                not row.get("evaluation_failed") for row in results
            ),
            "error_rows": len(errors),
            "score_name": "oracle_ratio_score_percent",
            "score": summary.get("oracle_ratio_score_percent"),
        }
    )
    return {
        "task_id": "task1",
        "summary": summary,
        "rows": results,
        "errors": errors,
    }


def task1_score(results: Iterable[dict[str, Any]]) -> float | None:
    values = [
        float(row["oracle_ratio_score"])
        for row in results
        if row.get("oracle_ratio_score") is not None and np.isfinite(row["oracle_ratio_score"])
    ]
    return 100.0 * float(np.mean(values)) if values else None
