from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from worldsimprobe.common.metrics import finite_values
from worldsimprobe.evaluation.task2_action_source.robotseg_flow import (
    evaluate_task2_manifest_robotseg_flow,
)


def evaluate_task3_manifest_robotseg_flow(
    manifest: Path,
    root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate heterogeneous Task 3 action sources with the shared flow metric."""
    result = evaluate_task2_manifest_robotseg_flow(manifest, root, **kwargs)
    result["task_id"] = "task3"
    result["reference"] = "source_specific_control_reference"
    result["metric"] = "robotseg_masked_source_reference_windowed_flow"
    rows = result.get("rows") or []
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({str(row.get("action_source_group") or "unspecified") for row in rows}):
        source_rows = [row for row in rows if str(row.get("action_source_group") or "unspecified") == source]
        scores = finite_values(source_rows, "mean_window_gt_robot_flow_score_0_to_100")
        by_source[source] = {
            "instances_scored": len(source_rows),
            "mean_window_gt_robot_flow_score_0_to_100": (
                float(np.mean(scores)) if scores else None
            ),
        }
    source_scores = [
        float(stats["mean_window_gt_robot_flow_score_0_to_100"])
        for stats in by_source.values()
        if stats["mean_window_gt_robot_flow_score_0_to_100"] is not None
    ]
    result["summary"]["by_action_source"] = by_source
    result["summary"]["source_macro_score_0_to_100"] = (
        float(np.mean(source_scores)) if source_scores else None
    )
    return result
