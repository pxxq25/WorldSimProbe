import json
from pathlib import Path

import numpy as np
import pytest

from worldsimprobe.common.join import join_references_with_submission
from worldsimprobe.evaluation.task1_action_calibration import evaluator as task1_evaluator
from worldsimprobe.evaluation.task2_action_source import robotseg_flow
from worldsimprobe.evaluation.task4_interaction_grounding import tracker as task4_tracker
from worldsimprobe.submission.preflight import (
    expected_duration_sec,
    validate_full_horizon,
)


def test_join_rejects_missing_assigned_sample(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    submission = tmp_path / "submission.jsonl"
    references.write_text(
        "\n".join(
            json.dumps({"sample_id": sample_id, "task_id": "task4"})
            for sample_id in ("sample-a", "sample-b")
        )
        + "\n",
        encoding="utf-8",
    )
    submission.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "task_id": "task4",
                "model_id": "model",
                "videos": {"candidate": "sample-a.mp4"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample-b"):
        join_references_with_submission(references, submission, tmp_path, task_id="task4")


def test_expected_horizon_comes_from_nested_model_input() -> None:
    row = {
        "row_id": "sample",
        "model_input": {
            "action_trajectory": {
                "duration_sec": 4.5,
                "timestamps_sec": [0.0, 4.5],
            }
        },
    }
    assert expected_duration_sec(row, "candidate") == (4.5, "model_input.action_trajectory")


def test_full_horizon_allows_one_frame_rounding_but_not_truncation() -> None:
    validate_full_horizon(actual_duration_sec=9.9, expected_duration=10.0, fps=10.0)
    with pytest.raises(ValueError, match="shorter"):
        validate_full_horizon(actual_duration_sec=9.8, expected_duration=10.0, fps=10.0)
    with pytest.raises(ValueError, match="longer"):
        validate_full_horizon(actual_duration_sec=10.2, expected_duration=10.0, fps=10.0)


def test_join_ignores_submitted_timing_and_uses_evaluator_config(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    submission = tmp_path / "submission.jsonl"
    references.write_text('{"sample_id":"sample","task_id":"task2"}\n', encoding="utf-8")
    submission.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "task_id": "task2",
                "model_id": "model",
                "videos": {"candidate": "candidate.mp4"},
                "video_timing": {
                    "candidate": {
                        "fps": 12.0,
                        "frame_timestamps_sec": [0.0, 1.0 / 12.0],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = join_references_with_submission(references, submission, tmp_path, task_id="task2")
    assert len(rows) == 1
    assert "generation_seed" not in rows[0]
    assert "generation_index" not in rows[0]
    assert rows[0]["candidate_fps"] == 10.0
    assert "candidate_frame_timestamps_sec" not in rows[0]
    assert rows[0]["video_timing"] == {"candidate": {"fps": 10.0}}
    assert rows[0]["video_timing_source"] == "evaluator_config"


def test_task1_evaluator_failure_remains_in_denominator(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(task1_evaluator, "evaluate_task1_row", fail)
    result = task1_evaluator.evaluate_task1_rows(
        [{"sample_id": "sample", "simulator_small_mse": 1.0, "simulator_large_mse": 4.0}],
        tmp_path,
    )
    assert result["summary"]["rows_requested"] == 1
    assert result["summary"]["rows_scored"] == 1
    assert result["summary"]["score"] == 0.0


def test_rmfa_evaluator_failure_remains_in_denominator(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "joined.jsonl"
    manifest.write_text('{"row_id":"sample"}\n', encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("flow failure")

    monkeypatch.setattr(robotseg_flow, "evaluate_task2_row_robotseg_flow", fail)
    result = robotseg_flow.evaluate_task2_manifest_robotseg_flow(
        manifest,
        tmp_path,
        flow_model="farneback",
    )
    assert result["rows_requested"] == 1
    assert result["rows_scored"] == 1
    assert result["summary"]["mean_window_gt_robot_flow_score_0_to_100"] == 0.0


def test_rmfa_aggregates_one_score_per_sample(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "joined.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "row_id": sample_id,
                    "sample_id": sample_id,
                    "score": score,
                }
            )
            for sample_id, score in (("a", 0.0), ("b", 50.0), ("c", 100.0))
        )
        + "\n",
        encoding="utf-8",
    )

    def evaluate(row, *args, **kwargs):
        return {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "mean_window_gt_robot_flow_score_0_to_100": row["score"],
        }

    monkeypatch.setattr(robotseg_flow, "evaluate_task2_row_robotseg_flow", evaluate)
    result = robotseg_flow.evaluate_task2_manifest_robotseg_flow(
        manifest,
        tmp_path,
        flow_model="farneback",
    )
    assert result["rows_scored"] == 3
    assert "generation_rows_scored" not in result
    assert result["summary"]["mean_window_gt_robot_flow_score_0_to_100"] == 50.0


def test_rmfa_rejects_empty_active_reference_robot_mask() -> None:
    candidate = np.zeros((2, 4, 4, 2), dtype=np.float32)
    reference = np.zeros_like(candidate)
    robot_mask = np.ones((2, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="empty active reference robot-motion mask"):
        robotseg_flow.robotseg_masked_gt_reference_flow_metrics(
            candidate,
            reference,
            robot_mask,
        )


def test_task4_evaluator_failure_remains_in_denominator(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "joined.jsonl"
    manifest.write_text(
        '{"row_id":"sample","object_binding_subset":"distractor"}\n',
        encoding="utf-8",
    )

    def fail(*args, **kwargs):
        raise RuntimeError("tracking failure")

    monkeypatch.setattr(task4_tracker, "evaluate_task4_row_tapnextpp", fail)
    result = task4_tracker.evaluate_task4_manifest_tapnextpp(manifest, tmp_path)
    assert result["rows_requested"] == 1
    assert result["rows_scored"] == 1
    assert result["summary"]["overall_score"] == 0.0
