import numpy as np
import pytest

from worldsimprobe.evaluation.task5_interaction_dynamics import frozen_protocol
from worldsimprobe.evaluation.task5_interaction_dynamics.evaluator import (
    _sample_physical_time,
)


def test_physical_time_sampling_uses_full_reference_horizon() -> None:
    frames = np.arange(11, dtype=np.uint8)[:, None, None, None]
    row = {"prediction_metadata": {"full_horizon_validation": {"expected_duration_sec": 1.0}}}
    sampled, timestamps, indices, candidate_duration, expected_duration = _sample_physical_time(
        row, frames, decoded_fps=10.0, sample_count=5
    )
    assert len(sampled) == 5
    assert timestamps.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert indices.tolist() == [0, 2, 5, 7, 10]
    assert candidate_duration == 1.0
    assert expected_duration == 1.0


def test_physical_time_sampling_uses_shorter_common_horizon() -> None:
    frames = np.arange(6, dtype=np.uint8)[:, None, None, None]
    row = {"prediction_metadata": {"full_horizon_validation": {"expected_duration_sec": 1.0}}}
    _, timestamps, indices, candidate_duration, expected_duration = _sample_physical_time(
        row, frames, decoded_fps=10.0, sample_count=12
    )
    assert timestamps[-1] == 0.5
    assert indices[-1] == 5
    assert candidate_duration == 0.5
    assert expected_duration == 1.0


def test_physical_time_sampling_ignores_frames_beyond_expected_horizon() -> None:
    frames = np.arange(14, dtype=np.uint8)[:, None, None, None]
    row = {"prediction_metadata": {"full_horizon_validation": {"expected_duration_sec": 1.0}}}
    _, timestamps, indices, candidate_duration, expected_duration = _sample_physical_time(
        row, frames, decoded_fps=10.0, sample_count=12
    )
    assert timestamps[-1] == 1.0
    assert indices[-1] == 10
    assert candidate_duration == 1.3
    assert expected_duration == 1.0


def test_task5_primary_accuracy_is_macro_averaged_by_primitive(tmp_path, monkeypatch) -> None:
    config = tmp_path / "task5.yaml"
    config.write_text(
        "primitives: [push, pull]\nprimitive_descriptions:\n  push: {}\n  pull: {}\n",
        encoding="utf-8",
    )
    rows = [
        {"row_id": "push-1", "primitive": "push", "match": 1},
        {"row_id": "push-2", "primitive": "push", "match": 1},
        {"row_id": "pull-1", "primitive": "pull", "match": 0},
    ]
    monkeypatch.setattr(frozen_protocol, "select_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(frozen_protocol, "load_qwen3_vl", lambda **kwargs: (object(), object()))

    def evaluate(row, **kwargs):
        match = int(row["match"])
        return {
            "row_id": row["row_id"],
            "intended_primitive": row["primitive"],
            "predicted_primitive": row["primitive"] if match else "push",
            "primitive_match": match,
            "forced_choice_primitive_match": match,
            "agent_motion_match_int": match,
            "object_motion_match_int": match,
            "motion_gate_match": match,
            "interaction_visible": 1,
            "integrity_ok": 1,
            "computed_pass": match,
            "response_format_valid": 1,
        }

    monkeypatch.setattr(frozen_protocol, "evaluate_task5_row_qwen3_vl_vqa", evaluate)
    result = frozen_protocol.evaluate_task5_manifest_qwen3_vl_vqa(
        manifest=tmp_path / "unused.jsonl",
        root=tmp_path,
        task_config=config,
        model_name_or_path="unused",
    )
    assert result["summary"]["primitive_accuracy"] == 0.5
    assert result["summary"]["primitive_accuracy_micro"] == pytest.approx(2.0 / 3.0)


def test_task5_scores_each_sample_once(tmp_path, monkeypatch) -> None:
    config = tmp_path / "task5.yaml"
    config.write_text(
        "primitives: [push, pull]\nprimitive_descriptions:\n  push: {}\n  pull: {}\n",
        encoding="utf-8",
    )
    rows = [
        {
            "row_id": sample_id,
            "sample_id": sample_id,
            "primitive": primitive,
            "match": match,
        }
        for sample_id, primitive, match in (
            ("push-1", "push", 1),
            ("pull-1", "pull", 0),
        )
    ]
    monkeypatch.setattr(frozen_protocol, "select_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(frozen_protocol, "load_qwen3_vl", lambda **kwargs: (object(), object()))

    def evaluate(row, **kwargs):
        match = int(row["match"])
        return {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "intended_primitive": row["primitive"],
            "predicted_primitive": row["primitive"] if match else "invalid",
            "primitive_match": match,
            "forced_choice_primitive_match": match,
            "agent_motion_match_int": match,
            "object_motion_match_int": match,
            "motion_gate_match": match,
            "interaction_visible": 1,
            "integrity_ok": 1,
            "computed_pass": match,
            "response_format_valid": 1,
        }

    monkeypatch.setattr(frozen_protocol, "evaluate_task5_row_qwen3_vl_vqa", evaluate)
    result = frozen_protocol.evaluate_task5_manifest_qwen3_vl_vqa(
        manifest=tmp_path / "unused.jsonl",
        root=tmp_path,
        task_config=config,
        model_name_or_path="unused",
    )
    assert result["rows_scored"] == 2
    assert "generation_rows_scored" not in result
    assert result["summary"]["primitive_accuracy"] == 0.5
