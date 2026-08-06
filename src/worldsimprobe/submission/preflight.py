from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from worldsimprobe.common.tasks import required_video_roles
from worldsimprobe.common.video import probe_video_timing
from worldsimprobe.submission.video_config import VideoTimingConfig, load_video_timing_config


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _duration_from_timestamps(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        start = float(value[0])
        end = float(value[-1])
    except (TypeError, ValueError):
        return None
    duration = end - start
    return duration if math.isfinite(duration) and duration > 0.0 else None


def expected_duration_sec(row: dict[str, Any], role: str) -> tuple[float, str]:
    """Resolve the benchmark-owned prediction horizon for one submitted role."""
    metadata = row.get("prediction_metadata")
    if isinstance(metadata, dict):
        validation = metadata.get("full_horizon_validation")
        if isinstance(validation, dict):
            by_role = validation.get("expected_duration_sec_by_role")
            if isinstance(by_role, dict):
                duration = _positive_float(by_role.get(role))
                if duration is not None:
                    return duration, f"prediction_metadata.full_horizon_validation.{role}"
            duration = _positive_float(validation.get("expected_duration_sec"))
            if duration is not None:
                return duration, "prediction_metadata.full_horizon_validation.expected_duration_sec"

    for key in ("canonical_eval_timestamps_sec", "eval_timestamps_sec"):
        duration = _duration_from_timestamps(row.get(key))
        if duration is not None:
            return duration, key

    model_input = row.get("model_input")
    if isinstance(model_input, dict):
        duration = _duration_from_timestamps(model_input.get("eval_timestamps_sec"))
        if duration is not None:
            return duration, "model_input.eval_timestamps_sec"

        trajectories = model_input.get("action_trajectories")
        if isinstance(trajectories, dict) and isinstance(trajectories.get(role), dict):
            trajectory = trajectories[role]
            duration = _positive_float(trajectory.get("duration_sec"))
            if duration is None:
                duration = _duration_from_timestamps(trajectory.get("timestamps_sec"))
            if duration is not None:
                return duration, f"model_input.action_trajectories.{role}"

        trajectory = model_input.get("action_trajectory")
        if isinstance(trajectory, dict):
            duration = _positive_float(trajectory.get("duration_sec"))
            if duration is None:
                duration = _duration_from_timestamps(trajectory.get("timestamps_sec"))
            if duration is not None:
                return duration, "model_input.action_trajectory"

    raise ValueError(
        f"reference row {row.get('sample_id') or row.get('row_id')} has no benchmark prediction horizon"
    )


def horizon_tolerance_sec(fps: float | None) -> float:
    """Allow at most one decoded frame of container/timestamp rounding."""
    frame_interval = 1.0 / fps if fps and fps > 0.0 else 0.0
    return max(0.05, frame_interval + 1e-6)


def validate_full_horizon(
    *,
    actual_duration_sec: float,
    expected_duration: float,
    fps: float | None,
) -> None:
    tolerance = horizon_tolerance_sec(fps)
    if actual_duration_sec + tolerance < expected_duration:
        raise ValueError(
            f"decoded duration {actual_duration_sec:.3f}s is shorter than the required "
            f"{expected_duration:.3f}s horizon (tolerance {tolerance:.3f}s)"
        )
    if actual_duration_sec - tolerance > expected_duration:
        raise ValueError(
            f"decoded duration {actual_duration_sec:.3f}s is longer than the required "
            f"{expected_duration:.3f}s horizon (tolerance {tolerance:.3f}s)"
        )


def validate_joined_submission(
    rows: list[dict[str, Any]],
    *,
    video_config: VideoTimingConfig | None = None,
) -> dict[str, Any]:
    """Validate decoded videos against the fixed sample set and reference horizons."""
    configured_timing = video_config or load_video_timing_config()
    errors: list[str] = []
    checked = 0
    for row in rows:
        sample_id = str(
            row.get("evaluation_id") or row.get("sample_id") or row.get("row_id") or "<unknown>"
        )
        task_id = str(row.get("worldsimprobe_task_id") or row.get("task_id") or "")
        videos = row.get("videos")
        if not isinstance(videos, dict):
            errors.append(f"{sample_id}: missing joined videos")
            continue
        for role in required_video_roles(task_id):
            value = videos.get(role)
            if not value:
                errors.append(f"{sample_id}/{role}: missing video")
                continue
            try:
                expected, source = expected_duration_sec(row, role)
                metadata = probe_video_timing(Path(str(value)))
                configured_timing.validate_fps(metadata.get("fps"))
                validate_full_horizon(
                    actual_duration_sec=float(metadata["duration_sec"]),
                    expected_duration=expected,
                    fps=configured_timing.fps,
                )
                checked += 1
                prediction_metadata = row.get("prediction_metadata")
                if not isinstance(prediction_metadata, dict):
                    prediction_metadata = {}
                    row["prediction_metadata"] = prediction_metadata
                validation_by_role = prediction_metadata.setdefault(
                    "full_horizon_validation_by_role", {}
                )
                validation = {
                    "role": role,
                    "expected_duration_sec": expected,
                    "expected_duration_source": source,
                    "decoded_duration_sec": float(metadata["duration_sec"]),
                    "decoded_frame_count": int(metadata["frame_count"]),
                    "decoded_fps": metadata.get("fps"),
                    "configured_fps": configured_timing.fps,
                    "tolerance_sec": horizon_tolerance_sec(configured_timing.fps),
                    "passed": True,
                }
                validation_by_role[role] = validation
                if len(required_video_roles(task_id)) == 1:
                    prediction_metadata["full_horizon_validation"] = validation
            except Exception as exc:
                errors.append(f"{sample_id}/{role}: {type(exc).__name__}: {exc}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
        raise ValueError(f"submission preflight failed:\n{preview}{suffix}")
    return {
        "rows": len(rows),
        "videos_checked": checked,
        "video_timing_config": configured_timing.as_dict(),
        "full_horizon_passed": True,
    }
