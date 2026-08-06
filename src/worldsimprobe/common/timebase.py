from __future__ import annotations

from typing import Any

import numpy as np


def positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def timestamps_from_count(
    *,
    count: int,
    fps: float | None = None,
    timestamps_sec: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"timestamp count must be positive, got {count}")
    if timestamps_sec is not None:
        timestamps = np.asarray(timestamps_sec, dtype=np.float64)
        if len(timestamps) != count:
            raise ValueError(f"timestamp length {len(timestamps)} does not match frame count {count}")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("timestamps must be finite")
        if np.any(np.diff(timestamps) < -1e-9):
            raise ValueError("timestamps must be monotonically non-decreasing")
        return timestamps

    native_fps = positive_float(fps)
    if native_fps is None:
        raise ValueError("fps or explicit timestamps are required")
    return np.arange(count, dtype=np.float64) / native_fps


def timestamps_from_info(count: int, info: dict[str, Any]) -> np.ndarray:
    timestamps = info.get("frame_timestamps_sec")
    if timestamps is None:
        timestamps = info.get("timestamps_sec")
    fps = info.get("native_fps") or info.get("fps")
    return timestamps_from_count(count=count, fps=positive_float(fps), timestamps_sec=timestamps)


def nearest_indices_for_timestamps(
    source_timestamps: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    if len(source_timestamps) == 0:
        raise ValueError("source_timestamps must not be empty")
    positions = np.searchsorted(source_timestamps, target_timestamps, side="left")
    positions = np.clip(positions, 0, len(source_timestamps) - 1)
    previous = np.clip(positions - 1, 0, len(source_timestamps) - 1)
    use_previous = np.abs(source_timestamps[previous] - target_timestamps) <= np.abs(
        source_timestamps[positions] - target_timestamps
    )
    return np.where(use_previous, previous, positions).astype(np.int64)


def common_reference_timestamps(
    reference_timestamps: np.ndarray,
    candidate_timestamps: np.ndarray,
    *,
    max_samples: int | None = None,
    target_timestamps: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    start = max(float(reference_timestamps[0]), float(candidate_timestamps[0]))
    end = min(float(reference_timestamps[-1]), float(candidate_timestamps[-1]))
    if target_timestamps is not None:
        base = np.asarray(target_timestamps, dtype=np.float64)
    else:
        base = reference_timestamps
    valid = base[(base >= start - 1e-9) & (base <= end + 1e-9)]
    if len(valid) == 0:
        valid = np.asarray([start], dtype=np.float64)
    if max_samples is not None and max_samples > 0 and len(valid) > max_samples:
        indices = np.round(np.linspace(0, len(valid) - 1, max_samples)).astype(np.int64)
        valid = valid[indices]
    return valid.astype(np.float64)


def align_frames_to_timestamps(
    frames: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    indices = nearest_indices_for_timestamps(source_timestamps, target_timestamps)
    return frames[indices], [int(index) for index in indices]


def time_align_frame_pair(
    *,
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    reference_info: dict[str, Any],
    candidate_info: dict[str, Any],
    max_samples: int | None = None,
    target_timestamps: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_timestamps = timestamps_from_info(len(reference_frames), reference_info)
    candidate_timestamps = timestamps_from_info(len(candidate_frames), candidate_info)
    timestamps = common_reference_timestamps(
        reference_timestamps,
        candidate_timestamps,
        max_samples=max_samples,
        target_timestamps=target_timestamps,
    )
    reference_aligned, reference_indices = align_frames_to_timestamps(
        reference_frames,
        reference_timestamps,
        timestamps,
    )
    candidate_aligned, candidate_indices = align_frames_to_timestamps(
        candidate_frames,
        candidate_timestamps,
        timestamps,
    )
    return (
        reference_aligned,
        candidate_aligned,
        {
            "method": "nearest_frame_common_timebase",
            "reference_fps": positive_float(reference_info.get("native_fps") or reference_info.get("fps")),
            "candidate_fps": positive_float(candidate_info.get("native_fps") or candidate_info.get("fps")),
            "reference_span_seconds": float(reference_timestamps[-1] - reference_timestamps[0]),
            "candidate_span_seconds": float(candidate_timestamps[-1] - candidate_timestamps[0]),
            "common_start_seconds": float(timestamps[0]),
            "common_end_seconds": float(timestamps[-1]),
            "sample_count": len(timestamps),
            "timestamps_seconds": timestamps.tolist(),
            "reference_frame_indices": reference_indices,
            "candidate_frame_indices": candidate_indices,
        },
    )
