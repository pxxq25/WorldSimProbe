from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.common.metrics import finite_values
from worldsimprobe.common.robotseg_masks import flow_masks_from_frame_masks, robotseg_masks_for_frames
from worldsimprobe.common.timebase import time_align_frame_pair, timestamps_from_info
from worldsimprobe.evaluation.task2_action_source.flow import (
    candidate_path_for_mode,
    flow_distance,
    json_safe,
    load_rgb,
    optical_flow_sequence,
    resample_flow,
    resolve_path,
)

DEFAULT_MOTION_FLOOR = 3.16


def _action_source_group(row: dict[str, Any]) -> str:
    keys = ("action_source_group", "source_group", "source_type", "action_source", "control_source")
    for container in (row, row.get("metadata"), row.get("source_metadata")):
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    if row.get("policy_checkpoint") is not None or row.get("checkpoint") is not None:
        return "policy_checkpoint"
    return "unspecified"


def _resample_bool_sequence(mask: np.ndarray, n: int) -> np.ndarray:
    if len(mask) == n:
        return mask
    if len(mask) == 1:
        return np.repeat(mask, n, axis=0)
    indices = np.round(np.linspace(0, len(mask) - 1, n)).astype(int)
    return mask[indices]


def _resample_frame_sequence(frames: np.ndarray, n: int) -> np.ndarray:
    if len(frames) == n:
        return frames
    if len(frames) == 1:
        return np.repeat(frames, n, axis=0)
    indices = np.round(np.linspace(0, len(frames) - 1, n)).astype(int)
    return frames[indices]


def _positive_fps(value: Any) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if np.isfinite(fps) and fps > 0.0 else None


def _fps_from_row(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    fps = _positive_fps(row.get(key))
    return fps if fps is not None else _positive_fps(default)


def _fps_from_timestamps(timestamps: list[float] | tuple[float, ...] | np.ndarray | None) -> float | None:
    if timestamps is None:
        return None
    values = np.asarray(timestamps, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return None
    duration = float(values[-1] - values[0])
    if duration <= 0.0:
        return None
    return _positive_fps((len(values) - 1) / duration)


def _timing_dicts_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    timing_dicts: list[dict[str, Any]] = []
    for value in (row.get("actbench_timing"), row.get("timing")):
        if isinstance(value, dict):
            timing_dicts.append(value)
    eval_block = row.get("eval")
    if isinstance(eval_block, dict) and isinstance(eval_block.get("timing"), dict):
        timing_dicts.append(eval_block["timing"])
    model_input = row.get("model_input")
    if isinstance(model_input, dict):
        if isinstance(model_input.get("timing"), dict):
            timing_dicts.append(model_input["timing"])
        action_trajectory = model_input.get("action_trajectory")
        if isinstance(action_trajectory, dict) and isinstance(action_trajectory.get("timing"), dict):
            timing_dicts.append(action_trajectory["timing"])
    return timing_dicts


def _timing_fps_for_row(row: dict[str, Any], *keys: str) -> float | None:
    for timing in _timing_dicts_for_row(row):
        for key in keys:
            fps = _positive_fps(timing.get(key))
            if fps is not None:
                return fps
    return None


def _hdf5_fps(path: Path) -> float | None:
    if path.suffix.lower() not in {".hdf5", ".h5"}:
        return None
    try:
        with h5py.File(path, "r") as f:
            for key in ("fps", "video_fps", "recorded_action_hz"):
                fps = _positive_fps(f.attrs.get(key))
                if fps is not None:
                    return fps
    except OSError:
        return None
    return None


def _reference_fps_for_row(
    row: dict[str, Any],
    reference_gt_path: Path,
    reference_timestamps: list[float] | None,
    default: float | None,
) -> float | None:
    return (
        _fps_from_timestamps(reference_timestamps)
        or _fps_from_row(row, "canonical_video_fps")
        or _fps_from_row(row, "video_fps")
        or _timing_fps_for_row(row, "video_fps", "hdf5_fps", "recorded_action_hz")
        or _hdf5_fps(reference_gt_path)
        or _positive_fps(default)
    )


def _candidate_fps_for_row(
    row: dict[str, Any],
    candidate_timestamps: list[float] | None,
    default: float | None,
) -> float | None:
    return (
        _fps_from_timestamps(candidate_timestamps)
        or _fps_from_row(row, "candidate_fps")
        or _fps_from_row(row, "native_fps")
        or _fps_from_row(row, "fps")
        or _positive_fps(default)
    )


def _has_timebase(info: dict[str, Any]) -> bool:
    return (
        info.get("frame_timestamps_sec") is not None
        or _positive_fps(info.get("native_fps") or info.get("fps")) is not None
    )


def _reference_gt_path_for_row(row: dict[str, Any]) -> str:
    for key in (
        "reference_video",
        "output_video",
        "reference_gt_hdf5",
        "gt_hdf5",
        "output_hdf5",
        "model_hdf5",
    ):
        value = row.get(key)
        if value:
            return str(value)
    references = row.get("eval", {}).get("references", {}) if isinstance(row.get("eval"), dict) else {}
    for key in (
        "reference_video",
        "output_video",
        "reference_gt_hdf5",
        "gt_hdf5",
        "output_hdf5",
        "model_hdf5",
    ):
        value = references.get(key)
        if value:
            return str(value)
    raise KeyError("Task 2 GT-reference flow evaluation requires a GT HDF5 or reference video")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _fixed_eval_timestamps(
    *,
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    reference_info: dict[str, Any],
    candidate_info: dict[str, Any],
    eval_fps: float,
) -> np.ndarray:
    if not np.isfinite(eval_fps) or eval_fps <= 0.0:
        raise ValueError(f"eval_fps must be positive, got {eval_fps}")
    reference_timestamps = timestamps_from_info(len(reference_frames), reference_info)
    candidate_timestamps = timestamps_from_info(len(candidate_frames), candidate_info)
    start = max(float(reference_timestamps[0]), float(candidate_timestamps[0]))
    end = min(float(reference_timestamps[-1]), float(candidate_timestamps[-1]))
    if end <= start:
        return np.asarray([start], dtype=np.float64)
    step = 1.0 / float(eval_fps)
    count = int(np.floor((end - start) / step + 1e-9)) + 1
    timestamps = start + np.arange(max(count, 1), dtype=np.float64) * step
    timestamps = timestamps[timestamps <= end + 1e-9]
    if len(timestamps) < 2:
        timestamps = np.asarray([start, end], dtype=np.float64)
    return timestamps


def _timestamps_for_loaded_frames(
    timestamps: list[float] | tuple[float, ...] | np.ndarray | None,
    *,
    offset: int,
    count: int,
) -> list[float] | None:
    if timestamps is None:
        return None
    values = list(timestamps)
    if len(values) == count:
        return [float(value) for value in values]
    end = offset + count
    if len(values) >= end:
        return [float(value) for value in values[offset:end]]
    return None


class DPFlowEstimator:
    def __init__(
        self,
        ckpt_path: str = "things",
        device: str = "cuda",
        fp16: bool = False,
    ) -> None:
        import ptlflow
        import torch

        self.torch = torch
        self.device = device
        self.fp16 = fp16
        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                torch.cuda.set_device(int(device.split(":", 1)[1]))
            self.cuda = True
        else:
            self.cuda = False
        self.model = ptlflow.get_model("dpflow", ckpt_path=ckpt_path).eval()
        if self.cuda:
            self.model = self.model.cuda()
        if fp16:
            self.model = self.model.half()
        self.io_adapters: dict[tuple[int, int], Any] = {}

    def flow_sequence(
        self,
        frames: np.ndarray,
        width: int = 160,
        height: int = 120,
        stride: int = 1,
    ) -> np.ndarray:
        import cv2
        from ptlflow.utils.io_adapter import IOAdapter

        if stride > 1:
            frames = frames[::stride]
        resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
        if len(resized) < 2:
            raise RuntimeError("not enough frames to compute DPFlow")

        key = (height, width)
        if key not in self.io_adapters:
            self.io_adapters[key] = IOAdapter(
                self.model,
                input_size=(height, width),
                cuda=self.cuda,
                fp16=self.fp16,
            )
        io_adapter = self.io_adapters[key]

        flows = []
        with self.torch.no_grad():
            for prev, nxt in zip(resized[:-1], resized[1:]):
                inputs = io_adapter.prepare_inputs([prev, nxt])
                preds = self.model(inputs)
                preds = io_adapter.unscale(preds)
                flow_tensor = preds["flows"]
                while flow_tensor.dim() > 3:
                    flow_tensor = flow_tensor[0]
                flow = flow_tensor.permute(1, 2, 0).float().cpu().numpy()
                flows.append(flow.astype(np.float32))
        return np.stack(flows, axis=0)


def robotseg_masked_gt_reference_flow_metrics(
    candidate_flow: np.ndarray,
    reference_flow: np.ndarray,
    reference_flow_mask: np.ndarray,
    motion_threshold: float = 0.25,
    motion_floor: float = DEFAULT_MOTION_FLOOR,
    flow_fps: float | None = 5.0,
    window_sec: float = 2.0,
    window_stride_sec: float = 1.0,
) -> dict[str, Any]:
    n = min(len(candidate_flow), len(reference_flow), len(reference_flow_mask))
    if n < 1:
        raise ValueError("empty flow sequence")
    candidate = resample_flow(candidate_flow, n)
    reference = resample_flow(reference_flow, n)
    reference_mask = _resample_bool_sequence(reference_flow_mask, n)

    def eval_mask_for_window(
        ref_flow: np.ndarray, ref_mask: np.ndarray
    ) -> tuple[np.ndarray, None, np.ndarray]:
        reference_mag = np.linalg.norm(ref_flow, axis=-1)
        active_reference_motion_mask = reference_mag > motion_threshold
        active_gt_robot_mask = ref_mask & active_reference_motion_mask
        if not active_gt_robot_mask.any():
            raise ValueError("empty active reference robot-motion mask")
        return active_gt_robot_mask, None, active_gt_robot_mask

    def score_window(start: int, end: int) -> dict[str, Any]:
        cand = candidate[start:end]
        ref = reference[start:end]
        ref_mask = reference_mask[start:end]
        eval_mask, mask_fallback, active_gt_robot_mask = eval_mask_for_window(ref, ref_mask)
        flow_error = flow_distance(cand, ref, eval_mask)
        flow_rms = flow_distance(np.zeros_like(ref), ref, eval_mask)
        denominator = max(flow_rms, float(motion_floor), 1e-6)
        score = 100.0 * max(0.0, 1.0 - flow_error / denominator)
        gt_mask_for_metric = ref_mask if ref_mask.any() else eval_mask
        return {
            "start_flow_index": int(start),
            "end_flow_index": int(end),
            "score": score,
            "flow_error": flow_error,
            "flow_rms": flow_rms,
            "score_denominator": denominator,
            "gt_mask_flow_error": flow_distance(cand, ref, gt_mask_for_metric),
            "eval_mask_fraction": float(np.mean(eval_mask)),
            "active_gt_robot_fraction": float(np.mean(active_gt_robot_mask)),
            "gt_robot_mask_fraction": float(np.mean(ref_mask)),
            "mask_fallback": mask_fallback,
        }

    global_metrics = score_window(0, n)
    fps = (
        float(flow_fps)
        if flow_fps is not None and np.isfinite(float(flow_fps)) and float(flow_fps) > 0
        else 5.0
    )
    window_frames = max(2, int(round(float(window_sec) * fps)))
    window_step = max(1, int(round(float(window_stride_sec) * fps)))
    starts = list(range(0, max(n - window_frames + 1, 1), window_step))
    final_start = max(n - window_frames, 0)
    if starts and starts[-1] != final_start:
        starts.append(final_start)
    if not starts:
        starts = [0]
    windows = []
    for start in starts:
        end = min(start + window_frames, n)
        if end - start < 2:
            continue
        item = score_window(start, end)
        item["start_sec"] = float(start / fps)
        item["end_sec"] = float(end / fps)
        item["mid_sec"] = float(((start + end) / 2.0) / fps)
        windows.append(item)

    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    window_scores = [float(item["score"]) for item in windows]
    window_errors = [float(item["flow_error"]) for item in windows]
    midpoint_sec = (n / fps) / 2.0
    first_half_scores = [float(item["score"]) for item in windows if float(item["mid_sec"]) <= midpoint_sec]
    second_half_scores = [float(item["score"]) for item in windows if float(item["mid_sec"]) > midpoint_sec]
    score_summary = summarize(window_scores)
    error_summary = summarize(window_errors)
    return {
        "gt_robot_flow_error": global_metrics["flow_error"],
        "gt_robot_flow_rms": global_metrics["flow_rms"],
        "gt_robot_flow_score_denominator": global_metrics["score_denominator"],
        "gt_robot_flow_score_0_to_100": global_metrics["score"],
        "gt_robot_global_flow_score_0_to_100": global_metrics["score"],
        "gt_mask_flow_error": global_metrics["gt_mask_flow_error"],
        "eval_mask_fraction": global_metrics["eval_mask_fraction"],
        "active_robot_fraction": global_metrics["eval_mask_fraction"],
        "active_gt_robot_fraction": global_metrics["active_gt_robot_fraction"],
        "gt_robot_mask_fraction": global_metrics["gt_robot_mask_fraction"],
        "mean_window_gt_robot_flow_score_0_to_100": score_summary["mean"],
        "median_window_gt_robot_flow_score_0_to_100": score_summary["median"],
        "min_window_gt_robot_flow_score_0_to_100": score_summary["min"],
        "max_window_gt_robot_flow_score_0_to_100": score_summary["max"],
        "first_half_window_gt_robot_flow_score_0_to_100": summarize(first_half_scores)["mean"],
        "second_half_window_gt_robot_flow_score_0_to_100": summarize(second_half_scores)["mean"],
        "mean_window_gt_robot_flow_error": error_summary["mean"],
        "window_count": len(windows),
        "window_sec": float(window_sec),
        "window_stride_sec": float(window_stride_sec),
        "window_flow_fps": fps,
        "window_frames": int(window_frames),
        "window_step_frames": int(window_step),
        "windows": windows,
        "frames_compared": int(n),
        "motion_threshold": motion_threshold,
        "motion_floor": float(motion_floor),
        "mask_policy": "gt_robot_mask",
        "mask_fallback": global_metrics["mask_fallback"],
        "mask_categories": ["robot"],
        "reference": "gt_cross_action",
    }


def evaluate_task2_row_robotseg_flow(
    row: dict[str, Any],
    root: Path,
    candidate_mode: str = "output",
    timeline_mode: str | None = None,
    alignment_timebase: str = "candidate",
    candidate_frame_offset: int = 0,
    donor_frame_offset: int = 0,
    eval_fps: float | None = 5.0,
    camera: str = "head_camera",
    width: int = 160,
    height: int = 120,
    stride: int = 1,
    max_frames: int | None = None,
    motion_threshold: float = 0.25,
    motion_floor: float = DEFAULT_MOTION_FLOOR,
    window_sec: float = 2.0,
    window_stride_sec: float = 1.0,
    robotseg_root: str = "checkpoints/RobotSeg",
    robotseg_checkpoint: str = "checkpoints/robotseg.pt",
    device: str = "cuda",
    flow_model: str = "dpflow",
    dpflow_estimator: DPFlowEstimator | None = None,
) -> dict[str, Any]:
    candidate_path = resolve_path(root, candidate_path_for_mode(row, candidate_mode))
    reference_gt_path = resolve_path(root, _reference_gt_path_for_row(row))
    donor_path = resolve_path(root, row["donor_hdf5"]) if row.get("donor_hdf5") else None

    candidate_frames = load_rgb(candidate_path, camera=camera, max_frames=max_frames)
    reference_frames = load_rgb(reference_gt_path, camera=camera, max_frames=max_frames)
    candidate_original_frames = len(candidate_frames)
    reference_original_frames = len(reference_frames)
    if candidate_frame_offset:
        candidate_frames = candidate_frames[candidate_frame_offset:]
    if donor_frame_offset:
        reference_frames = reference_frames[donor_frame_offset:]
    if len(candidate_frames) < 2:
        raise RuntimeError(f"candidate has too few frames after offset: {candidate_path}")
    if len(reference_frames) < 2:
        raise RuntimeError(f"GT reference has too few frames after offset: {reference_gt_path}")
    time_alignment = None
    candidate_timestamps = _timestamps_for_loaded_frames(
        row.get("candidate_frame_timestamps_sec") or row.get("frame_timestamps_sec"),
        offset=candidate_frame_offset,
        count=len(candidate_frames),
    )
    reference_timestamps = _timestamps_for_loaded_frames(
        row.get("canonical_eval_timestamps_sec"),
        offset=donor_frame_offset,
        count=len(reference_frames),
    )
    default_fps = eval_fps or 10.0
    reference_fps = _reference_fps_for_row(
        row,
        reference_gt_path,
        reference_timestamps,
        default_fps,
    )
    candidate_fps = _candidate_fps_for_row(
        row,
        candidate_timestamps,
        reference_fps,
    )
    candidate_info = {
        "fps": candidate_fps,
        "native_fps": candidate_fps,
        "frame_timestamps_sec": candidate_timestamps,
    }
    reference_info = {
        "fps": reference_fps,
        "native_fps": reference_fps,
        "frame_timestamps_sec": reference_timestamps,
    }
    alignment_timebase = alignment_timebase.lower()
    if alignment_timebase == "donor":
        alignment_timebase = "reference"
    if alignment_timebase not in {"candidate", "reference"}:
        raise ValueError(f"unknown alignment_timebase: {alignment_timebase}")
    if _has_timebase(candidate_info) and _has_timebase(reference_info):
        target_timestamps = None
        if eval_fps is not None:
            target_timestamps = _fixed_eval_timestamps(
                reference_frames=reference_frames,
                candidate_frames=candidate_frames,
                reference_info=reference_info,
                candidate_info=candidate_info,
                eval_fps=float(eval_fps),
            )
        elif alignment_timebase == "candidate":
            # Compare flow over the model's generated timesteps. For lower-FPS
            # models this downsamples the GT reference to the candidate
            # timestamps instead of repeating candidate frames on the reference
            # timeline.
            target_timestamps = timestamps_from_info(len(candidate_frames), candidate_info)
        elif alignment_timebase == "reference":
            target_timestamps = reference_info.get("frame_timestamps_sec")
        reference_frames, candidate_frames, time_alignment = time_align_frame_pair(
            reference_frames=reference_frames,
            candidate_frames=candidate_frames,
            reference_info=reference_info,
            candidate_info=candidate_info,
            max_samples=max_frames,
            target_timestamps=target_timestamps,
        )
        if eval_fps is not None and time_alignment is not None:
            time_alignment["eval_fps"] = float(eval_fps)
            time_alignment["timebase"] = "fixed_eval_fps"
    timeline_path = None
    timeline_frames = None
    if timeline_mode is not None:
        timeline_path = resolve_path(root, candidate_path_for_mode(row, timeline_mode))
        timeline_frames_rgb = load_rgb(timeline_path, camera=camera, max_frames=max_frames)
        timeline_frames = len(timeline_frames_rgb)
        if timeline_frames < 2:
            raise RuntimeError(f"timeline has too few frames: {timeline_path}")
        candidate_frames = _resample_frame_sequence(candidate_frames, timeline_frames)
        reference_frames = _resample_frame_sequence(reference_frames, timeline_frames)

    if flow_model == "dpflow":
        if dpflow_estimator is None:
            raise RuntimeError("flow_model=dpflow requires a DPFlowEstimator")
        reference_flow = dpflow_estimator.flow_sequence(
            reference_frames,
            width=width,
            height=height,
            stride=stride,
        )
        if _same_path(candidate_path, reference_gt_path):
            candidate_flow = reference_flow.copy()
        else:
            candidate_flow = dpflow_estimator.flow_sequence(
                candidate_frames,
                width=width,
                height=height,
                stride=stride,
            )
    elif flow_model == "farneback":
        reference_flow = optical_flow_sequence(reference_frames, width=width, height=height, stride=stride)
        if _same_path(candidate_path, reference_gt_path):
            candidate_flow = reference_flow.copy()
        else:
            candidate_flow = optical_flow_sequence(
                candidate_frames, width=width, height=height, stride=stride
            )
    else:
        raise ValueError(f"unknown flow_model: {flow_model}")

    reference_mask_frames = reference_frames[::stride] if stride > 1 else reference_frames
    reference_masks = robotseg_masks_for_frames(
        reference_mask_frames,
        categories=("robot",),
        robotseg_root=robotseg_root,
        checkpoint=robotseg_checkpoint,
        device=device,
    )
    reference_flow_mask = flow_masks_from_frame_masks(
        reference_masks["union_mask"], width=width, height=height
    )
    flow_fps = None
    if eval_fps is not None:
        flow_fps = float(eval_fps) / max(int(stride), 1)
    elif candidate_fps is not None:
        flow_fps = float(candidate_fps) / max(int(stride), 1)
    metrics = robotseg_masked_gt_reference_flow_metrics(
        candidate_flow,
        reference_flow,
        reference_flow_mask,
        motion_threshold=motion_threshold,
        motion_floor=motion_floor,
        flow_fps=flow_fps,
        window_sec=window_sec,
        window_stride_sec=window_stride_sec,
    )
    return json_safe(
        {
            "row_id": row.get("row_id"),
            "sample_id": row.get("sample_id") or row.get("row_id"),
            "action_source_group": _action_source_group(row),
            "receiver_task": row.get("receiver_task"),
            "donor_task": row.get("donor_task"),
            "episode": row.get("episode"),
            "candidate_mode": candidate_mode,
            "candidate_path": str(candidate_path),
            "reference_gt_path": str(reference_gt_path),
            "donor_path": str(donor_path) if donor_path is not None else None,
            "candidate_original_frames": candidate_original_frames,
            "reference_original_frames": reference_original_frames,
            "candidate_fps": candidate_fps,
            "reference_fps": reference_fps,
            "candidate_frame_offset": int(candidate_frame_offset),
            "reference_frame_offset": int(donor_frame_offset),
            "candidate_frames": len(candidate_frames),
            "reference_frames": len(reference_frames),
            "timeline_mode": timeline_mode,
            "alignment_timebase": alignment_timebase,
            "eval_fps": float(eval_fps) if eval_fps is not None else None,
            "timeline_path": str(timeline_path) if timeline_path is not None else None,
            "timeline_frames": timeline_frames,
            "time_alignment": time_alignment,
            "flow_model": flow_model,
            "robotseg_root": robotseg_root,
            "robotseg_checkpoint": robotseg_checkpoint,
            **metrics,
        }
    )


def evaluate_task2_manifest_robotseg_flow(
    manifest: Path,
    root: Path,
    candidate_mode: str = "output",
    timeline_mode: str | None = None,
    alignment_timebase: str = "candidate",
    candidate_frame_offset: int = 0,
    donor_frame_offset: int = 0,
    eval_fps: float | None = 5.0,
    limit: int | None = None,
    row_ids: set[str] | None = None,
    camera: str = "head_camera",
    width: int = 160,
    height: int = 120,
    stride: int = 1,
    max_frames: int | None = None,
    motion_threshold: float = 0.25,
    motion_floor: float = DEFAULT_MOTION_FLOOR,
    window_sec: float = 2.0,
    window_stride_sec: float = 1.0,
    robotseg_root: str = "checkpoints/RobotSeg",
    robotseg_checkpoint: str = "checkpoints/robotseg.pt",
    device: str = "cuda",
    flow_model: str = "dpflow",
    dpflow_ckpt: str = "checkpoints/dpflow-things-2012b5d6.ckpt",
    dpflow_fp16: bool = False,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest)]
    if row_ids is not None:
        rows = [row for row in rows if str(row.get("row_id")) in row_ids]
    if limit is not None:
        rows = rows[:limit]

    dpflow_estimator = None
    if flow_model == "dpflow":
        dpflow_estimator = DPFlowEstimator(
            ckpt_path=dpflow_ckpt,
            device=device,
            fp16=dpflow_fp16,
        )
    elif flow_model != "farneback":
        raise ValueError(f"unknown flow_model: {flow_model}")

    results = []
    errors = []
    for row in rows:
        try:
            results.append(
                evaluate_task2_row_robotseg_flow(
                    row,
                    root,
                    candidate_mode=candidate_mode,
                    timeline_mode=timeline_mode,
                    alignment_timebase=alignment_timebase,
                    candidate_frame_offset=candidate_frame_offset,
                    donor_frame_offset=donor_frame_offset,
                    eval_fps=eval_fps,
                    camera=camera,
                    width=width,
                    height=height,
                    stride=stride,
                    max_frames=max_frames,
                    motion_threshold=motion_threshold,
                    motion_floor=motion_floor,
                    window_sec=window_sec,
                    window_stride_sec=window_stride_sec,
                    robotseg_root=robotseg_root,
                    robotseg_checkpoint=robotseg_checkpoint,
                    device=device,
                    flow_model=flow_model,
                    dpflow_estimator=dpflow_estimator,
                )
            )
        except Exception as exc:
            error = {
                "row_id": row.get("row_id"),
                "sample_id": row.get("sample_id") or row.get("row_id"),
                "action_source_group": _action_source_group(row),
                "receiver_task": row.get("receiver_task"),
                "donor_task": row.get("donor_task"),
                "episode": row.get("episode"),
                "error": repr(exc),
            }
            errors.append(error)
            results.append(
                {
                    **error,
                    "evaluation_failed": 1,
                    "gt_robot_flow_score_0_to_100": 0.0,
                    "gt_robot_global_flow_score_0_to_100": 0.0,
                    "mean_window_gt_robot_flow_score_0_to_100": 0.0,
                    "median_window_gt_robot_flow_score_0_to_100": 0.0,
                    "min_window_gt_robot_flow_score_0_to_100": 0.0,
                    "max_window_gt_robot_flow_score_0_to_100": 0.0,
                    "first_half_window_gt_robot_flow_score_0_to_100": 0.0,
                    "second_half_window_gt_robot_flow_score_0_to_100": 0.0,
                }
            )

    def mean(key: str) -> float | None:
        values = finite_values(results, key)
        if not values:
            return None
        return float(np.mean(values))

    def median(key: str) -> float | None:
        values = finite_values(results, key)
        if not values:
            return None
        return float(np.median(values))

    return json_safe(
        {
            "manifest": str(manifest),
            "root": str(root),
            "candidate_mode": candidate_mode,
            "timeline_mode": timeline_mode,
            "alignment_timebase": alignment_timebase,
            "candidate_frame_offset": candidate_frame_offset,
            "reference_frame_offset": donor_frame_offset,
            "eval_fps": float(eval_fps) if eval_fps is not None else None,
            "metric": "robotseg_gt_masked_gt_reference_windowed_flow",
            "primary_metric": "mean_window_gt_robot_flow_score_0_to_100",
            "mask_policy": "gt_robot_mask",
            "reference": "gt_cross_action",
            "flow_model": flow_model,
            "dpflow_ckpt": dpflow_ckpt if flow_model == "dpflow" else None,
            "motion_floor": float(motion_floor),
            "window_sec": float(window_sec),
            "window_stride_sec": float(window_stride_sec),
            "score_formula": (
                "per window: 100 * max(0, 1 - gt_robot_flow_error / "
                "max(gt_robot_flow_rms, motion_floor)); primary score is mean over windows"
            ),
            "score_direction": "higher_is_better",
            "primary_score_direction": "higher_is_better",
            "rows_requested": len(rows),
            "rows_scored": len(results),
            "errors": errors,
            "summary": {
                "mean_gt_robot_flow_error": mean("gt_robot_flow_error"),
                "median_gt_robot_flow_error": median("gt_robot_flow_error"),
                "mean_gt_robot_flow_rms": mean("gt_robot_flow_rms"),
                "mean_gt_robot_flow_score_denominator": mean("gt_robot_flow_score_denominator"),
                "mean_gt_robot_flow_score_0_to_100": mean("gt_robot_flow_score_0_to_100"),
                "median_gt_robot_flow_score_0_to_100": median("gt_robot_flow_score_0_to_100"),
                "mean_gt_robot_global_flow_score_0_to_100": mean("gt_robot_global_flow_score_0_to_100"),
                "mean_window_gt_robot_flow_score_0_to_100": mean("mean_window_gt_robot_flow_score_0_to_100"),
                "median_window_gt_robot_flow_score_0_to_100": median(
                    "mean_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_median_window_gt_robot_flow_score_0_to_100": mean(
                    "median_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_min_window_gt_robot_flow_score_0_to_100": mean(
                    "min_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_max_window_gt_robot_flow_score_0_to_100": mean(
                    "max_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_first_half_window_gt_robot_flow_score_0_to_100": mean(
                    "first_half_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_second_half_window_gt_robot_flow_score_0_to_100": mean(
                    "second_half_window_gt_robot_flow_score_0_to_100"
                ),
                "mean_window_gt_robot_flow_error": mean("mean_window_gt_robot_flow_error"),
                "mean_gt_mask_flow_error": mean("gt_mask_flow_error"),
                "mean_eval_mask_fraction": mean("eval_mask_fraction"),
                "mean_active_gt_robot_fraction": mean("active_gt_robot_fraction"),
                "mean_gt_robot_mask_fraction": mean("gt_robot_mask_fraction"),
            },
            "rows": results,
        }
    )
