from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.common.metrics import finite_values
from worldsimprobe.common.robotseg_masks import robotseg_masks_for_frames
from worldsimprobe.common.timebase import time_align_frame_pair


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any, str | None]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value, key
    return None, None


def task4_subset(row: dict[str, Any]) -> str:
    explicit = row.get("task4_subset") or row.get("object_binding_subset")
    if explicit:
        key = str(explicit).strip().lower().replace("-", "_")
        if key in {"distractor", "distractor_hallucination"}:
            return "distractor_hallucination"
        if key in {
            "target_shift",
            "proximity",
            "spatial_proximity",
            "spatial_proximity_hallucination",
            "proximity_hallucination",
        }:
            return "proximity_hallucination"
        if key in {
            "no_gripper_close",
            "fake_contact",
            "visual_contact",
            "appearance_induced_false_contact",
            "fake_contact_hallucination",
        }:
            return "fake_contact_hallucination"
        return str(explicit)
    text = " ".join(
        str(row.get(key, ""))
        for key in (
            "counterfactual_type",
            "task",
            "output_task_config",
            "row_id",
            "source_manifest",
        )
    ).lower()
    if "target_shift" in text:
        return "proximity_hallucination"
    if "no_gripper_close" in text:
        return "fake_contact_hallucination"
    return "distractor_hallucination"


def candidate_is_future_only(row: dict[str, Any]) -> bool | None:
    metadata = row.get("prediction_metadata")
    if not isinstance(metadata, dict):
        return None
    if isinstance(metadata.get("candidate_is_future_only"), bool):
        return bool(metadata["candidate_is_future_only"])
    if isinstance(metadata.get("context_frame_in_candidate_video"), bool):
        return not bool(metadata["context_frame_in_candidate_video"])
    serialized_context = metadata.get("serialized_context_frame_count")
    if serialized_context is not None:
        try:
            return int(serialized_context) == 0
        except (TypeError, ValueError):
            pass
    contract = metadata.get("candidate_video_contract") or metadata.get("output_contract")
    if contract is not None:
        normalized = str(contract).strip().lower().replace("-", "_")
        if "future_only" in normalized or "no_context" in normalized:
            return True
    return None


def reference_frames_for_candidate(
    row: dict[str, Any],
    frames: np.ndarray,
    info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    timing = row.get("actbench_timing") or row.get("timing") or {}
    if not isinstance(timing, dict):
        timing = {}
    alignment = str(timing.get("frame_action_alignment") or "")
    action_count = timing.get("action_count")
    try:
        action_count = int(action_count) if action_count is not None else None
    except (TypeError, ValueError):
        action_count = None

    declared_start = timing.get("reference_frame_start_index", 0)
    try:
        declared_start = int(declared_start)
    except (TypeError, ValueError):
        declared_start = 0
    future_only = candidate_is_future_only(row)
    has_context_contract = "f0=context" in alignment.lower()
    if future_only is True and (
        has_context_contract or (action_count is not None and len(frames) == action_count + 1)
    ):
        start = 1
    elif future_only is not False:
        start = declared_start
    else:
        start = 0
    if start < 0 or start >= len(frames):
        raise RuntimeError(f"invalid Task 4 reference frame start {start} for {len(frames)} frames")

    selected = frames[start:]
    selected_info = dict(info)
    selected_info["full_reference_frame_count"] = len(frames)
    selected_info["reference_frame_start_index_applied"] = int(start)
    selected_info["candidate_is_future_only"] = future_only
    selected_info["frames"] = len(selected)

    canonical = row.get("canonical_eval_timestamps_sec")
    timestamps = None
    if isinstance(canonical, list):
        if len(canonical) == len(selected):
            timestamps = np.asarray(canonical, dtype=np.float64)
        elif len(canonical) == len(frames):
            timestamps = np.asarray(canonical[start:], dtype=np.float64)
    if timestamps is not None and len(timestamps):
        timestamps = timestamps - float(timestamps[0])
        values = timestamps.tolist()
        selected_info["target_eval_timestamps_sec"] = values
        selected_info["frame_timestamps_sec"] = values
    return selected, selected_info


def evaluated_object_name(subset: str) -> str:
    return "distractor" if subset == "distractor_hallucination" else "target"


def task4_no_contact_score(
    object_displacement_px: float,
    robot_displacement_px: float,
    *,
    object_threshold_px: float = 10.0,
    robot_threshold_px: float = 20.0,
) -> dict[str, Any]:
    object_static_pass = int(object_displacement_px <= object_threshold_px)
    robot_motion_gate_pass = int(robot_displacement_px >= robot_threshold_px)
    passed = int(object_static_pass and robot_motion_gate_pass)
    return {
        "object_static_pass": object_static_pass,
        "robot_motion_gate_pass": robot_motion_gate_pass,
        "task4_pass": passed,
        "task4_score": 100.0 * passed,
    }


def task4_scores(
    subset: str,
    target_accuracy: int,
    distractor_accuracy: int | None,
    has_distractor: bool,
) -> tuple[int, int, str]:
    """Compatibility helper for legacy model-vs-reference diagnostics."""
    combined = int(target_accuracy and (distractor_accuracy if has_distractor else 1))
    if evaluated_object_name(subset) == "distractor":
        if distractor_accuracy is None:
            raise RuntimeError("distractor-hallucination row is missing distractor pose metadata")
        return combined, int(distractor_accuracy), "distractor_trajectory_accuracy"
    return combined, int(target_accuracy), "target_trajectory_accuracy"


def pose_xyz(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[list[float] | None, str | None]:
    value, key = first_present(row, keys)
    if value is None:
        return None, None
    try:
        if len(value) < 3:
            return None, key
        return [float(value[0]), float(value[1]), float(value[2])], key
    except (TypeError, ValueError):
        return None, key


def project_xyz_to_uv(
    xyz: list[float] | tuple[float, ...] | np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[float, float, float]:
    point = np.asarray([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    ext = np.asarray(extrinsic, dtype=np.float64)
    # Handle 4x4 extrinsic by taking first 3 rows so result is 3-dim
    if ext.shape[0] == 4:
        ext = ext[:3]
    camera = ext @ point
    depth = float(camera[2])
    pixel = np.asarray(intrinsic, dtype=np.float64) @ camera
    return float(pixel[0] / depth), float(pixel[1] / depth), depth


def pixel_distance(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    aa = np.asarray(a[:2], dtype=np.float64)
    bb = np.asarray(b[:2], dtype=np.float64)
    return float(np.linalg.norm(aa - bb))


def candidate_video_path(
    row: dict[str, Any],
    root: Path,
    candidate_mode: str,
    candidate_root: Path | None = None,
    candidate_field: str = "model_video",
) -> Path | None:
    if candidate_mode in {"gt", "first"}:
        return None
    if candidate_mode == "field":
        value = row.get(candidate_field)
        if not value:
            raise RuntimeError(f"missing candidate field {candidate_field!r} for row {row.get('row_id')}")
        return resolve_path(candidate_root or root, str(value))
    if candidate_mode == "mirror":
        if candidate_root is None:
            raise RuntimeError("--candidate-root is required for mirror mode")
        return resolve_path(candidate_root, str(row["output_video"]))
    if candidate_mode == "row-id":
        if candidate_root is None:
            raise RuntimeError("--candidate-root is required for row-id mode")
        return candidate_root / f"{row['row_id']}.mp4"
    raise ValueError(f"unknown candidate mode: {candidate_mode}")


def load_camera(path: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        intrinsic = np.asarray(f[f"observation/{camera}/intrinsic_cv"][0], dtype=np.float64)
        extrinsic = np.asarray(f[f"observation/{camera}/extrinsic_cv"][0], dtype=np.float64)
    return intrinsic, extrinsic


def camera_calibration_from_row(row: dict[str, Any], camera: str) -> tuple[np.ndarray, np.ndarray] | None:
    calibration = row.get("camera_calibration")
    if not isinstance(calibration, dict):
        eval_block = row.get("eval")
        references = eval_block.get("references") if isinstance(eval_block, dict) else None
        if isinstance(references, dict):
            calibration = references.get("camera_calibration")
    if not isinstance(calibration, dict):
        return None
    selected = str(calibration.get("camera") or camera)
    if selected != camera:
        raise RuntimeError(f"JSON camera calibration is for {selected!r}, requested {camera!r}")
    intrinsic = np.asarray(calibration.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(calibration.get("extrinsic_cv"), dtype=np.float64)
    if intrinsic.shape != (3, 3) or extrinsic.shape not in {(3, 4), (4, 4)}:
        raise RuntimeError(f"invalid JSON camera calibration shapes: {intrinsic.shape}, {extrinsic.shape}")
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise RuntimeError("JSON camera calibration contains non-finite values")
    return intrinsic, extrinsic


def load_camera_for_row(row: dict[str, Any], root: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    embedded = camera_calibration_from_row(row, camera)
    if embedded is not None:
        return embedded
    value = row.get("output_hdf5")
    if not value:
        raise RuntimeError("row has neither embedded camera_calibration nor legacy output_hdf5")
    return load_camera(resolve_path(root, str(value)), camera)


def load_video_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for Task 4 TAPNext++ evaluation") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames: {path}")
    return (
        np.stack(frames, axis=0),
        {"path": str(path), "frames": frame_count, "fps": fps, "width": width, "height": height},
    )


def resize_frames_like(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if frames.shape[1:3] == reference.shape[1:3]:
        return frames
    import cv2

    out = [
        cv2.resize(frame, (reference.shape[2], reference.shape[1]), interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]
    return np.stack(out, axis=0)


def sample_video(frames: np.ndarray, max_frames: int) -> tuple[np.ndarray, list[int]]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames, list(range(len(frames)))
    idx = np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int)
    return frames[idx], [int(i) for i in idx]


def sample_video_by_indices(frames: np.ndarray, indices: list[int]) -> np.ndarray:
    return frames[np.asarray(indices, dtype=np.int64)]


def time_aligned_video_samples(
    gt_frames: np.ndarray,
    gt_info: dict[str, Any],
    candidate_frames: np.ndarray,
    candidate_info: dict[str, Any],
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gt_aligned, candidate_aligned, alignment = time_align_frame_pair(
        reference_frames=gt_frames,
        candidate_frames=candidate_frames,
        reference_info=gt_info,
        candidate_info=candidate_info,
        max_samples=max_frames,
        target_timestamps=gt_info.get("target_eval_timestamps_sec"),
    )
    return (
        gt_aligned,
        candidate_aligned,
        {
            **alignment,
            "gt_fps": alignment.get("reference_fps"),
            "gt_frame_indices": alignment.get("reference_frame_indices"),
            "common_span_seconds": alignment.get("common_end_seconds", 0.0)
            - alignment.get("common_start_seconds", 0.0),
        },
    )


def object_points(center_uv: tuple[float, float], radius: int, grid: int) -> np.ndarray:
    if grid <= 1:
        return np.asarray([[center_uv[0], center_uv[1]]], dtype=np.float32)
    # Inscribe the Cartesian grid in the requested circular radius. Using
    # [-radius, radius] on both axes puts corner queries at radius * sqrt(2).
    # The small inward margin prevents float32 center-plus-offset rounding from
    # placing a corner microscopically outside the requested radius.
    extent = float(radius) / np.sqrt(2.0) * (1.0 - 1e-6)
    offsets = np.linspace(-extent, extent, grid, dtype=np.float32)
    points = []
    for dy in offsets:
        for dx in offsets:
            points.append([float(center_uv[0]) + float(dx), float(center_uv[1]) + float(dy)])
    return np.asarray(points, dtype=np.float32)


def in_frame_mask(points: np.ndarray, width: int, height: int, margin: float = 1.0) -> np.ndarray:
    return (
        (points[:, 0] >= margin)
        & (points[:, 0] <= width - 1 - margin)
        & (points[:, 1] >= margin)
        & (points[:, 1] <= height - 1 - margin)
    )


def explicit_query_points_from_row(
    row: dict[str, Any],
    object_name: str,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, str | None]:
    key = f"{object_name}_query_points_uv"
    frame_key = f"{object_name}_query_points_frame"
    source_key = f"{object_name}_query_points_source"
    references: dict[str, Any] = {}
    eval_block = row.get("eval")
    if isinstance(eval_block, dict) and isinstance(eval_block.get("references"), dict):
        references = eval_block["references"]

    raw = row.get(key)
    if raw is None:
        raw = references.get(key)
    if raw is None:
        return None, None
    points = np.asarray(raw, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise RuntimeError(f"invalid {key} shape: {points.shape}")
    if not np.isfinite(points).all():
        raise RuntimeError(f"{key} contains non-finite coordinates")

    frame = row.get(frame_key)
    if frame is None:
        frame = references.get(frame_key)
    if frame is not None:
        if not isinstance(frame, dict):
            raise RuntimeError(f"invalid {frame_key}: expected an object")
        query_frame_index = int(frame.get("frame_index", 0))
        if query_frame_index != 0:
            raise RuntimeError(f"{frame_key}.frame_index must be 0 for TAPNext++, got {query_frame_index}")
        source_width = float(frame.get("width", width))
        source_height = float(frame.get("height", height))
        if source_width <= 0.0 or source_height <= 0.0:
            raise RuntimeError(f"invalid {frame_key} dimensions")
        points = points.copy()
        points[:, 0] *= float(width) / source_width
        points[:, 1] *= float(height) / source_height

    valid = in_frame_mask(points, width, height)
    if not bool(np.all(valid)):
        raise RuntimeError(f"{key} contains points outside the evaluation frame")
    source = row.get(source_key)
    if source is None:
        source = references.get(source_key)
    return points, str(source or "manifest_explicit")


@lru_cache(maxsize=2)
def load_tapnextpp(device: str, checkpoint_path: str):
    import torch
    from tapnet.tapnext.tapnext_torch import TAPNext

    if not device.startswith("cuda"):
        raise RuntimeError("TAPNext++ torch inference currently requires a CUDA device")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
        raise RuntimeError(f"TAPNext++ checkpoint is missing or empty: {ckpt_path}")
    model = TAPNext(image_size=(256, 256))
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict({key.replace("tapnext.", ""): value for key, value in ckpt["state_dict"].items()})
    model = model.to(device)
    model.eval()
    return model


def resize_tapnextpp_frames(frames: np.ndarray) -> np.ndarray:
    import cv2

    resized = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR) for frame in frames]
    return np.stack(resized, axis=0).astype(np.float32) / 255.0 * 2.0 - 1.0


def run_tapnextpp(
    frames: np.ndarray,
    target_initial_uv: tuple[float, float],
    distractor_initial_uv: tuple[float, float] | None,
    point_radius: int = 10,
    point_grid: int = 3,
    max_frames: int = 96,
    device: str = "cuda",
    visibility_threshold: float = 0.5,
    checkpoint_path: str = "checkpoints/tapnextpp_ckpt.pt",
    target_query_points_uv: np.ndarray | None = None,
    distractor_query_points_uv: np.ndarray | None = None,
) -> dict[str, Any]:
    import torch

    sampled, frame_indices = sample_video(frames, max_frames=max_frames)
    h, w = sampled[0].shape[:2]
    target_query_source = "projected_pose_grid"
    if target_query_points_uv is None:
        canonical_center = (
            float(target_initial_uv[0]) * 256.0 / float(w),
            float(target_initial_uv[1]) * 256.0 / float(h),
        )
        target_points = object_points(canonical_center, point_radius, point_grid)
        target_points[:, 0] *= float(w) / 256.0
        target_points[:, 1] *= float(h) / 256.0
    else:
        target_points = np.asarray(target_query_points_uv, dtype=np.float32)
        target_query_source = "manifest_explicit"
    target_points = target_points[in_frame_mask(target_points, w, h)]
    if len(target_points) == 0:
        raise RuntimeError("TAPNext++ target query points are out of frame")

    has_distractor = distractor_initial_uv is not None
    distractor_points = np.empty((0, 2), dtype=np.float32)
    distractor_query_source = None
    if has_distractor:
        distractor_query_source = "projected_pose_grid"
        if distractor_query_points_uv is None:
            canonical_center = (
                float(distractor_initial_uv[0]) * 256.0 / float(w),
                float(distractor_initial_uv[1]) * 256.0 / float(h),
            )
            distractor_points = object_points(canonical_center, point_radius, point_grid)
            distractor_points[:, 0] *= float(w) / 256.0
            distractor_points[:, 1] *= float(h) / 256.0
        else:
            distractor_points = np.asarray(distractor_query_points_uv, dtype=np.float32)
            distractor_query_source = "manifest_explicit"
        distractor_points = distractor_points[in_frame_mask(distractor_points, w, h)]
        if len(distractor_points) == 0:
            raise RuntimeError("TAPNext++ distractor query points are out of frame")

    points = np.concatenate([target_points, distractor_points], axis=0) if has_distractor else target_points
    tap_points = points.copy().astype(np.float32)
    tap_points[:, 0] *= 256.0 / float(w)
    tap_points[:, 1] *= 256.0 / float(h)
    queries = np.zeros((1, len(points), 3), dtype=np.float32)
    queries[0, :, 1] = tap_points[:, 1]  # TAPNext++ query format is [t, y, x].
    queries[0, :, 2] = tap_points[:, 0]

    video = torch.from_numpy(resize_tapnextpp_frames(sampled))[None].to(device)
    query_tensor = torch.from_numpy(queries).to(device)
    model = load_tapnextpp(device, checkpoint_path)
    pred_tracks = []
    pred_visible_logits = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        tracks, _, visible_logits, tracking_state = model(video=video[:, :1], query_points=query_tensor)
        pred_tracks.append(tracks.detach().float().cpu())
        pred_visible_logits.append(visible_logits.detach().float().cpu())
        for frame in range(1, video.shape[1]):
            tracks, _, visible_logits, tracking_state = model(
                video=video[:, frame : frame + 1],
                state=tracking_state,
            )
            pred_tracks.append(tracks.detach().float().cpu())
            pred_visible_logits.append(visible_logits.detach().float().cpu())

    tracks_np = torch.cat(pred_tracks, dim=1).transpose(1, 2)[0].numpy()[..., ::-1]
    visible_prob_np = torch.sigmoid(torch.cat(pred_visible_logits, dim=1).transpose(1, 2))[0, ..., 0].numpy()
    tracks_np[..., 0] *= float(w) / 256.0
    tracks_np[..., 1] *= float(h) / 256.0

    def centroid_for(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        track = tracks_np[start:end].transpose(1, 0, 2)
        vis = visible_prob_np[start:end].T
        centroids = []
        visible_counts = []
        last = np.median(track[0], axis=0)
        for t in range(track.shape[0]):
            mask = vis[t] >= visibility_threshold
            if np.count_nonzero(mask) == 0:
                centroids.append(last.copy())
                visible_counts.append(0)
                continue
            last = np.median(track[t, mask], axis=0)
            centroids.append(last.copy())
            visible_counts.append(int(np.count_nonzero(mask)))
        return np.stack(centroids, axis=0), np.asarray(visible_counts, dtype=np.int32)

    def raw_for(start: int, end: int) -> dict[str, Any]:
        track = tracks_np[start:end].transpose(1, 0, 2)
        vis = visible_prob_np[start:end].T
        return {
            "raw_trajectory": track.tolist(),
            "raw_visibility": vis.tolist(),
            "raw_median_trajectory": np.median(track, axis=1).tolist(),
        }

    target_centroid, target_counts = centroid_for(0, len(target_points))
    target_raw = raw_for(0, len(target_points))
    result = {
        "target": {
            "trajectory": target_centroid.tolist(),
            "visible_counts": target_counts.tolist(),
            "query_count": len(target_points),
            "query_points_uv": target_points.tolist(),
            "query_source": target_query_source,
            "visible_fraction": float(np.mean(target_counts > 0)),
            "source_height": int(h),
            "source_width": int(w),
            **target_raw,
        },
        "frame_indices": frame_indices,
        "sampled_frames": len(sampled),
        "height": int(h),
        "width": int(w),
    }
    if has_distractor:
        distractor_centroid, distractor_counts = centroid_for(len(target_points), len(points))
        distractor_raw = raw_for(len(target_points), len(points))
        result["distractor"] = {
            "trajectory": distractor_centroid.tolist(),
            "visible_counts": distractor_counts.tolist(),
            "query_count": len(distractor_points),
            "query_points_uv": distractor_points.tolist(),
            "query_source": distractor_query_source,
            "visible_fraction": float(np.mean(distractor_counts > 0)),
            "source_height": int(h),
            "source_width": int(w),
            **distractor_raw,
        }
    return result


def resample_trajectory(trajectory: list[list[float]], length: int) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=np.float64)
    if len(arr) == length:
        return arr
    if length <= 1:
        return arr[[0]]
    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, length)
    x = np.interp(dst, src, arr[:, 0])
    y = np.interp(dst, src, arr[:, 1])
    return np.stack([x, y], axis=1)


def trajectory_errors(
    candidate: dict[str, Any], gt: dict[str, Any], length: int | None = None
) -> dict[str, Any]:
    if length is None:
        length = min(len(candidate["trajectory"]), len(gt["trajectory"]))
    cand = resample_trajectory(candidate["trajectory"], length)
    ref = resample_trajectory(gt["trajectory"], length)
    distances = np.linalg.norm(cand - ref, axis=1)
    return {
        "frames_compared": int(length),
        "mean_error_px": float(np.mean(distances)),
        "median_error_px": float(np.median(distances)),
        "max_error_px": float(np.max(distances)),
        "final_error_px": float(distances[-1]),
        "trajectory": cand.tolist(),
        "gt_trajectory": ref.tolist(),
    }


def draw_track_panel(
    gt_frames: np.ndarray,
    candidate_frames: np.ndarray,
    row_result: dict[str, Any],
    out_path: Path,
    fps: float,
    gt_tracks: dict[str, Any] | None = None,
    candidate_tracks: dict[str, Any] | None = None,
    visibility_threshold: float = 0.5,
    tracker_label: str = "TAPNext++",
) -> None:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(gt_frames), len(candidate_frames), row_result["target"]["frames_compared"])
    gt_target = np.asarray(row_result["target"]["gt_trajectory"], dtype=np.float64)
    cand_target = np.asarray(row_result["target"]["trajectory"], dtype=np.float64)
    has_distractor = "distractor" in row_result
    gt_dist = (
        np.asarray(row_result["distractor"]["gt_trajectory"], dtype=np.float64) if has_distractor else None
    )
    cand_dist = (
        np.asarray(row_result["distractor"]["trajectory"], dtype=np.float64) if has_distractor else None
    )
    h, w = gt_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

    def draw_raw_points(
        bgr: np.ndarray,
        tracks: dict[str, Any] | None,
        object_name: str,
        idx: int,
        color: tuple[int, int, int],
    ) -> None:
        if tracks is None or object_name not in tracks:
            return
        object_tracks = tracks[object_name]
        raw = np.asarray(object_tracks.get("raw_trajectory", []), dtype=np.float64)
        vis = np.asarray(object_tracks.get("raw_visibility", []), dtype=np.float64)
        if raw.ndim != 3 or vis.ndim != 2 or idx >= raw.shape[0] or idx >= vis.shape[0]:
            return

        visible = 0
        for point, confidence in zip(raw[idx], vis[idx]):
            x, y = np.round(point).astype(int)
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if confidence >= visibility_threshold:
                visible += 1
                cv2.circle(bgr, (x, y), 3, color, -1, cv2.LINE_AA)
                cv2.circle(bgr, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                low_color = (150, 150, 150)
                cv2.circle(bgr, (x, y), 4, low_color, 1, cv2.LINE_AA)
                cv2.line(bgr, (x - 3, y - 3), (x + 3, y + 3), low_color, 1, cv2.LINE_AA)
                cv2.line(bgr, (x - 3, y + 3), (x + 3, y - 3), low_color, 1, cv2.LINE_AA)

        total = int(raw.shape[1])
        label = f"{object_name} raw visible {visible}/{total}"
        text_y = 42 if object_name == "target" else 60
        cv2.putText(bgr, label, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def draw_raw_median_path(
        bgr: np.ndarray,
        tracks: dict[str, Any] | None,
        object_name: str,
        idx: int,
        color: tuple[int, int, int],
    ) -> None:
        if tracks is None or object_name not in tracks:
            return
        raw_median = np.asarray(tracks[object_name].get("raw_median_trajectory", []), dtype=np.float64)
        if raw_median.ndim != 2:
            return
        for j in range(max(1, idx - 20), idx + 1):
            if j <= 0 or j >= len(raw_median):
                continue
            p0 = tuple(np.round(raw_median[j - 1]).astype(int))
            p1 = tuple(np.round(raw_median[j]).astype(int))
            cv2.line(bgr, p0, p1, color, 1, cv2.LINE_AA)

    def draw(
        frame: np.ndarray,
        target: np.ndarray,
        distractor: np.ndarray | None,
        tracks: dict[str, Any] | None,
        title: str,
        idx: int,
    ) -> np.ndarray:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        draw_raw_median_path(bgr, tracks, "target", idx, (0, 180, 180))
        if distractor is not None:
            draw_raw_median_path(bgr, tracks, "distractor", idx, (0, 0, 180))
        objects = [(target, (0, 255, 255), "target")]
        if distractor is not None:
            objects.append((distractor, (0, 0, 255), "distractor"))
        for pts, color, label in objects:
            for j in range(max(1, idx - 20), idx + 1):
                if j <= 0 or j >= len(pts):
                    continue
                cv2.line(
                    bgr,
                    tuple(np.round(pts[j - 1]).astype(int)),
                    tuple(np.round(pts[j]).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            p = tuple(np.round(pts[idx]).astype(int))
            cv2.circle(bgr, p, 5, color, -1)
            cv2.putText(
                bgr,
                f"{label} carried centroid",
                (p[0] + 7, p[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        draw_raw_points(bgr, tracks, "target", idx, (0, 255, 255))
        if distractor is not None:
            draw_raw_points(bgr, tracks, "distractor", idx, (0, 0, 255))
        cv2.rectangle(bgr, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(bgr, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bgr

    for i in range(n):
        gt_panel = draw(gt_frames[i], gt_target, gt_dist, gt_tracks, f"GT {tracker_label} raw points", i)
        candidate_panel = draw(
            candidate_frames[i],
            cand_target,
            cand_dist,
            candidate_tracks,
            f"model {tracker_label} raw points",
            i,
        )
        writer.write(np.concatenate([gt_panel, candidate_panel], axis=1))
    writer.release()


def canonical_object_displacement(track: dict[str, Any]) -> dict[str, Any]:
    trajectory = np.asarray(track.get("trajectory"), dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) < 2:
        raise RuntimeError("TAPNext++ did not return a valid object trajectory")
    width = float(track.get("source_width") or 0.0)
    height = float(track.get("source_height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("TAPNext++ result is missing source dimensions")
    canonical = trajectory.copy()
    canonical[:, 0] *= 256.0 / width
    canonical[:, 1] *= 256.0 / height
    displacement = np.linalg.norm(canonical - canonical[0], axis=1)
    return {
        "initial_centroid_256": canonical[0].tolist(),
        "final_centroid_256": canonical[-1].tolist(),
        "max_displacement_px_256": float(np.max(displacement)),
        "final_displacement_px_256": float(displacement[-1]),
        "frames_tracked": int(len(canonical)),
    }


def robot_motion_gate_from_frames(
    frames: np.ndarray,
    *,
    threshold_px: float = 20.0,
    robotseg_root: str = "checkpoints/RobotSeg",
    robotseg_checkpoint: str = "checkpoints/robotseg.pt",
    device: str = "cuda",
) -> dict[str, Any]:
    if len(frames) < 3:
        raise RuntimeError("Task 4 robot-motion gate requires at least three frames")
    indices = np.round(np.linspace(0, len(frames) - 1, 3)).astype(np.int64)
    selected = frames[indices]
    segmentation = robotseg_masks_for_frames(
        selected,
        categories=("robot",),
        robotseg_root=robotseg_root,
        checkpoint=robotseg_checkpoint,
        device=device,
    )
    masks = np.asarray(segmentation["union_mask"], dtype=bool)
    centroids = []
    height, width = selected.shape[1:3]
    for frame_index, mask in zip(indices, masks):
        y, x = np.nonzero(mask)
        if len(x) == 0:
            raise RuntimeError(f"RobotSeg returned an empty mask at gate frame {int(frame_index)}")
        centroids.append([float(np.mean(x)) * 256.0 / width, float(np.mean(y)) * 256.0 / height])
    centroid_array = np.asarray(centroids, dtype=np.float64)
    displacement = np.linalg.norm(centroid_array - centroid_array[0], axis=1)
    maximum = float(np.max(displacement))
    return {
        "frame_indices": [int(index) for index in indices],
        "centroids_256": centroid_array.tolist(),
        "max_displacement_px_256": maximum,
        "threshold_px_256": float(threshold_px),
        "passed": int(maximum >= threshold_px),
        "robotseg_mask_fraction": float(segmentation["mask_fraction"]),
    }


def _evaluate_task4_row_legacy_model_vs_gt(
    row: dict[str, Any],
    root: Path,
    candidate_mode: str = "row-id",
    candidate_root: Path | None = None,
    candidate_field: str = "model_video",
    camera: str = "head_camera",
    point_radius: int = 10,
    point_grid: int = 3,
    max_frames: int = 96,
    device: str = "cuda",
    trajectory_threshold_px: float = 10.0,
    final_threshold_px: float = 10.0,
    visibility_threshold: float = 0.5,
    annotate_dir: Path | None = None,
    tapnextpp_checkpoint: str = "checkpoints/tapnextpp_ckpt.pt",
) -> dict[str, Any]:
    gt_video_path = resolve_path(root, row["output_video"])
    intrinsic, extrinsic = load_camera_for_row(row, root, camera)
    gt_frames_full, gt_info_full = load_video_frames(gt_video_path)
    if row.get("canonical_video_fps"):
        gt_info_full["fps"] = row.get("canonical_video_fps")

    if candidate_mode == "gt" or candidate_mode == "first":
        candidate_frames = None
        candidate_info = None
    else:
        path = candidate_video_path(
            row, root, candidate_mode, candidate_root=candidate_root, candidate_field=candidate_field
        )
        if path is None:
            raise RuntimeError("candidate path resolved to None")
        candidate_frames_raw, candidate_info = load_video_frames(path)
        candidate_info["mode"] = candidate_mode
        candidate_frames = resize_frames_like(candidate_frames_raw, gt_frames_full)
        if row.get("candidate_fps"):
            candidate_info["fps"] = row.get("candidate_fps")
            candidate_info["native_fps"] = row.get("candidate_fps")
        if row.get("candidate_frame_timestamps_sec"):
            candidate_info["frame_timestamps_sec"] = row.get("candidate_frame_timestamps_sec")

    gt_frames, gt_info = reference_frames_for_candidate(row, gt_frames_full, gt_info_full)
    if candidate_mode == "gt":
        candidate_frames = gt_frames.copy()
        candidate_info = {"mode": "gt", "path": str(gt_video_path), **gt_info}
    elif candidate_mode == "first":
        candidate_frames = np.repeat(gt_frames[:1], len(gt_frames), axis=0)
        candidate_info = {"mode": "first", "path": str(gt_video_path), **gt_info}
    if candidate_frames is None or candidate_info is None:
        raise RuntimeError("Task 4 candidate frames were not initialized")

    subset = task4_subset(row)
    target_initial_xyz, target_initial_pose_key = pose_xyz(
        row,
        ("receiver_target_initial_pose", "target_shifted_initial_pose", "target_initial_pose"),
    )
    target_final_xyz, target_final_pose_key = pose_xyz(
        row,
        (
            "receiver_target_final_pose",
            "target_final_pose",
            "target_shifted_preview_pose",
            "target_shifted_initial_pose",
        ),
    )
    distractor_initial_xyz, distractor_initial_pose_key = pose_xyz(row, ("distractor_initial_pose",))
    distractor_final_xyz, distractor_final_pose_key = pose_xyz(row, ("distractor_final_pose",))
    if target_initial_xyz is None or target_final_xyz is None:
        raise RuntimeError("Task 4 row is missing target pose metadata")
    has_distractor = distractor_initial_xyz is not None and distractor_final_xyz is not None

    target_initial_uv = project_xyz_to_uv(target_initial_xyz, intrinsic, extrinsic)
    target_final_uv = project_xyz_to_uv(target_final_xyz, intrinsic, extrinsic)
    distractor_initial_uv = (
        project_xyz_to_uv(distractor_initial_xyz, intrinsic, extrinsic) if has_distractor else None
    )
    distractor_final_uv = (
        project_xyz_to_uv(distractor_final_xyz, intrinsic, extrinsic) if has_distractor else None
    )

    gt_aligned, candidate_aligned, time_alignment = time_aligned_video_samples(
        gt_frames,
        gt_info,
        candidate_frames,
        candidate_info,
        max_frames=max_frames,
    )
    time_alignment["reference_frame_start_index_applied"] = gt_info.get(
        "reference_frame_start_index_applied", 0
    )
    time_alignment["full_reference_frame_count"] = gt_info.get("full_reference_frame_count", len(gt_frames))
    time_alignment["candidate_is_future_only"] = gt_info.get("candidate_is_future_only")

    aligned_height, aligned_width = gt_aligned.shape[1:3]
    target_query_points, target_query_source = explicit_query_points_from_row(
        row, "target", aligned_width, aligned_height
    )
    distractor_query_points, distractor_query_source = explicit_query_points_from_row(
        row, "distractor", aligned_width, aligned_height
    )

    gt_tracks = run_tapnextpp(
        gt_aligned,
        target_initial_uv[:2],
        distractor_initial_uv[:2] if distractor_initial_uv is not None else None,
        point_radius=point_radius,
        point_grid=point_grid,
        max_frames=0,
        device=device,
        visibility_threshold=visibility_threshold,
        checkpoint_path=tapnextpp_checkpoint,
        target_query_points_uv=target_query_points,
        distractor_query_points_uv=distractor_query_points,
    )
    candidate_tracks = run_tapnextpp(
        candidate_aligned,
        target_initial_uv[:2],
        distractor_initial_uv[:2] if distractor_initial_uv is not None else None,
        point_radius=point_radius,
        point_grid=point_grid,
        max_frames=0,
        device=device,
        visibility_threshold=visibility_threshold,
        checkpoint_path=tapnextpp_checkpoint,
        target_query_points_uv=target_query_points,
        distractor_query_points_uv=distractor_query_points,
    )
    common_length = min(
        len(gt_tracks["target"]["trajectory"]),
        len(candidate_tracks["target"]["trajectory"]),
    )
    if has_distractor:
        common_length = min(
            common_length,
            len(gt_tracks["distractor"]["trajectory"]),
            len(candidate_tracks["distractor"]["trajectory"]),
        )
    target = trajectory_errors(candidate_tracks["target"], gt_tracks["target"], common_length)
    target_accuracy = int(
        target["mean_error_px"] < trajectory_threshold_px and target["final_error_px"] < final_threshold_px
    )
    distractor = None
    distractor_accuracy = None
    reliability_values = [
        gt_tracks["target"]["visible_fraction"],
        candidate_tracks["target"]["visible_fraction"],
    ]
    if has_distractor:
        distractor = trajectory_errors(candidate_tracks["distractor"], gt_tracks["distractor"], common_length)
        distractor_accuracy = int(
            distractor["mean_error_px"] < trajectory_threshold_px
            and distractor["final_error_px"] < final_threshold_px
        )
        reliability_values.extend(
            [
                gt_tracks["distractor"]["visible_fraction"],
                candidate_tracks["distractor"]["visible_fraction"],
            ]
        )
    combined_accuracy, diagnostic_score, diagnostic_score_source = task4_scores(
        subset,
        target_accuracy,
        distractor_accuracy,
        has_distractor,
    )
    subset_score = diagnostic_score
    subset_score_source = diagnostic_score_source
    tracker_reliability = float(min(reliability_values))

    annotation_video = None
    if annotate_dir is not None:
        annotation_video = annotate_dir / f"{row.get('row_id', 'row')}_tapnextpp.mp4"
        n = min(len(gt_aligned), len(candidate_aligned), common_length)
        row_result = {"target": target}
        if distractor is not None:
            row_result["distractor"] = distractor
        draw_track_panel(
            gt_aligned[:n],
            candidate_aligned[:n],
            row_result,
            annotation_video,
            fps=float(min(time_alignment["gt_fps"], time_alignment["candidate_fps"])),
            gt_tracks=gt_tracks,
            candidate_tracks=candidate_tracks,
            visibility_threshold=visibility_threshold,
            tracker_label="TAPNext++",
        )

    def compact_track(track: dict[str, Any]) -> dict[str, Any]:
        heavy_keys = {"trajectory", "raw_trajectory", "raw_visibility", "raw_median_trajectory"}
        return {key: value for key, value in track.items() if key not in heavy_keys}

    distractor_pixels = None
    if distractor_initial_uv is not None and distractor_final_uv is not None:
        distractor_pixels = {
            "distractor_initial_uv": distractor_initial_uv[:2],
            "distractor_final_uv": distractor_final_uv[:2],
        }

    return json_safe(
        {
            "row_id": row.get("row_id"),
            "task": row.get("task"),
            "episode": row.get("episode"),
            "donor_episode": row.get("donor_episode"),
            "task4_subset": subset,
            "mode": "task4_tapnextpp_model_vs_gt",
            "tracker": "tapnextpp",
            "tapnextpp_checkpoint": tapnextpp_checkpoint,
            "camera": camera,
            "gt_video": gt_info,
            "candidate": candidate_info,
            "time_alignment": time_alignment,
            "gt_pixels": {
                "target_initial_uv": target_initial_uv[:2],
                "target_final_uv": target_final_uv[:2],
                **(distractor_pixels or {}),
            },
            "metadata_pose_keys": {
                "target_initial": target_initial_pose_key,
                "target_final": target_final_pose_key,
                "distractor_initial": distractor_initial_pose_key,
                "distractor_final": distractor_final_pose_key,
            },
            "query_points": {
                "target_source": target_query_source or "projected_pose_grid",
                "target_count": int(gt_tracks["target"]["query_count"]),
                "distractor_source": (
                    distractor_query_source or "projected_pose_grid" if has_distractor else None
                ),
                "distractor_count": (int(gt_tracks["distractor"]["query_count"]) if has_distractor else 0),
            },
            "target": {
                key: value for key, value in target.items() if key not in {"trajectory", "gt_trajectory"}
            },
            "distractor": None
            if distractor is None
            else {
                key: value for key, value in distractor.items() if key not in {"trajectory", "gt_trajectory"}
            },
            "target_track": {
                "gt": compact_track(gt_tracks["target"]),
                "candidate": compact_track(candidate_tracks["target"]),
            },
            "distractor_track": None
            if not has_distractor
            else {
                "gt": compact_track(gt_tracks["distractor"]),
                "candidate": compact_track(candidate_tracks["distractor"]),
            },
            "trajectory_threshold_px": trajectory_threshold_px,
            "final_threshold_px": final_threshold_px,
            "target_trajectory_accuracy": target_accuracy,
            "distractor_trajectory_accuracy": distractor_accuracy,
            "combined_trajectory_accuracy": combined_accuracy,
            "task4_subset_score": subset_score,
            "task4_subset_score_source": subset_score_source,
            "task4_subset_diagnostic_score": diagnostic_score,
            "task4_subset_diagnostic_score_source": diagnostic_score_source,
            "requires_distractor": int(has_distractor),
            "tracker_reliability": tracker_reliability,
            "annotation_video": str(annotation_video) if annotation_video else None,
        }
    )


def evaluate_task4_row_tapnextpp(
    row: dict[str, Any],
    root: Path,
    candidate_mode: str = "row-id",
    candidate_root: Path | None = None,
    candidate_field: str = "model_video",
    camera: str = "head_camera",
    point_radius: int = 10,
    point_grid: int = 3,
    max_frames: int = 0,
    device: str = "cuda",
    object_threshold_px: float = 10.0,
    robot_motion_threshold_px: float = 20.0,
    visibility_threshold: float = 0.5,
    tapnextpp_checkpoint: str = "checkpoints/tapnextpp_ckpt.pt",
    robotseg_root: str = "checkpoints/RobotSeg",
    robotseg_checkpoint: str = "checkpoints/robotseg.pt",
    annotate_dir: Path | None = None,
) -> dict[str, Any]:
    del annotate_dir
    gt_video_path = resolve_path(root, row["output_video"])
    gt_frames_full, gt_info_full = load_video_frames(gt_video_path)
    if row.get("canonical_video_fps"):
        gt_info_full["fps"] = row["canonical_video_fps"]
    gt_frames, gt_info = reference_frames_for_candidate(row, gt_frames_full, gt_info_full)

    if candidate_mode == "gt":
        candidate_frames = gt_frames.copy()
        candidate_info = {"mode": "gt", "path": str(gt_video_path), **gt_info}
    elif candidate_mode == "first":
        candidate_frames = np.repeat(gt_frames[:1], len(gt_frames), axis=0)
        candidate_info = {"mode": "first", "path": str(gt_video_path), **gt_info}
    else:
        candidate_path = candidate_video_path(
            row,
            root,
            candidate_mode,
            candidate_root=candidate_root,
            candidate_field=candidate_field,
        )
        if candidate_path is None:
            raise RuntimeError("candidate path resolved to None")
        candidate_frames_raw, candidate_info = load_video_frames(candidate_path)
        candidate_info["mode"] = candidate_mode
        candidate_frames = resize_frames_like(candidate_frames_raw, gt_frames)
        if row.get("candidate_fps"):
            candidate_info["fps"] = row["candidate_fps"]
            candidate_info["native_fps"] = row["candidate_fps"]
        if row.get("candidate_frame_timestamps_sec"):
            candidate_info["frame_timestamps_sec"] = row["candidate_frame_timestamps_sec"]

    _, candidate_aligned, time_alignment = time_aligned_video_samples(
        gt_frames,
        gt_info,
        candidate_frames,
        candidate_info,
        max_frames=max_frames,
    )
    time_alignment["reference_frame_start_index_applied"] = gt_info.get(
        "reference_frame_start_index_applied", 0
    )

    subset = task4_subset(row)
    object_name = evaluated_object_name(subset)
    pose_keys = (
        ("distractor_initial_pose",)
        if object_name == "distractor"
        else ("receiver_target_initial_pose", "target_shifted_initial_pose", "target_initial_pose")
    )
    initial_xyz, initial_pose_key = pose_xyz(row, pose_keys)
    if initial_xyz is None:
        raise RuntimeError(f"Task 4 {subset} row is missing {object_name} initial-pose metadata")

    intrinsic, extrinsic = load_camera_for_row(row, root, camera)
    initial_uv = project_xyz_to_uv(initial_xyz, intrinsic, extrinsic)
    height, width = candidate_aligned.shape[1:3]
    query_points, query_source = explicit_query_points_from_row(
        row,
        object_name,
        width,
        height,
    )
    tracks = run_tapnextpp(
        candidate_aligned,
        initial_uv[:2],
        None,
        point_radius=point_radius,
        point_grid=point_grid,
        max_frames=0,
        device=device,
        visibility_threshold=visibility_threshold,
        checkpoint_path=tapnextpp_checkpoint,
        target_query_points_uv=query_points,
    )
    if float(tracks["target"]["visible_fraction"]) <= 0.0:
        raise RuntimeError("TAPNext++ did not observe any valid object point")
    object_motion = canonical_object_displacement(tracks["target"])
    robot_gate = robot_motion_gate_from_frames(
        candidate_aligned,
        threshold_px=robot_motion_threshold_px,
        robotseg_root=robotseg_root,
        robotseg_checkpoint=robotseg_checkpoint,
        device=device,
    )
    score = task4_no_contact_score(
        object_motion["max_displacement_px_256"],
        robot_gate["max_displacement_px_256"],
        object_threshold_px=object_threshold_px,
        robot_threshold_px=robot_motion_threshold_px,
    )

    return json_safe(
        {
            "row_id": row.get("row_id"),
            "sample_id": row.get("sample_id") or row.get("row_id"),
            "task": row.get("task"),
            "episode": row.get("episode"),
            "task4_subset": subset,
            "mode": "task4_no_contact_displacement",
            "evaluated_object": object_name,
            "evaluated_object_initial_pose_key": initial_pose_key,
            "evaluated_object_initial_uv": initial_uv[:2],
            "query_source": query_source or "projected_pose_grid",
            "query_count": int(tracks["target"]["query_count"]),
            "query_grid_geometry": "true_radius_inscribed_square",
            "tracker": "tapnextpp",
            "tapnextpp_checkpoint": tapnextpp_checkpoint,
            "tracker_reliability": float(tracks["target"]["visible_fraction"]),
            "object_motion": object_motion,
            "object_threshold_px_256": float(object_threshold_px),
            "object_static_pass": score["object_static_pass"],
            "robot_motion_gate": robot_gate,
            "robot_motion_gate_pass": score["robot_motion_gate_pass"],
            "task4_pass": score["task4_pass"],
            "task4_score": score["task4_score"],
            "task4_subset_score": score["task4_score"],
            "candidate": candidate_info,
            "time_alignment": time_alignment,
        }
    )


def evaluate_task4_manifest_tapnextpp(
    manifest: Path,
    root: Path,
    limit: int | None = None,
    candidate_mode: str = "row-id",
    candidate_root: Path | None = None,
    candidate_field: str = "model_video",
    camera: str = "head_camera",
    point_radius: int = 10,
    point_grid: int = 3,
    max_frames: int = 0,
    device: str = "cuda",
    trajectory_threshold_px: float = 10.0,
    final_threshold_px: float = 10.0,
    robot_motion_threshold_px: float = 20.0,
    visibility_threshold: float = 0.5,
    annotate_dir: Path | None = None,
    tapnextpp_checkpoint: str = "checkpoints/tapnextpp_ckpt.pt",
    robotseg_root: str = "checkpoints/RobotSeg",
    robotseg_checkpoint: str = "checkpoints/robotseg.pt",
    allowed_subsets: tuple[str, ...] = (
        "distractor_hallucination",
        "fake_contact_hallucination",
        "proximity_hallucination",
    ),
) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    unsupported = [
        str(row.get("row_id"))
        for row in rows
        if allowed_subsets and task4_subset(row) not in allowed_subsets
    ]
    if unsupported:
        raise ValueError(f"Task 4 manifest contains unsupported conditions: {unsupported[:10]}")
    if limit is not None:
        rows = rows[:limit]
    results = []
    errors = []
    for row in rows:
        try:
            results.append(
                evaluate_task4_row_tapnextpp(
                    row,
                    root,
                    candidate_mode=candidate_mode,
                    candidate_root=candidate_root,
                    candidate_field=candidate_field,
                    camera=camera,
                    point_radius=point_radius,
                    point_grid=point_grid,
                    max_frames=max_frames,
                    device=device,
                    object_threshold_px=trajectory_threshold_px,
                    robot_motion_threshold_px=robot_motion_threshold_px,
                    visibility_threshold=visibility_threshold,
                    annotate_dir=annotate_dir,
                    tapnextpp_checkpoint=tapnextpp_checkpoint,
                    robotseg_root=robotseg_root,
                    robotseg_checkpoint=robotseg_checkpoint,
                )
            )
        except Exception as exc:
            error = {
                "row_id": row.get("row_id"),
                "sample_id": row.get("sample_id") or row.get("row_id"),
                "task": row.get("task"),
                "error": repr(exc),
            }
            errors.append(error)
            results.append(
                {
                    **error,
                    "sample_id": row.get("sample_id") or row.get("row_id"),
                    "task4_subset": task4_subset(row),
                    "mode": "task4_no_contact_displacement",
                    "evaluation_failed": 1,
                    "object_static_pass": 0,
                    "robot_motion_gate_pass": 0,
                    "task4_pass": 0,
                    "task4_score": 0.0,
                    "task4_subset_score": 0.0,
                }
            )

    def mean(path: tuple[str, ...] | str, items: list[dict[str, Any]] | None = None) -> float | None:
        selected = results if items is None else items
        if not selected:
            return None
        values = finite_values(selected, path)
        return float(np.mean(values)) if values else None

    def subset_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows_scored": len(items),
            "score_0_to_100": mean("task4_score", items),
            "object_static_pass_rate": mean("object_static_pass", items),
            "robot_motion_gate_pass_rate": mean("robot_motion_gate_pass", items),
            "mean_object_max_displacement_px_256": mean(
                ("object_motion", "max_displacement_px_256"), items
            ),
            "mean_tracker_reliability": mean("tracker_reliability", items),
        }

    subset_order = (
        "distractor_hallucination",
        "proximity_hallucination",
        "fake_contact_hallucination",
    )
    subset_rows = {
        subset: [row for row in results if row.get("task4_subset") == subset] for subset in subset_order
    }
    extra_subsets = sorted(
        {str(row.get("task4_subset")) for row in results if row.get("task4_subset") not in subset_rows}
    )
    for subset in extra_subsets:
        subset_rows[subset] = [row for row in results if row.get("task4_subset") == subset]
    subset_summaries = {subset: subset_summary(items) for subset, items in subset_rows.items() if items}

    return json_safe(
        {
            "manifest": str(manifest),
            "root": str(root),
            "candidate_mode": candidate_mode,
            "candidate_root": str(candidate_root) if candidate_root else None,
            "candidate_field": candidate_field,
            "mode": "task4_no_contact_displacement",
            "tracker": "tapnextpp",
            "tapnextpp_checkpoint": tapnextpp_checkpoint,
            "robot_motion_gate": "robotseg_centroid_three_frames",
            "robotseg_checkpoint": robotseg_checkpoint,
            "point_radius_px_256": float(point_radius),
            "point_grid": int(point_grid),
            "point_grid_geometry": "true_radius_inscribed_square",
            "object_threshold_px_256": float(trajectory_threshold_px),
            "robot_motion_threshold_px_256": float(robot_motion_threshold_px),
            "allowed_subsets": list(allowed_subsets),
            "rows_requested": len(rows),
            "rows_scored": len(results),
            "errors": errors,
            "summary": {
                "overall_score": mean("task4_score"),
                "object_static_pass_rate": mean("object_static_pass"),
                "robot_motion_gate_pass_rate": mean("robot_motion_gate_pass"),
                "distractor_hallucination_score": subset_summaries.get("distractor_hallucination", {}).get(
                    "score_0_to_100"
                ),
                "proximity_hallucination_score": subset_summaries.get("proximity_hallucination", {}).get(
                    "score_0_to_100"
                ),
                "fake_contact_hallucination_score": subset_summaries.get(
                    "fake_contact_hallucination", {}
                ).get("score_0_to_100"),
                "mean_object_max_displacement_px_256": mean(
                    ("object_motion", "max_displacement_px_256")
                ),
                "mean_tracker_reliability": mean("tracker_reliability"),
                "subsets": subset_summaries,
            },
            "rows": results,
        }
    )
