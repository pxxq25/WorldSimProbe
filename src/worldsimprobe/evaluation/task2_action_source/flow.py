from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldsimprobe.common.manifest import read_jsonl


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


def resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_hdf5_rgb(path: Path, camera: str = "head_camera", max_frames: int | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for Task 2 flow evaluation") from exc

    with h5py.File(path, "r") as f:
        key = f"observation/{camera}/rgb"
        raw = f[key][()]
    frames = []
    for item in raw[:max_frames]:
        data = item if isinstance(item, bytes) else bytes(item)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(frames) < 2:
        raise RuntimeError(f"not enough decodable HDF5 frames: {path}")
    return np.stack(frames, axis=0)


def load_video_rgb(path: Path, max_frames: int | None = None) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for Task 2 flow evaluation") from exc

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"not enough decodable video frames: {path}")
    return np.stack(frames, axis=0)


def load_rgb(path: Path, camera: str = "head_camera", max_frames: int | None = None) -> np.ndarray:
    if path.suffix.lower() in {".hdf5", ".h5"}:
        return load_hdf5_rgb(path, camera=camera, max_frames=max_frames)
    return load_video_rgb(path, max_frames=max_frames)


def resize_gray(frame_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    resized = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)


def optical_flow_sequence(
    frames: np.ndarray,
    width: int = 160,
    height: int = 120,
    stride: int = 1,
) -> np.ndarray:
    import cv2

    if stride > 1:
        frames = frames[::stride]
    grays = [resize_gray(frame, (width, height)) for frame in frames]
    flows = []
    for prev, nxt in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow.astype(np.float32))
    if not flows:
        raise RuntimeError("not enough frames to compute optical flow")
    return np.stack(flows, axis=0)


def resample_flow(flow: np.ndarray, n: int) -> np.ndarray:
    if len(flow) == n:
        return flow
    if len(flow) == 1:
        return np.repeat(flow, n, axis=0)
    indices = np.linspace(0, len(flow) - 1, n)
    lo = np.floor(indices).astype(int)
    hi = np.ceil(indices).astype(int)
    alpha = (indices - lo).astype(np.float32)
    return (1.0 - alpha)[:, None, None, None] * flow[lo] + alpha[:, None, None, None] * flow[hi]


def flow_distance(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = candidate - reference
    sq = np.sum(diff * diff, axis=-1)
    if mask.any():
        return float(np.sqrt(np.mean(sq[mask])))
    return float(np.sqrt(np.mean(sq)))


def flow_preference_metrics(
    candidate_flow: np.ndarray,
    donor_flow: np.ndarray,
    receiver_flow: np.ndarray,
    motion_threshold: float = 0.25,
    margin: float = 0.02,
) -> dict[str, Any]:
    n = min(len(candidate_flow), len(donor_flow), len(receiver_flow))
    candidate = resample_flow(candidate_flow, n)
    donor = resample_flow(donor_flow, n)
    receiver = resample_flow(receiver_flow, n)

    candidate_mag = np.linalg.norm(candidate, axis=-1)
    donor_mag = np.linalg.norm(donor, axis=-1)
    receiver_mag = np.linalg.norm(receiver, axis=-1)
    active_mask = (
        (candidate_mag > motion_threshold)
        | (donor_mag > motion_threshold)
        | (receiver_mag > motion_threshold)
    )

    donor_error = flow_distance(candidate, donor, active_mask)
    receiver_error = flow_distance(candidate, receiver, active_mask)
    all_donor_error = flow_distance(candidate, donor, np.ones_like(active_mask, dtype=bool))
    all_receiver_error = flow_distance(candidate, receiver, np.ones_like(active_mask, dtype=bool))
    donor_preference = int(donor_error + margin < receiver_error)
    return {
        "donor_flow_error": donor_error,
        "receiver_flow_error": receiver_error,
        "all_pixel_donor_flow_error": all_donor_error,
        "all_pixel_receiver_flow_error": all_receiver_error,
        "donor_motion_preference": donor_preference,
        "flow_error_gap_receiver_minus_donor": receiver_error - donor_error,
        "active_fraction": float(np.mean(active_mask)),
        "frames_compared": int(n),
        "motion_threshold": motion_threshold,
        "margin": margin,
    }


def receiver_hdf5_for_row(row: dict[str, Any]) -> str:
    task = row["receiver_task"]
    config = row.get("source_task_config", "cross_clean_50")
    episode = int(row["episode"])
    return f"data/{task}/{config}/data/episode{episode}.hdf5"


def candidate_path_for_mode(row: dict[str, Any], mode: str) -> str:
    if mode == "output":
        return row.get("candidate_video") or row.get("output_video") or row["output_hdf5"]
    if mode == "output_hdf5":
        return row["output_hdf5"]
    if mode == "donor":
        return row["donor_hdf5"]
    if mode == "receiver":
        return receiver_hdf5_for_row(row)
    raise ValueError(f"unknown candidate mode: {mode}")


def evaluate_task2_row_flow(
    row: dict[str, Any],
    root: Path,
    candidate_mode: str = "output",
    camera: str = "head_camera",
    width: int = 160,
    height: int = 120,
    stride: int = 1,
    max_frames: int | None = None,
    motion_threshold: float = 0.25,
    margin: float = 0.02,
) -> dict[str, Any]:
    candidate_path = resolve_path(root, candidate_path_for_mode(row, candidate_mode))
    donor_path = resolve_path(root, row["donor_hdf5"])
    receiver_path = resolve_path(root, receiver_hdf5_for_row(row))

    candidate_frames = load_rgb(candidate_path, camera=camera, max_frames=max_frames)
    donor_frames = load_rgb(donor_path, camera=camera, max_frames=max_frames)
    receiver_frames = load_rgb(receiver_path, camera=camera, max_frames=max_frames)
    candidate_flow = optical_flow_sequence(candidate_frames, width=width, height=height, stride=stride)
    donor_flow = optical_flow_sequence(donor_frames, width=width, height=height, stride=stride)
    receiver_flow = optical_flow_sequence(receiver_frames, width=width, height=height, stride=stride)
    metrics = flow_preference_metrics(
        candidate_flow,
        donor_flow,
        receiver_flow,
        motion_threshold=motion_threshold,
        margin=margin,
    )
    return json_safe(
        {
            "row_id": row.get("row_id"),
            "receiver_task": row.get("receiver_task"),
            "donor_task": row.get("donor_task"),
            "episode": row.get("episode"),
            "candidate_mode": candidate_mode,
            "candidate_path": str(candidate_path),
            "donor_path": str(donor_path),
            "receiver_path": str(receiver_path),
            "candidate_frames": len(candidate_frames),
            "donor_frames": len(donor_frames),
            "receiver_frames": len(receiver_frames),
            **metrics,
        }
    )


def evaluate_task2_manifest_flow(
    manifest: Path,
    root: Path,
    candidate_mode: str = "output",
    limit: int | None = None,
    row_ids: set[str] | None = None,
    camera: str = "head_camera",
    width: int = 160,
    height: int = 120,
    stride: int = 1,
    max_frames: int | None = None,
    motion_threshold: float = 0.25,
    margin: float = 0.02,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest)]
    if row_ids is not None:
        rows = [row for row in rows if str(row.get("row_id")) in row_ids]
    if limit is not None:
        rows = rows[:limit]

    results = []
    errors = []
    for row in rows:
        try:
            results.append(
                evaluate_task2_row_flow(
                    row,
                    root,
                    candidate_mode=candidate_mode,
                    camera=camera,
                    width=width,
                    height=height,
                    stride=stride,
                    max_frames=max_frames,
                    motion_threshold=motion_threshold,
                    margin=margin,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "row_id": row.get("row_id"),
                    "receiver_task": row.get("receiver_task"),
                    "donor_task": row.get("donor_task"),
                    "episode": row.get("episode"),
                    "error": repr(exc),
                }
            )

    def mean(key: str) -> float | None:
        if not results:
            return None
        return float(np.mean([item[key] for item in results]))

    return json_safe(
        {
            "manifest": str(manifest),
            "root": str(root),
            "candidate_mode": candidate_mode,
            "rows_requested": len(rows),
            "rows_scored": len(results),
            "errors": errors,
            "summary": {
                "donor_motion_preference": mean("donor_motion_preference"),
                "mean_donor_flow_error": mean("donor_flow_error"),
                "mean_receiver_flow_error": mean("receiver_flow_error"),
                "mean_flow_error_gap_receiver_minus_donor": mean("flow_error_gap_receiver_minus_donor"),
                "mean_active_fraction": mean("active_fraction"),
            },
            "rows": results,
        }
    )
