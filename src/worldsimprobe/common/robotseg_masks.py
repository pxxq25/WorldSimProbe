from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_robotseg_import(robotseg_root: Path) -> None:
    import sys

    root = str(robotseg_root)
    if root not in sys.path:
        sys.path.insert(0, root)


@lru_cache(maxsize=1)
def load_robotseg_predictor(
    robotseg_root: str = "checkpoints/RobotSeg",
    checkpoint: str = "checkpoints/robotseg.pt",
    device: str = "cuda",
) -> Any:
    robotseg_root_path = Path(robotseg_root)
    _ensure_robotseg_import(robotseg_root_path)
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from robotseg.build_robotseg import build_robotseg_video_predictor

    config_dir = str(robotseg_root_path / "robotseg" / "configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        predictor = build_robotseg_video_predictor("robotseg-infer", checkpoint, device=device)
    return predictor


def _write_frame_dir(frames: np.ndarray, directory: Path) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(directory / f"{idx:06d}.jpg"), bgr)


def _resize_bool_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    resized = []
    for mask in masks:
        out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        resized.append(out.astype(bool))
    return np.stack(resized, axis=0)


def _auto_robot_seed_frame(frames: np.ndarray) -> int:
    """Pick a frame where the robot is likely visible for RobotSeg auto prompting."""
    import cv2

    scores = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        value = hsv[..., 2]
        sat = hsv[..., 1]
        # Robot arms/grippers in RoboTwin are mostly dark/black/gray. This also sees
        # shadows, so use it only to choose a seed frame, not as the final mask.
        dark_neutral = (value < 95) & (sat < 120)
        scores.append(float(np.mean(dark_neutral)))
    if not scores:
        return 0
    return int(np.argmax(scores))


def robotseg_masks_for_frames(
    frames: np.ndarray,
    categories: tuple[str, ...] = ("robot",),
    robotseg_root: str = "checkpoints/RobotSeg",
    checkpoint: str = "checkpoints/robotseg.pt",
    device: str = "cuda",
    seed_frame_idx: int | str = "auto",
) -> dict[str, Any]:
    import torch

    predictor = load_robotseg_predictor(robotseg_root=robotseg_root, checkpoint=checkpoint, device=device)
    results_by_category: dict[str, list[np.ndarray]] = {}
    with tempfile.TemporaryDirectory(prefix="actbench_robotseg_") as tmp:
        frame_dir = Path(tmp) / "frames"
        _write_frame_dir(frames, frame_dir)
        if seed_frame_idx == "auto":
            start_frame_idx = _auto_robot_seed_frame(frames)
        else:
            start_frame_idx = int(seed_frame_idx)
            start_frame_idx = max(0, min(start_frame_idx, len(frames) - 1))
        for category in categories:
            per_frame: dict[int, np.ndarray] = {}
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")),
            ):
                state = predictor.init_state(
                    video_path=str(frame_dir),
                    async_loading_frames=False,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                )
                predictor.add_new_robot(
                    inference_state=state,
                    frame_idx=start_frame_idx,
                    obj_id=0,
                    robot=category,
                )
                for reverse in (False, True):
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        inference_state=state,
                        robot=category,
                        start_frame_idx=start_frame_idx,
                        reverse=reverse,
                    ):
                        if 0 not in out_obj_ids:
                            continue
                        obj_index = list(out_obj_ids).index(0)
                        mask = (out_mask_logits[obj_index] > 0.0).detach().cpu().numpy()
                        per_frame[int(out_frame_idx)] = np.squeeze(mask).astype(bool)
            fallback_shape = frames[0].shape[:2]
            category_masks = []
            for idx in range(len(frames)):
                mask = per_frame.get(idx)
                if mask is None:
                    mask = np.zeros(fallback_shape, dtype=bool)
                category_masks.append(mask)
            results_by_category[category] = category_masks

    union = None
    category_arrays: dict[str, np.ndarray] = {}
    for category, masks in results_by_category.items():
        arr = np.stack(masks, axis=0).astype(bool)
        category_arrays[category] = arr
        union = arr if union is None else (union | arr)
    if union is None:
        union = np.zeros(frames.shape[:3], dtype=bool)
    return {
        "categories": list(categories),
        "seed_frame_idx": start_frame_idx,
        "category_masks": category_arrays,
        "union_mask": union,
        "mask_fraction": float(np.mean(union)),
    }


def flow_masks_from_frame_masks(frame_masks: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(frame_masks) < 2:
        raise RuntimeError("need at least two frame masks to derive flow masks")
    resized = _resize_bool_masks(frame_masks, width=width, height=height)
    return resized[:-1] | resized[1:]
