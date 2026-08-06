from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from worldsimprobe.common.timebase import time_align_frame_pair

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def _read_video_imageio(path: Path, max_frames: int | None) -> tuple[list[np.ndarray], float | None]:
    reader = imageio.get_reader(str(path))
    frames: list[np.ndarray] = []
    try:
        metadata = reader.get_meta_data()
        for index, frame in enumerate(reader):
            if max_frames is not None and index >= max_frames:
                break
            array = np.asarray(frame)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            frames.append(array[..., :3].astype(np.uint8))
    finally:
        reader.close()
    return frames, metadata.get("fps")


def _read_video_opencv(path: Path, max_frames: int | None) -> tuple[list[np.ndarray], float | None]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV fallback is unavailable") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames, fps


def read_video_rgb(path: Path, max_frames: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        frames, fps = _read_video_imageio(path, max_frames)
    except Exception as imageio_error:
        try:
            frames, fps = _read_video_opencv(path, max_frames)
        except Exception as opencv_error:
            raise RuntimeError(
                f"video decode failed for {path}; "
                f"imageio={type(imageio_error).__name__}: {imageio_error}; "
                f"opencv={type(opencv_error).__name__}: {opencv_error}"
            ) from opencv_error

    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = None
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        fps = None
    array = np.stack(frames, axis=0)
    return array, {
        "fps": fps,
        "frame_count": len(array),
        "height": int(array.shape[1]),
        "width": int(array.shape[2]),
        "duration_sec": (float((len(array) - 1) / fps) if fps and len(array) > 1 else 0.0),
    }


def probe_video(path: Path) -> dict[str, Any]:
    frames, metadata = read_video_rgb(path)
    del frames
    return metadata


def probe_video_timing(path: Path) -> dict[str, Any]:
    """Read decoded timing metadata without loading the full video into memory."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for full-horizon validation") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, _ = capture.read()
    finally:
        capture.release()
    if not ok or frame_count <= 0:
        raise RuntimeError(f"video has no decodable frames: {path}")
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"video has invalid decoded fps={fps}: {path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": float((frame_count - 1) / fps) if frame_count > 1 else 0.0,
    }


def resize_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame[..., :3]
    return np.asarray(
        Image.fromarray(frame[..., :3]).resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def time_aligned_video_mse(
    first_path: Path,
    second_path: Path,
    first_timing: dict[str, Any] | None = None,
    second_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first, first_file_info = read_video_rgb(first_path)
    second, second_file_info = read_video_rgb(second_path)
    first_info = {**first_file_info, **(first_timing or {})}
    second_info = {**second_file_info, **(second_timing or {})}
    if not first_info.get("fps") and not first_info.get("frame_timestamps_sec"):
        raise ValueError(f"missing timing metadata for {first_path}")
    if not second_info.get("fps") and not second_info.get("frame_timestamps_sec"):
        raise ValueError(f"missing timing metadata for {second_path}")

    first_aligned, second_aligned, alignment = time_align_frame_pair(
        reference_frames=first,
        candidate_frames=second,
        reference_info=first_info,
        candidate_info=second_info,
    )
    height, width = first_aligned[0].shape[:2]
    squared_error = 0.0
    for first_frame, second_frame in zip(first_aligned, second_aligned):
        a = resize_rgb(first_frame, height, width).astype(np.float32)
        b = resize_rgb(second_frame, height, width).astype(np.float32)
        squared_error += float(np.mean(np.square(a - b)))
    return {
        "mse": squared_error / len(first_aligned),
        "frames": len(first_aligned),
        "first_frame_count": len(first),
        "second_frame_count": len(second),
        "time_alignment": alignment,
    }
