from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VideoTimingConfig:
    fps: float
    fps_tolerance: float

    def validate_fps(self, actual_fps: Any) -> float:
        try:
            actual = float(actual_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decoded video has invalid fps={actual_fps!r}") from exc
        if not math.isfinite(actual) or actual <= 0.0:
            raise ValueError(f"decoded video has invalid fps={actual!r}")
        if abs(actual - self.fps) > self.fps_tolerance:
            raise ValueError(
                f"decoded fps {actual:.6g} does not match configured fps {self.fps:.6g} "
                f"(tolerance {self.fps_tolerance:.6g})"
            )
        return actual

    def as_dict(self) -> dict[str, float]:
        return {"fps": self.fps, "fps_tolerance": self.fps_tolerance}


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a finite {qualifier} number")
    return result


def load_video_timing_config(path: Path | None = None) -> VideoTimingConfig:
    if path is None:
        text = files("worldsimprobe.configs").joinpath("video.yaml").read_text(encoding="utf-8")
        source = "packaged worldsimprobe.configs/video.yaml"
    else:
        source = str(path)
        text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    video = raw.get("video") if isinstance(raw, dict) else None
    if not isinstance(video, dict):
        raise ValueError(f"{source}: expected a 'video' mapping")
    return VideoTimingConfig(
        fps=_positive_number(video.get("fps"), f"{source}: video.fps"),
        fps_tolerance=_positive_number(
            video.get("fps_tolerance", 0.01),
            f"{source}: video.fps_tolerance",
            allow_zero=True,
        ),
    )


def default_video_fps() -> float:
    return load_video_timing_config().fps
