#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FPS = 10.0
FRAME_COUNT = 24
SIZE = 96


def frame(
    index: int,
    *,
    object_delta: float,
    arm_delta: float,
    distractor: bool = False,
) -> np.ndarray:
    image = Image.new("RGB", (SIZE, SIZE), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 63, SIZE, SIZE), fill=(183, 157, 126))
    progress = index / (FRAME_COUNT - 1)
    arm_x = int(27 + arm_delta * progress)
    object_x = int(55 + object_delta * progress)
    draw.line((16, 18, arm_x, 48), fill=(68, 83, 99), width=7)
    draw.rounded_rectangle((arm_x - 7, 43, arm_x + 7, 56), radius=2, fill=(47, 58, 69))
    draw.rectangle((object_x - 6, 53, object_x + 6, 64), fill=(32, 128, 160))
    if distractor:
        draw.ellipse((71, 51, 84, 64), fill=(200, 68, 78))
    return np.asarray(image, dtype=np.uint8)


def write_video(path: Path, *, object_delta: float, arm_delta: float, distractor: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the example videos")
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{SIZE}x{SIZE}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for index in range(FRAME_COUNT):
        process.stdin.write(
            frame(
                index,
                object_delta=object_delta,
                arm_delta=arm_delta,
                distractor=distractor,
            ).tobytes()
        )
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr.strip()}")


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "examples"
    videos = root / "example_submission" / "videos"
    specifications = {
        "example-task1__original.mp4": (0.0, 14.0, False),
        "example-task1__small.mp4": (3.0, 17.0, False),
        "example-task1__large.mp4": (12.0, 26.0, False),
        "example-task2.mp4": (9.0, 22.0, False),
        "example-task3.mp4": (-7.0, -18.0, False),
        "example-task4.mp4": (8.0, 20.0, True),
        "example-task5.mp4": (14.0, 27.0, False),
    }
    for name, (object_delta, arm_delta, distractor) in specifications.items():
        write_video(
            videos / name,
            object_delta=object_delta,
            arm_delta=arm_delta,
            distractor=distractor,
        )

    rows = [
        {
            "schema_version": "1.0",
            "sample_id": "example-task1",
            "task_id": "task1",
            "model_id": "example-model",
            "videos": {
                "original": "videos/example-task1__original.mp4",
                "small": "videos/example-task1__small.mp4",
                "large": "videos/example-task1__large.mp4",
            },
        }
    ]
    for task_id in ("task2", "task3", "task4", "task5"):
        rows.append(
            {
                "schema_version": "1.0",
                "sample_id": f"example-{task_id}",
                "task_id": task_id,
                "model_id": "example-model",
                "videos": {"candidate": f"videos/example-{task_id}.mp4"},
            }
        )
    manifest = root / "example_submission" / "submission.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
