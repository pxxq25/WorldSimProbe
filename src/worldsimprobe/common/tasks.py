from __future__ import annotations

from typing import Final

TASK_IDS: Final[tuple[str, ...]] = ("task1", "task2", "task3", "task4", "task5")
TASK_NAMES: Final[dict[str, str]] = {
    "task1": "Action Calibration",
    "task2": "Action Source",
    "task3": "Action Coverage",
    "task4": "Interaction Grounding",
    "task5": "Interaction Dynamics",
}
TASK1_VIDEO_ROLES: Final[tuple[str, ...]] = ("original", "small", "large")
DEFAULT_VIDEO_ROLES: Final[tuple[str, ...]] = ("candidate",)


def normalize_task_id(value: str) -> str:
    key = value.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "task1",
        "task1": "task1",
        "2": "task2",
        "task2": "task2",
        "3": "task3",
        "task3": "task3",
        "4": "task4",
        "task4": "task4",
        "5": "task5",
        "task5": "task5",
    }
    if key not in aliases:
        raise ValueError(f"unknown WorldSimProbe task: {value!r}")
    return aliases[key]


def required_video_roles(task_id: str) -> tuple[str, ...]:
    return TASK1_VIDEO_ROLES if normalize_task_id(task_id) == "task1" else DEFAULT_VIDEO_ROLES
