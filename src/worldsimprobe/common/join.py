from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.common.tasks import normalize_task_id
from worldsimprobe.submission.validator import resolve_relative_video
from worldsimprobe.submission.video_config import VideoTimingConfig, load_video_timing_config


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("row_id") or "")
        if not sample_id:
            raise ValueError(f"{label} row is missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        result[sample_id] = row
    return result


def join_references_with_submission(
    reference_manifest: Path,
    submission_manifest: Path,
    submission_root: Path,
    *,
    task_id: str | None = None,
    video_config: VideoTimingConfig | None = None,
) -> list[dict[str, Any]]:
    configured_timing = video_config or load_video_timing_config()
    references = read_jsonl(reference_manifest)
    submissions = _index(read_jsonl(submission_manifest), "submission")
    requested_task = normalize_task_id(task_id) if task_id else None
    joined: list[dict[str, Any]] = []
    missing_sample_ids: list[str] = []

    for reference in references:
        sample_id = str(reference.get("sample_id") or reference.get("row_id") or "")
        if not sample_id:
            raise ValueError("reference row is missing sample_id")
        row_task = normalize_task_id(str(reference.get("worldsimprobe_task_id") or reference.get("task_id")))
        if requested_task and row_task != requested_task:
            continue
        submission = submissions.get(sample_id)
        if submission is None:
            missing_sample_ids.append(sample_id)
            continue
        if normalize_task_id(str(submission.get("task_id"))) != row_task:
            raise ValueError(f"task mismatch for {sample_id}")

        row = deepcopy(reference)
        row["sample_id"] = sample_id
        row["row_id"] = sample_id
        row["worldsimprobe_task_id"] = row_task
        row["submission_model_id"] = submission.get("model_id")
        row["evaluation_id"] = sample_id
        videos = {
            role: str(resolve_relative_video(submission_root, value))
            for role, value in submission["videos"].items()
        }
        timing = {role: {"fps": configured_timing.fps} for role in videos}
        row["videos"] = videos
        row["video_timing"] = timing
        row["video_timing_source"] = "evaluator_config"
        if row_task != "task1":
            row["candidate_video"] = videos["candidate"]
            candidate_timing = timing.get("candidate")
            row["candidate_video_timing"] = candidate_timing
            row["candidate_fps"] = configured_timing.fps
        joined.append(row)

    if missing_sample_ids:
        preview = ", ".join(missing_sample_ids[:10])
        remainder = len(missing_sample_ids) - 10
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        task_label = requested_task or "requested evaluation"
        raise ValueError(
            f"submission is missing {len(missing_sample_ids)} assigned {task_label} samples: "
            f"{preview}{suffix}"
        )
    return joined
