from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.common.tasks import normalize_task_id, required_video_roles
from worldsimprobe.common.video import VIDEO_SUFFIXES, probe_video
from worldsimprobe.submission.video_config import VideoTimingConfig, load_video_timing_config

SUBMISSION_SCHEMA = json.loads(
    files("worldsimprobe.schemas").joinpath("submission.schema.json").read_text(encoding="utf-8")
)
SCHEMA_VALIDATOR = Draft202012Validator(SUBMISSION_SCHEMA)


def _schema_errors(row: dict[str, Any]) -> list[str]:
    errors = sorted(SCHEMA_VALIDATOR.iter_errors(row), key=lambda error: list(error.absolute_path))
    result = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<row>"
        result.append(f"schema {location}: {error.message}")
    return result


def resolve_relative_video(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("video paths must be relative to the submission root")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"video path escapes the submission root: {value}")
    return resolved


def _validate_row(
    row: dict[str, Any],
    root: Path,
    *,
    decode: bool,
    video_config: VideoTimingConfig,
) -> tuple[dict[str, Any], list[str]]:
    errors = _schema_errors(row)
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append("sample_id must be a non-empty string")

    try:
        task_id = normalize_task_id(str(row.get("task_id", "")))
    except ValueError as exc:
        task_id = None
        errors.append(str(exc))

    model_id = row.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append("model_id must be a non-empty string")

    videos = row.get("videos")
    metadata: dict[str, Any] = {}
    if not isinstance(videos, dict):
        errors.append("videos must be an object")
    elif task_id is not None:
        required = required_video_roles(task_id)
        missing = [role for role in required if role not in videos]
        extra = sorted(set(videos) - set(required))
        if missing:
            errors.append(f"videos: missing video roles: {missing}")
        if extra:
            errors.append(f"videos: unexpected video roles: {extra}")
        for role in required:
            value = videos.get(role)
            if not isinstance(value, str) or not value:
                continue
            try:
                path = resolve_relative_video(root, value)
            except ValueError as exc:
                errors.append(f"videos/{role}: {exc}")
                continue
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                errors.append(f"videos/{role}: unsupported video extension {path.suffix}")
            if not path.is_file():
                errors.append(f"videos/{role}: video not found: {value}")
                continue
            if decode:
                try:
                    decoded = probe_video(path)
                    video_config.validate_fps(decoded.get("fps"))
                    metadata[role] = decoded
                except Exception as exc:
                    errors.append(f"videos/{role}: video validation failed: {type(exc).__name__}: {exc}")

    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "model_id": model_id,
        "video_metadata": metadata,
    }, errors


def validate_submission(
    manifest: Path,
    root: Path,
    *,
    decode: bool = False,
    video_config: VideoTimingConfig | None = None,
) -> dict[str, Any]:
    timing = video_config or load_video_timing_config()
    rows = read_jsonl(manifest)
    seen: set[str] = set()
    results = []
    errors = []
    for index, row in enumerate(rows, 1):
        result, row_errors = _validate_row(row, root, decode=decode, video_config=timing)
        sample_id = result.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in seen:
                row_errors.append(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
        results.append(result)
        if row_errors:
            errors.append(
                {
                    "line": index,
                    "sample_id": sample_id,
                    "errors": row_errors,
                }
            )
    return {
        "manifest": str(manifest),
        "root": str(root),
        "rows": len(rows),
        "valid_rows": len(rows) - len(errors),
        "error_rows": len(errors),
        "decode_checked": bool(decode),
        "video_timing_config": timing.as_dict(),
        "passes": not errors and bool(rows),
        "errors": errors,
        "results": results,
    }


def require_valid_submission(
    manifest: Path,
    root: Path,
    *,
    decode: bool = False,
    video_config: VideoTimingConfig | None = None,
) -> dict[str, Any]:
    result = validate_submission(
        manifest,
        root,
        decode=decode,
        video_config=video_config,
    )
    if not result["passes"]:
        preview = json.dumps(result["errors"][:10], sort_keys=True)
        raise ValueError(f"invalid submission manifest: {preview}")
    return result
