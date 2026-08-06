from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.evaluation.task2_action_source.flow import (
    candidate_path_for_mode,
    json_safe,
    load_rgb,
    resolve_path,
)


def save_sampled_frames(frames: np.ndarray, out_dir: Path) -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, frame in enumerate(frames):
        path = out_dir / f"frame_{idx:02d}.jpg"
        rgb = np.asarray(frame, dtype=np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def load_qwen3_vl(
    model_name_or_path: str = "checkpoints/Qwen3-VL-4B-Instruct",
    device_map: str = "auto",
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    return model, processor


def first_last_frames(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        raise RuntimeError("video has no frames")
    if len(frames) == 1:
        return frames
    return np.stack([frames[0], frames[-1]], axis=0)


def endpoint_integrity_prompt(row: dict[str, Any]) -> str:
    return """
You are judging endpoint object consistency for a robot world-model video.

You will see exactly two images:
- First frame: the start of the video.
- Last frame: the end of the video.

Judge only non-robot scene objects. Ignore the robot arm/gripper as scene objects; it may move, enter the view, leave the view, hold objects, or cover objects. Do not judge task success, action correctness, or motion quality.

Task and PASS/FAIL rule:
In the first frame, identify each visible non-robot object. Note its object kind, shape, color, and distinctive visible parts.

Pass condition:
In the last frame, each first-frame non-robot object can still be matched by object kind, shape, and color. It may be repositioned, held or partly hidden or covered by robot arm.

Fail conditions:
- In the last frame, a first-frame non-robot object no longer has the same object kind, shape, or color.
- A first-frame non-robot object disappears entirely in the last frame; before marking it as missing, first check that it is not partly visible, partly hidden, held, or covered by robot arm.
- A new or duplicate non-robot object appears in the last frame.

Otherwise return PASS.

Return only valid JSON with these keys:
{
  "endpoint_integrity_pass": <0 or 1>,
  "failure_type": "<none | missing_object | new_object | changed_object | multiple>",
  "affected_entities": [<short strings>],
  "rationale": "<one concise sentence>"
}

If PASS, use exactly:
{
  "endpoint_integrity_pass": 1,
  "failure_type": "none",
  "affected_entities": [],
  "rationale": "No confirmed non-robot object inconsistency is visible."
}
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _as_binary(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"})
    return int(bool(value))


ROBOT_ENTITY_TERMS = (
    "robot",
    "robotic",
    "arm",
    "gripper",
    "wrist",
    "finger",
    "end-effector",
    "end effector",
    "black link",
    "robot link",
    "tool",
    "suction",
    "claw",
)

NON_ROBOT_ENTITY_HINTS = (
    "non-robot",
    "cube",
    "block",
    "bottle",
    "cup",
    "box",
    "ball",
    "plate",
    "bowl",
    "drawer",
    "door",
    "handle",
    "lid",
    "cap",
    "cloth",
    "rope",
    "shoe",
    "toy",
)

FAILURE_TYPES = {"none", "missing_object", "new_object", "changed_object", "multiple"}
PASS_RATIONALE = "No confirmed non-robot object inconsistency is visible."
FAIL_RATIONALE = "A confirmed non-robot object inconsistency is visible."


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _is_robot_entity(text: Any) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in ROBOT_ENTITY_TERMS)


def _has_non_robot_hint(text: Any) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in NON_ROBOT_ENTITY_HINTS)


def _all_named_entities_are_robot(data: dict[str, Any]) -> bool:
    text_fields = _as_list(data.get("affected_entities", []))
    text_fields += _as_list(data.get("main_failure_modes", []))
    rationale = str(data.get("rationale", "")).strip()
    if rationale:
        text_fields.append(rationale)

    if not text_fields:
        return False
    combined = " ".join(text_fields)
    if _has_non_robot_hint(combined):
        return False
    return all(_is_robot_entity(item) for item in text_fields)


def _normalize_failure_type(value: Any) -> str:
    failure_type = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    return failure_type if failure_type in FAILURE_TYPES else "multiple"


def _strip_robot_text(value: Any) -> list[str]:
    return [item for item in _as_list(value) if not _is_robot_entity(item) or _has_non_robot_hint(item)]


def _clean_endpoint_text_fields(data: dict[str, Any], endpoint_pass: int) -> None:
    data["affected_entities"] = _strip_robot_text(data.get("affected_entities", []))
    rationale = str(data.get("rationale", "")).strip()
    if endpoint_pass:
        data["affected_entities"] = []
        data["rationale"] = PASS_RATIONALE
    elif not rationale or _is_robot_entity(rationale):
        data["rationale"] = FAIL_RATIONALE


def _parse_endpoint_response(text: str) -> dict[str, Any]:
    data = _extract_json(text)
    required = {"endpoint_integrity_pass", "failure_type", "affected_entities", "rationale"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"endpoint response missing required keys: {missing}")

    data["affected_entities"] = _as_list(data.get("affected_entities", []))
    endpoint_pass = _as_binary(data.get("endpoint_integrity_pass"))
    failure_type = _normalize_failure_type(data.get("failure_type"))
    if endpoint_pass:
        failure_type = "none"
    elif failure_type == "none":
        failure_type = "multiple"

    data_for_robot_check = {
        "affected_entities": data.get("affected_entities", []),
        "main_failure_modes": [],
        "rationale": data.get("rationale", ""),
    }
    if _all_named_entities_are_robot(data_for_robot_check):
        endpoint_pass = 1
        failure_type = "none"

    missing_object = int(failure_type == "missing_object")
    new_object = int(failure_type == "new_object")
    changed_object = int(failure_type == "changed_object")
    multiple = int(failure_type == "multiple")
    if endpoint_pass:
        missing_object = 0
        new_object = 0
        changed_object = 0
        multiple = 0

    data["endpoint_integrity_pass"] = endpoint_pass
    data["failure_type"] = failure_type
    data["missing_object"] = missing_object
    data["new_object"] = new_object
    data["changed_object"] = changed_object
    data["multiple"] = multiple
    _clean_endpoint_text_fields(data, endpoint_pass)
    return data


def qwen3_vl_judge_endpoint(
    frame_paths: list[Path],
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    if len(frame_paths) not in {1, 2}:
        raise ValueError(f"endpoint judge expects 1 or 2 frames, got {len(frame_paths)}")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append({"type": "text", "text": "First frame:"})
    content.append({"type": "image", "image": str(frame_paths[0])})
    if len(frame_paths) == 2:
        content.append({"type": "text", "text": "Last frame:"})
        content.append({"type": "image", "image": str(frame_paths[-1])})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device if hasattr(model, "device") else "cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = _parse_endpoint_response(response)
    parsed["raw_response"] = response
    return parsed


def evaluate_task2_row_qwen_endpoint_integrity(
    row: dict[str, Any],
    root: Path,
    model: Any,
    processor: Any,
    candidate_mode: str = "output",
    camera: str = "head_camera",
    max_frames: int | None = None,
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    candidate_path = resolve_path(root, candidate_path_for_mode(row, candidate_mode))
    frames = load_rgb(candidate_path, camera=camera, max_frames=max_frames)
    endpoints = first_last_frames(frames)
    with tempfile.TemporaryDirectory(prefix="actbench_qwen3vl_endpoint_") as tmp:
        frame_paths = save_sampled_frames(endpoints, Path(tmp))
        judge = qwen3_vl_judge_endpoint(
            frame_paths,
            endpoint_integrity_prompt(row),
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )
    return json_safe(
        {
            "row_id": row.get("row_id"),
            "receiver_task": row.get("receiver_task"),
            "donor_task": row.get("donor_task"),
            "episode": row.get("episode"),
            "candidate_mode": candidate_mode,
            "candidate_path": str(candidate_path),
            "candidate_frames": len(frames),
            "endpoint_frames": len(endpoints),
            **judge,
        }
    )


def evaluate_task2_manifest_qwen_endpoint_integrity(
    manifest: Path,
    root: Path,
    model_name_or_path: str = "checkpoints/Qwen3-VL-4B-Instruct",
    candidate_mode: str = "output",
    limit: int | None = None,
    row_ids: set[str] | None = None,
    camera: str = "head_camera",
    max_frames: int | None = None,
    max_new_tokens: int = 320,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest)]
    if row_ids is not None:
        rows = [row for row in rows if str(row.get("row_id")) in row_ids]
    if limit is not None:
        rows = rows[:limit]
    model, processor = load_qwen3_vl(model_name_or_path=model_name_or_path)

    results = []
    errors = []
    for row in rows:
        try:
            results.append(
                evaluate_task2_row_qwen_endpoint_integrity(
                    row,
                    root,
                    model,
                    processor,
                    candidate_mode=candidate_mode,
                    camera=camera,
                    max_frames=max_frames,
                    max_new_tokens=max_new_tokens,
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
            "model_name_or_path": model_name_or_path,
            "metric": "qwen3_vl_endpoint_integrity",
            "score_direction": "higher_is_better",
            "rows_requested": len(rows),
            "rows_scored": len(results),
            "errors": errors,
            "summary": {
                "endpoint_integrity_pass_rate": mean("endpoint_integrity_pass"),
                "missing_object_rate": mean("missing_object"),
                "new_object_rate": mean("new_object"),
                "changed_object_rate": mean("changed_object"),
                "multiple_failure_rate": mean("multiple"),
            },
            "rows": results,
        }
    )
