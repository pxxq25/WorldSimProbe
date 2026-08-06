"""Frozen Task 5 VLM protocol used for the reported benchmark results."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldsimprobe.common.manifest import read_jsonl


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def source_hdf5_for_row(row: dict[str, Any]) -> str:
    if row.get("source_hdf5"):
        return str(row["source_hdf5"])
    task = row["task"]
    episode = int(row["episode"])
    return f"data/{task}/cross_clean_50/data/episode{episode}.hdf5"


def branch_action_index(row: dict[str, Any]) -> int:
    primitive_result = row.get("primitive_result") or {}
    branch = primitive_result.get("branch") or {}
    value = branch.get("branch_action_index")
    if value is None:
        value = branch.get("start_action_index", 0)
    try:
        return max(0, int(value))
    except Exception:
        return 0


def load_rgb_sequence(path: Path, camera: str, max_frames: int | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        import imageio.v2 as imageio

        frames = []
        reader = imageio.get_reader(str(path))
        try:
            for i, frame in enumerate(reader):
                if max_frames is not None and i >= max_frames:
                    break
                arr = np.asarray(frame)
                if arr.ndim == 2:
                    arr = np.repeat(arr[..., None], 3, axis=2)
                frames.append(arr[..., :3].astype(np.uint8))
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"No RGB frames found in video: {path}")
        return np.stack(frames, axis=0)

    if suffix in {".hdf5", ".h5"}:
        try:
            import cv2
            import h5py
        except Exception as exc:
            raise RuntimeError(
                "HDF5 video decoding requires h5py and cv2 in the active environment; "
                "use mp4 candidate videos or install those packages."
            ) from exc
        with h5py.File(path, "r") as f:
            raw = f[f"observation/{camera}/rgb"][()]
        frames = []
        for i, item in enumerate(raw):
            if max_frames is not None and i >= max_frames:
                break
            data = item if isinstance(item, bytes) else bytes(item)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to decode RGB frame from {path}")
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            raise RuntimeError(f"No RGB frames found in HDF5: {path}")
        return np.stack(frames, axis=0)

    raise ValueError(f"Unsupported video path type: {path}")


ALLOWED_RESPONSE_KEYS = {
    "agent_motion_match",
    "agent_motion_reason",
    "object_motion_match",
    "object_motion_reason",
    "predicted_primitive",
    "primitive_confidence",
    "interaction_visibility_score",
    "visual_integrity_score",
    "physical_plausibility_score",
    "artifact_flags",
    "reason",
}


def load_task5_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "primitives" not in data or "primitive_descriptions" not in data:
        raise ValueError(f"Task 5 config is missing primitives/descriptions: {path}")
    return data


def primitive_definition_lines(config: dict[str, Any]) -> list[str]:
    motion_definitions = {
        "push": (
            "The active arm closes or keeps the gripper closed before contact, moves/descends to the side of the object without grasping it, then uses the closed gripper body/fingers as a pusher and moves horizontally into the object.",
            "The object translates on the table surface after side contact. It should not be lifted, carried, released, or tipped over; brief gripper closure is allowed as the pusher shape, not as a grasp.",
        ),
        "rotate": (
            "The active arm may close the gripper to grasp or hold the object and rotate it through wrist/end-effector yaw. It may also make tangential side contact around the object center; if yaw is not clearly visible, tangential side contact can still count.",
            "The object rotates in place or mostly around its center, with limited translation compared to push or slide_drag.",
        ),
        "slide_drag": (
            "The gripper grips the object and stays closed while maintaining contact, then moves mostly horizontally.",
            "The object translates along the table surface, similar to push, but the motion is caused by a closed-gripper grasp/drag rather than open-gripper side contact.",
        ),
        "pull": (
            "The robot grasps or contacts an articulated part, then moves backward/outward along the articulation direction.",
            "The articulated part opens, pulls out, or changes state along its hinge/slider/switch direction.",
        ),
        "tap": (
            "The gripper closes, briefly presses along the target-specific direction, settles, then retracts.",
            "The contacted object/control should remain mostly static or show little/no visible displacement; a subtle state change can still count.",
        ),
        "shake": (
            "The gripper closes on the object and moves horizontally back and forth for about 3 cycles, without extra vertical lift.",
            "The grasped object oscillates with the gripper and remains held; it should not be dropped, carried away, or simply pushed once.",
        ),
        "drop": (
            "The gripper closes/squeezes, lifts the object, then opens/releases it.",
            "The object falls and settles under gravity after release.",
        ),
        "knock_over": (
            "The robot makes high side contact, often after opening/releasing and then closing the gripper as a pusher, and pushes laterally.",
            "The object tips, falls, or changes from upright/stable to knocked-over orientation.",
        ),
    }
    lines = []
    for primitive in config["primitives"]:
        agent_motion, object_motion = motion_definitions[str(primitive)]
        lines.append(f"- {primitive}:\n  Agent motion: {agent_motion}\n  Object motion: {object_motion}")
    return lines


def task5_vqa_prompt(config: dict[str, Any]) -> str:
    primitives = [str(p) for p in config["primitives"]]
    labels = ", ".join(primitives)
    definitions = "\n".join(primitive_definition_lines(config))
    return f"""
You are evaluating a WorldSimProbe Task 5 robot manipulation video.

Do not assume the action label. Infer the primitive only from the video.

Allowed primitive labels, forced choice:
{labels}

Generation-grounded primitive definitions:
{definitions}

For each primitive, first judge whether the agent/robot motion matches the expected primitive. Then judge whether the object/environment motion matches the expected response. Both motion checks must pass for the primitive choice to count.

Evaluate these things independently:

1. Agent motion:
Judge whether the robot/agent motion matches the selected primitive. Describe the agent motion briefly.

2. Object/environment motion:
Judge whether the object/environment response matches the selected primitive. Describe the object motion briefly.

3. Primitive recognition:
Choose the single allowed primitive label that best matches the combined agent motion and object/environment motion.
This is a forced-choice task: always choose one of the eight allowed primitive labels.
Do not output invalid. If the interaction is subtle, brief, partially occluded, or ambiguous, still choose the closest primitive and lower primitive_confidence.
If no clear object contact is visible, infer the closest primitive from robot motion, gripper state, object motion, and final state.
Use the generation-grounded definitions above. For example, distinguish direct planar push from tangential rotate, closed-gripper horizontal slide_drag, articulated pull, short tap/press, horizontal shake, lift-and-release drop, and high-contact lateral knock_over.
Important: the forced-choice primitive label only counts as correct if agent_motion_match and object_motion_match are both true.

4. Interaction visibility:
Judge whether the clip visibly contains a robot-object interaction or object motion relevant to the chosen primitive.
This is separate from primitive recognition: still choose a primitive even when visibility is weak.
Score 5 when the interaction and object response are clear, 3 when subtle/brief/partially occluded, and 1 when the clip is mostly static or no relevant object interaction is visible.

5. Visual and physical integrity:
Judge whether the robot, gripper, and manipulated objects remain visually consistent and physically plausible over time.
Check for object disappearance, new object appearance, object deformation or melting, impossible object motion, robot/gripper deformation, impossible penetration, and severe temporal inconsistency.
Do not penalize normal occlusion, motion blur, grasping, release, falling, or object motion that follows visible robot contact.
If the primitive evidence is unclear but the video itself is visually coherent, keep integrity scores high and reflect uncertainty through primitive_confidence, interaction_visibility_score, and reason.

Scoring rules:
- predicted_primitive must be exactly one of: {labels}.
- agent_motion_match must be true if the visible robot/agent motion matches the selected primitive, otherwise false.
- object_motion_match must be true if the visible object/environment response matches the selected primitive, otherwise false.
- primitive_confidence must be a number from 0.0 to 1.0.
- interaction_visibility_score must be an integer from 1 to 5. 5 = clear robot-object interaction; 1 = no relevant interaction visible.
- visual_integrity_score must be an integer from 1 to 5. 5 = no visible artifact; 1 = severe visual corruption.
- physical_plausibility_score must be an integer from 1 to 5. 5 = physically plausible interaction; 1 = physically impossible interaction.
- Keep all reason fields short. Do not list every artifact that did not happen.
- If no artifact is visible, use artifact_flags: [].

Return only valid JSON with exactly these keys:
{{
  "agent_motion_match": true,
  "agent_motion_reason": "short reason",
  "object_motion_match": true,
  "object_motion_reason": "short reason",
  "predicted_primitive": "push|rotate|slide_drag|pull|tap|shake|drop|knock_over",
  "primitive_confidence": 0.0,
  "interaction_visibility_score": 1,
  "visual_integrity_score": 1,
  "physical_plausibility_score": 1,
  "artifact_flags": ["short strings, or empty list"],
  "reason": "short final reason"
}}
""".strip()


def candidate_path_for_mode(row: dict[str, Any], mode: str) -> str:
    if mode == "candidate":
        return (
            row.get("candidate_video")
            or row.get("output_video")
            or row.get("primitive_video")
            or row["primitive_hdf5"]
        )
    if mode == "primitive":
        return row.get("primitive_video") or row["primitive_hdf5"]
    if mode == "source":
        return source_hdf5_for_row(row)
    if mode in row:
        return str(row[mode])
    raise ValueError(f"unknown candidate mode or row field: {mode}")


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def clamp_float(value: Any, lo: float, hi: float, default: float = 0.0) -> tuple[float, bool]:
    try:
        val = float(value)
    except Exception:
        return default, False
    valid = lo <= val <= hi
    return float(min(hi, max(lo, val))), valid


def clamp_score(value: Any) -> tuple[int, bool]:
    try:
        if isinstance(value, bool):
            return 1, False
        val = int(round(float(value)))
    except Exception:
        return 1, False
    valid = 1 <= val <= 5
    return int(min(5, max(1, val))), valid


def bool_value(value: Any) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return value, True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True, True
        if lowered in {"false", "no", "0"}:
            return False, True
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value), True
    return False, False


def _partial_response_fields(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in ["agent_motion_reason", "object_motion_reason", "predicted_primitive", "reason"]:
        match = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', text, flags=re.DOTALL)
        if match:
            data[key] = match.group(1).replace('\\"', '"')
    for key in ["agent_motion_match", "object_motion_match"]:
        match = re.search(
            rf'"{key}"\s*:\s*(true|false|0|1|"true"|"false"|"yes"|"no")', text, flags=re.IGNORECASE
        )
        if match:
            raw = match.group(1).strip().strip('"').lower()
            data[key] = raw in {"true", "yes", "1"}
    for key in [
        "primitive_confidence",
        "interaction_visibility_score",
        "visual_integrity_score",
        "physical_plausibility_score",
    ]:
        match = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            raw = match.group(1)
            data[key] = float(raw) if "." in raw else int(raw)
    flags_match = re.search(r'"artifact_flags"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if flags_match:
        flags_raw = flags_match.group(1).strip()
        if not flags_raw:
            data["artifact_flags"] = []
        else:
            data["artifact_flags"] = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', flags_raw)
    return data


def parse_vqa_response(text: str, primitives: list[str]) -> dict[str, Any]:
    parse_mode = "json"
    parse_error = None
    try:
        data = extract_json(text)
    except Exception as exc:
        parse_mode = "partial_json"
        parse_error = f"{type(exc).__name__}: {exc}"
        data = _partial_response_fields(text)
        if not data:
            raise
    allowed = set(primitives) | {"invalid"}
    predicted = str(data.get("predicted_primitive", "invalid")).strip()
    if predicted not in allowed:
        predicted = "invalid"

    agent_match, agent_valid = bool_value(data.get("agent_motion_match"))
    object_match, object_valid = bool_value(data.get("object_motion_match"))
    confidence, confidence_valid = clamp_float(data.get("primitive_confidence"), 0.0, 1.0)
    visibility, visibility_valid = clamp_score(data.get("interaction_visibility_score"))
    visual, visual_valid = clamp_score(data.get("visual_integrity_score"))
    physics, physics_valid = clamp_score(data.get("physical_plausibility_score"))
    flags = data.get("artifact_flags", [])
    if isinstance(flags, str):
        flags = [flags]
    if not isinstance(flags, list):
        flags = []
    flags = [str(item) for item in flags]
    agent_reason = str(data.get("agent_motion_reason", "")).strip()
    object_reason = str(data.get("object_motion_reason", "")).strip()
    reason = str(data.get("reason", "")).strip()
    if len(agent_reason) > 180:
        agent_reason = agent_reason[:177].rstrip() + "..."
    if len(object_reason) > 180:
        object_reason = object_reason[:177].rstrip() + "..."
    if len(reason) > 180:
        reason = reason[:177].rstrip() + "..."

    extra_keys = sorted(set(data) - ALLOWED_RESPONSE_KEYS)
    missing_keys = sorted(ALLOWED_RESPONSE_KEYS - set(data))
    format_valid = (
        parse_mode == "json"
        and not extra_keys
        and not missing_keys
        and agent_valid
        and object_valid
        and confidence_valid
        and visibility_valid
        and visual_valid
        and physics_valid
    )
    return {
        "agent_motion_match": bool(agent_match),
        "agent_motion_reason": agent_reason,
        "object_motion_match": bool(object_match),
        "object_motion_reason": object_reason,
        "predicted_primitive": predicted,
        "primitive_confidence": confidence,
        "interaction_visibility_score": visibility,
        "visual_integrity_score": visual,
        "physical_plausibility_score": physics,
        "artifact_flags": flags,
        "reason": reason,
        "response_format_valid": int(format_valid),
        "response_parse_mode": parse_mode,
        "response_parse_error": parse_error,
        "response_extra_keys": extra_keys,
        "response_missing_keys": missing_keys,
        "raw_response": text,
    }


def select_rows(
    manifest: Path,
    root: Path,
    candidate_mode: str,
    limit: int | None,
    row_ids: set[str] | None,
    one_per_primitive: bool,
    seed: int,
    require_ok: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(read_jsonl(manifest)):
        item = dict(row)
        item.setdefault("manifest_index", index)
        rows.append(item)
    if require_ok:
        invalid = [str(row.get("row_id")) for row in rows if row.get("status") not in {None, "ok"}]
        if invalid:
            raise ValueError(f"Task 5 manifest contains non-evaluable rows: {invalid[:10]}")
    if row_ids is not None:
        rows = [
            row
            for row in rows
            if str(row.get("row_id")) in row_ids or str(row.get("manifest_index")) in row_ids
        ]
    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows)) if rows else []
        rows = [rows[int(i)] for i in order]

    if one_per_primitive:
        by_primitive: dict[str, dict[str, Any]] = {}
        for row in rows:
            primitive = str(row.get("primitive"))
            if primitive and primitive not in by_primitive:
                by_primitive[primitive] = row
        rows = list(by_primitive.values())
    if limit is not None:
        rows = rows[:limit]
    return rows


def sample_frames(frames: np.ndarray, sample_count: int) -> np.ndarray:
    if len(frames) <= sample_count:
        return frames
    indices = np.round(np.linspace(0, len(frames) - 1, sample_count)).astype(int)
    return frames[indices]


def save_sampled_frames(frames: np.ndarray, out_dir: Path) -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, frame in enumerate(frames):
        path = out_dir / f"frame_{idx:02d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def candidate_suffix_start(row: dict[str, Any], frame_count: int, reference_down_sample: int) -> int:
    branch_index = branch_action_index(row)
    canonical_hz = row.get("canonical_action_hz") or row.get("canonical_video_fps")
    candidate_timestamps = row.get("candidate_frame_timestamps_sec")
    if canonical_hz and candidate_timestamps:
        branch_time = float(branch_index) / float(canonical_hz)
        timestamps = np.asarray(candidate_timestamps, dtype=np.float64)
        if len(timestamps) == frame_count and len(timestamps) > 0:
            start = int(np.searchsorted(timestamps, branch_time, side="left"))
            return min(max(0, start), max(0, frame_count - 2))
    if canonical_hz and row.get("candidate_fps"):
        branch_time = float(branch_index) / float(canonical_hz)
        start = int(round(branch_time * float(row["candidate_fps"])))
        return min(max(0, start), max(0, frame_count - 2))
    start = max(0, branch_index // max(1, reference_down_sample))
    return min(start, max(0, frame_count - 2))


def qwen_task5_vqa_prompt(config: dict[str, Any]) -> str:
    base = task5_vqa_prompt(config)
    return base.replace(
        "You are evaluating a WorldSimProbe Task 5 robot manipulation video.",
        (
            "You are evaluating sampled frames from a WorldSimProbe Task 5 robot manipulation video. "
            "The frames are shown in chronological order."
        ),
    )


def load_qwen3_vl(
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    dtype_name: str = "bfloat16",
    device_map: str = "cuda",
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    return model, processor


def qwen3_vl_judge_frames(
    frame_paths: list[Path],
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int = 384,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": (
                "The following images are sampled frames from one video in chronological order. "
                "Use temporal changes across frames to classify the primitive and judge integrity."
            ),
        },
    ]
    for idx, path in enumerate(frame_paths):
        label = "start" if idx == 0 else "end" if idx == len(frame_paths) - 1 else f"frame {idx:02d}"
        content.append({"type": "text", "text": f"Frame {idx:02d} ({label}):"})
        content.append({"type": "image", "image": str(path)})

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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def evaluate_task5_row_qwen3_vl_vqa(
    row: dict[str, Any],
    root: Path,
    prompt: str,
    primitives: list[str],
    model: Any,
    processor: Any,
    candidate_mode: str = "primitive",
    camera: str = "head_camera",
    max_frames: int | None = None,
    suffix_only: bool = False,
    reference_down_sample: int = 3,
    sample_count: int = 12,
    max_new_tokens: int = 384,
    integrity_threshold: int = 4,
) -> dict[str, Any]:
    candidate_path = resolve_path(root, candidate_path_for_mode(row, candidate_mode))
    frames = load_rgb_sequence(candidate_path, camera=camera, max_frames=max_frames)
    start = 0
    if suffix_only:
        start = candidate_suffix_start(row, len(frames), reference_down_sample)
        frames = frames[start:]
    sampled = sample_frames(frames, sample_count)

    with tempfile.TemporaryDirectory(prefix="worldsimprobe_task5_qwen_vqa_") as tmp:
        frame_paths = save_sampled_frames(sampled, Path(tmp))
        response = qwen3_vl_judge_frames(
            frame_paths,
            prompt,
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )

    parse_error = None
    try:
        parsed = parse_vqa_response(response, primitives)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
        parsed = {
            "agent_motion_match": False,
            "agent_motion_reason": "unparseable response",
            "object_motion_match": False,
            "object_motion_reason": "unparseable response",
            "predicted_primitive": "invalid",
            "primitive_confidence": 0.0,
            "interaction_visibility_score": 1,
            "visual_integrity_score": 1,
            "physical_plausibility_score": 1,
            "artifact_flags": ["unparseable_vlm_response"],
            "reason": "The VLM response was not valid JSON.",
            "response_format_valid": 0,
            "response_parse_mode": "failed",
            "response_parse_error": parse_error,
            "response_extra_keys": [],
            "response_missing_keys": sorted(ALLOWED_RESPONSE_KEYS),
            "raw_response": response,
        }
    intended = str(row.get("primitive", ""))
    forced_choice_primitive_match = int(parsed["predicted_primitive"] == intended)
    agent_motion_match = int(bool(parsed.get("agent_motion_match", False)))
    object_motion_match = int(bool(parsed.get("object_motion_match", False)))
    motion_gate_match = int(agent_motion_match and object_motion_match)
    primitive_match = int(forced_choice_primitive_match and motion_gate_match)
    interaction_visible = int(parsed.get("interaction_visibility_score", 1) >= 3)
    integrity_ok = int(
        parsed["visual_integrity_score"] >= integrity_threshold
        and parsed["physical_plausibility_score"] >= integrity_threshold
    )
    return json_safe(
        {
            "row_id": row.get("row_id"),
            "manifest_index": row.get("manifest_index"),
            "task": row.get("task"),
            "episode": row.get("episode"),
            "source_episode": row.get("source_episode"),
            "intended_primitive": intended,
            "candidate_mode": candidate_mode,
            "candidate_path": str(candidate_path),
            "suffix_only": int(suffix_only),
            "branch_action_index": branch_action_index(row),
            "sampled_frames": len(sampled),
            "video_input_start_frame": int(start),
            **parsed,
            "parse_error": parse_error,
            "forced_choice_primitive_match": forced_choice_primitive_match,
            "agent_motion_match_int": agent_motion_match,
            "object_motion_match_int": object_motion_match,
            "motion_gate_match": motion_gate_match,
            "primitive_match": primitive_match,
            "interaction_visible": interaction_visible,
            "integrity_ok": integrity_ok,
            "computed_pass": int(primitive_match and interaction_visible and integrity_ok),
            "integrity_threshold": int(integrity_threshold),
        }
    )


def evaluate_task5_manifest_qwen3_vl_vqa(
    manifest: Path,
    root: Path,
    task_config: Path,
    model_name_or_path: str,
    candidate_mode: str = "primitive",
    limit: int | None = None,
    row_ids: set[str] | None = None,
    one_per_primitive: bool = False,
    seed: int = 20260608,
    require_ok: bool = True,
    camera: str = "head_camera",
    max_frames: int | None = None,
    suffix_only: bool = False,
    reference_down_sample: int = 3,
    sample_count: int = 12,
    max_new_tokens: int = 384,
    integrity_threshold: int = 4,
    dtype_name: str = "bfloat16",
    device_map: str = "cuda",
    local_files_only: bool = True,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    config = load_task5_config(task_config)
    primitives = [str(p) for p in config["primitives"]]
    prompt = prompt_override if prompt_override is not None else qwen_task5_vqa_prompt(config)
    rows = select_rows(
        manifest,
        root,
        candidate_mode=candidate_mode,
        limit=limit,
        row_ids=row_ids,
        one_per_primitive=one_per_primitive,
        seed=seed,
        require_ok=require_ok,
    )
    model, processor = load_qwen3_vl(
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
        device_map=device_map,
        local_files_only=local_files_only,
    )

    results = []
    errors = []
    for row in rows:
        try:
            results.append(
                evaluate_task5_row_qwen3_vl_vqa(
                    row,
                    root=root,
                    prompt=prompt,
                    primitives=primitives,
                    model=model,
                    processor=processor,
                    candidate_mode=candidate_mode,
                    camera=camera,
                    max_frames=max_frames,
                    suffix_only=suffix_only,
                    reference_down_sample=reference_down_sample,
                    sample_count=sample_count,
                    max_new_tokens=max_new_tokens,
                    integrity_threshold=integrity_threshold,
                )
            )
        except Exception as exc:
            error = {
                "row_id": row.get("row_id"),
                "sample_id": row.get("sample_id") or row.get("row_id"),
                "manifest_index": row.get("manifest_index"),
                "task": row.get("task"),
                "episode": row.get("episode"),
                "primitive": row.get("primitive"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            results.append(
                {
                    **error,
                    "intended_primitive": str(row.get("primitive", "")),
                    "predicted_primitive": "invalid",
                    "evaluation_failed": 1,
                    "forced_choice_primitive_match": 0,
                    "agent_motion_match_int": 0,
                    "object_motion_match_int": 0,
                    "motion_gate_match": 0,
                    "primitive_match": 0,
                    "interaction_visible": 0,
                    "integrity_ok": 0,
                    "computed_pass": 0,
                    "response_format_valid": 0,
                }
            )

    n = len(results)
    primitive_matches = sum(float(row.get("primitive_match", 0)) for row in results)
    forced_choice_primitive_matches = sum(
        float(row.get("forced_choice_primitive_match", 0)) for row in results
    )
    agent_motion_matches = sum(
        float(row.get("agent_motion_match_int", 0)) for row in results
    )
    object_motion_matches = sum(
        float(row.get("object_motion_match_int", 0)) for row in results
    )
    motion_gate_matches = sum(float(row.get("motion_gate_match", 0)) for row in results)
    integrity_ok = sum(float(row.get("integrity_ok", 0)) for row in results)
    interaction_visible = sum(float(row.get("interaction_visible", 0)) for row in results)
    computed_passes = sum(float(row.get("computed_pass", 0)) for row in results)
    format_valid = sum(float(row.get("response_format_valid", 0)) for row in results)
    confusion: dict[str, dict[str, int]] = {}
    by_primitive: dict[str, dict[str, Any]] = {}
    for row in results:
        intended = str(row.get("intended_primitive"))
        predicted = str(row.get("predicted_primitive"))
        confusion.setdefault(intended, {})[predicted] = (
            confusion.setdefault(intended, {}).get(predicted, 0) + 1
        )
    for row in results:
        intended = str(row.get("intended_primitive"))
        stats = by_primitive.setdefault(
            intended,
            {
                "n": 0,
                "primitive_matches": 0,
                "forced_choice_primitive_matches": 0,
                "agent_motion_matches": 0,
                "object_motion_matches": 0,
                "motion_gate_matches": 0,
                "interaction_visible": 0,
                "integrity_ok": 0,
                "computed_passes": 0,
            },
        )
        stats["n"] += 1
        stats["primitive_matches"] += float(row.get("primitive_match", 0))
        stats["forced_choice_primitive_matches"] += float(
            row.get("forced_choice_primitive_match", 0)
        )
        stats["agent_motion_matches"] += float(row.get("agent_motion_match_int", 0))
        stats["object_motion_matches"] += float(row.get("object_motion_match_int", 0))
        stats["motion_gate_matches"] += float(row.get("motion_gate_match", 0))
        stats["interaction_visible"] += float(row.get("interaction_visible", 0))
        stats["integrity_ok"] += float(row.get("integrity_ok", 0))
        stats["computed_passes"] += float(row.get("computed_pass", 0))

    primitive_rates = [
        stats["primitive_matches"] / stats["n"]
        for stats in by_primitive.values()
        if stats["n"]
    ]
    for stats in by_primitive.values():
        count = stats["n"]
        stats["primitive_accuracy"] = stats["primitive_matches"] / count if count else 0.0
    macro_primitive_accuracy = float(np.mean(primitive_rates)) if primitive_rates else 0.0

    return json_safe(
        {
            "manifest": str(manifest),
            "root": str(root),
            "task_config": str(task_config),
            "model": model_name_or_path,
            "candidate_mode": candidate_mode,
            "suffix_only": int(suffix_only),
            "sample_count": int(sample_count),
            "integrity_threshold": int(integrity_threshold),
            "rows_requested": n,
            "rows_scored": n,
            "errors": errors,
            "summary": {
                "primitive_accuracy": macro_primitive_accuracy,
                "primitive_accuracy_0_to_100": 100.0 * macro_primitive_accuracy,
                "primitive_accuracy_micro": primitive_matches / n if n else 0.0,
                "forced_choice_primitive_accuracy": forced_choice_primitive_matches / n if n else 0.0,
                "agent_motion_match_rate": agent_motion_matches / n if n else 0.0,
                "object_motion_match_rate": object_motion_matches / n if n else 0.0,
                "motion_gate_match_rate": motion_gate_matches / n if n else 0.0,
                "interaction_visible_rate": interaction_visible / n if n else 0.0,
                "integrity_ok_rate": integrity_ok / n if n else 0.0,
                "computed_pass_rate": computed_passes / n if n else 0.0,
                "response_format_valid_rate": format_valid / n if n else 0.0,
                "primitive_matches": primitive_matches,
                "forced_choice_primitive_matches": forced_choice_primitive_matches,
                "agent_motion_matches": agent_motion_matches,
                "object_motion_matches": object_motion_matches,
                "motion_gate_matches": motion_gate_matches,
                "interaction_visible": interaction_visible,
                "integrity_ok": integrity_ok,
                "computed_passes": computed_passes,
                "response_format_valid": format_valid,
                "by_primitive": by_primitive,
                "confusion": confusion,
            },
            "prompt": prompt,
            "rows": results,
        }
    )
