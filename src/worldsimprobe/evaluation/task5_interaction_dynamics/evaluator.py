"""Task 5 evaluator with shared-physical-timestamp native-video sampling.

The frozen protocol module owns the prompt, response parser, gates,
aggregation, and Qwen loader. This module supplies a native-video clip
sampled at shared physical timestamps between candidate and reference horizons.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from worldsimprobe.evaluation.task5_interaction_dynamics import frozen_protocol as BASE


def _read_video(path: Path, max_frames: int | None) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 0.0)
        frames = []
        for index, frame in enumerate(reader):
            if max_frames is not None and index >= max_frames:
                break
            frames.append(np.asarray(frame)[..., :3].astype(np.uint8))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid fps={fps} for {path}")
    return np.stack(frames), fps


def _source_timestamps(row: dict[str, Any], frame_count: int, decoded_fps: float) -> np.ndarray:
    values = row.get("candidate_frame_timestamps_sec")
    if isinstance(values, list) and len(values) == frame_count:
        timestamps = np.asarray(values, dtype=np.float64)
        if (
            np.all(np.isfinite(timestamps))
            and np.all(np.diff(timestamps) >= 0)
            and timestamps[-1] > timestamps[0]
        ):
            return timestamps - timestamps[0]
    return np.arange(frame_count, dtype=np.float64) / decoded_fps


def _expected_duration(row: dict[str, Any], fallback: float) -> float:
    containers = (row.get("prediction_metadata"), row.get("metadata"))
    for container in containers:
        if not isinstance(container, dict):
            continue
        validation = container.get("full_horizon_validation")
        if not isinstance(validation, dict):
            continue
        value = validation.get("expected_duration_sec")
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration

    canonical = row.get("canonical_eval_timestamps_sec")
    if isinstance(canonical, list) and len(canonical) >= 2:
        try:
            duration = float(canonical[-1]) - float(canonical[0])
        except (TypeError, ValueError):
            duration = 0.0
        if math.isfinite(duration) and duration > 0:
            return duration
    return fallback


def _nearest_indices(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, query, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(query - source[left]) <= np.abs(source[right] - query)
    return np.where(choose_left, left, right).astype(np.int64)


def _sample_physical_time(
    row: dict[str, Any],
    frames: np.ndarray,
    decoded_fps: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Sample frames at shared physical timestamps.

    Uses the common duration (min of candidate and expected) so candidate
    and reference share the same physical time window.
    """
    source_times = _source_timestamps(row, len(frames), decoded_fps)
    candidate_duration = float(source_times[-1]) if len(source_times) > 1 else 0.0
    expected_duration = _expected_duration(row, candidate_duration)
    common_duration = min(candidate_duration, expected_duration)
    if common_duration <= 0:
        query = np.zeros(sample_count, dtype=np.float64)
    else:
        query = np.linspace(0.0, common_duration, sample_count, dtype=np.float64)
    indices = _nearest_indices(source_times, query)
    return (
        frames[indices],
        query,
        indices,
        candidate_duration,
        expected_duration,
    )


def _write_normalized_video(
    path: Path, frames: np.ndarray, duration: float
) -> float:
    output_fps = max(1.0, (len(frames) - 1) / max(duration, 1e-6))
    writer = imageio.get_writer(
        str(path),
        fps=output_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return output_fps


def _judge_native_video(
    video_path: Path,
    prompt: str,
    model: Any,
    processor: Any,
    sample_count: int,
    max_new_tokens: int,
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "text",
                    "text": (
                        f"This is the temporally normalized {sample_count}-frame "
                        "clip. Frames are chronological and preserve the shared "
                        "physical time window."
                    ),
                },
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": int(sample_count),
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = [item[1] for item in video_inputs]
    video_tensors = [item[0] for item in video_inputs]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_tensors,
        video_metadata=video_metadata,
        return_metadata=True,
        padding=True,
        return_tensors=None,
        **video_kwargs,
    )
    inputs.pop("video_metadata", None)
    inputs = inputs.convert_to_tensors("pt")
    inputs = inputs.to(model.device if hasattr(model, "device") else "cuda")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed,
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
    candidate_mode: str = "candidate",
    camera: str = "head_camera",
    max_frames: int | None = None,
    suffix_only: bool = False,
    reference_down_sample: int = 3,
    sample_count: int = 12,
    max_new_tokens: int = 512,
    integrity_threshold: int = 4,
) -> dict[str, Any]:
    del camera, reference_down_sample
    if suffix_only:
        raise ValueError("shared physical-timestamp protocol requires suffix_only=False")

    candidate_path = BASE.resolve_path(root, BASE.candidate_path_for_mode(row, candidate_mode))
    frames, decoded_fps = _read_video(candidate_path, max_frames=max_frames)
    sampled, timestamps, indices, candidate_duration, expected_duration = _sample_physical_time(
        row, frames, decoded_fps, sample_count
    )
    common_duration = float(timestamps[-1]) if len(timestamps) else 0.0

    with tempfile.TemporaryDirectory(prefix="worldsimprobe_task5_native_video_") as tmp:
        normalized_path = Path(tmp) / "normalized.mp4"
        normalized_fps = _write_normalized_video(normalized_path, sampled, common_duration)
        response = _judge_native_video(
            normalized_path,
            prompt,
            model,
            processor,
            sample_count=sample_count,
            max_new_tokens=max_new_tokens,
        )

    parse_error = None
    try:
        parsed = BASE.parse_vqa_response(response, primitives)
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
            "response_missing_keys": sorted(BASE.ALLOWED_RESPONSE_KEYS),
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
    return BASE.json_safe(
        {
            "row_id": row.get("row_id"),
            "sample_id": row.get("sample_id") or row.get("row_id"),
            "manifest_index": row.get("manifest_index"),
            "task": row.get("task"),
            "episode": row.get("episode"),
            "source_episode": row.get("source_episode"),
            "intended_primitive": intended,
            "candidate_mode": candidate_mode,
            "candidate_path": str(candidate_path),
            "suffix_only": 0,
            "branch_action_index": BASE.branch_action_index(row),
            "sampled_frames": len(sampled),
            "video_input_start_frame": 0,
            "sampling_protocol": "shared_physical_timestamps_full_common_duration",
            "vision_input": "native_video",
            "source_frame_count": len(frames),
            "source_decoded_fps": float(decoded_fps),
            "source_indices": [int(value) for value in indices],
            "shared_timestamps_sec": [round(float(value), 6) for value in timestamps],
            "candidate_duration_sec": float(candidate_duration),
            "expected_duration_sec": float(expected_duration),
            "common_duration_sec": float(common_duration),
            "normalized_clip_fps": float(normalized_fps),
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


def evaluate_task5_manifest_qwen3_vl_vqa(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("candidate_mode", "candidate")
    kwargs.setdefault("max_new_tokens", 512)
    original = BASE.evaluate_task5_row_qwen3_vl_vqa
    BASE.evaluate_task5_row_qwen3_vl_vqa = evaluate_task5_row_qwen3_vl_vqa
    try:
        result = BASE.evaluate_task5_manifest_qwen3_vl_vqa(*args, **kwargs)
    finally:
        BASE.evaluate_task5_row_qwen3_vl_vqa = original
    result["task_id"] = "task5"
    result["sampling_protocol"] = {
        "name": "shared_physical_timestamps_full_common_duration",
        "vision_input": "native_video",
        "sample_count": int(kwargs.get("sample_count", 12)),
        "expected_duration_source": "benchmark prediction horizon",
        "protocol_module": "worldsimprobe.evaluation.task5_interaction_dynamics.frozen_protocol",
    }
    return result
