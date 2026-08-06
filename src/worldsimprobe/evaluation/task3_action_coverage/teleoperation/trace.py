from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCREEN_COMMANDS = {"up", "down", "left", "right"}
VERTICAL_COMMANDS = {"lift", "delift"}
ROTATION_COMMANDS = {"roll_neg", "roll_pos", "pitch_pos", "pitch_neg", "yaw_neg", "yaw_pos"}
GRIPPER_COMMANDS = {"open", "close"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean([1.0 if item else 0.0 for item in values]))


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def normalize_quat(quat: Any) -> np.ndarray:
    q = arr(quat).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_mat(quat: Any) -> np.ndarray:
    w, x, y, z = normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_delta_rotvec(before_quat: Any, after_quat: Any) -> np.ndarray:
    rel = quat_to_mat(after_quat) @ quat_to_mat(before_quat).T
    cos_angle = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_vec = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]], dtype=np.float64)
    if angle < 1e-8:
        return 0.5 * skew_vec
    denom = 2.0 * np.sin(angle)
    if abs(denom) < 1e-8:
        return 0.5 * skew_vec
    return (skew_vec / denom) * angle


def selected_eval(entry: dict[str, Any]) -> dict[str, Any]:
    controller_eval = entry.get("controller_eval") or {}
    return controller_eval.get("selected") or {}


def screen_direction_from_delta(command: str, uv_delta: Any) -> tuple[bool, float, float, float]:
    desired = {
        "up": np.array([0.0, -1.0], dtype=np.float64),
        "down": np.array([0.0, 1.0], dtype=np.float64),
        "left": np.array([-1.0, 0.0], dtype=np.float64),
        "right": np.array([1.0, 0.0], dtype=np.float64),
    }[command]
    delta = arr(uv_delta)
    primary = float(delta @ desired)
    off_axis = float(abs(delta @ np.array([-desired[1], desired[0]], dtype=np.float64)))
    magnitude = float(np.linalg.norm(delta))
    return bool(primary > 0.0), primary, off_axis, magnitude


def screen_uv_delta_from_entry(entry: dict[str, Any], selected: dict[str, Any]) -> Any:
    if selected.get("uv_delta") is not None:
        return selected.get("uv_delta")
    controller_eval = entry.get("controller_eval") or {}
    before = controller_eval.get("before_projection") or {}
    after = controller_eval.get("after_projection") or {}
    before_uv = before.get("uv")
    after_uv = after.get("uv")
    if before_uv is None or after_uv is None:
        return None
    return (arr(after_uv) - arr(before_uv)).tolist()


def arm_from_entry(entry: dict[str, Any]) -> str:
    return str(entry.get("arm") or entry.get("active_arm") or "left").lower()


def gripper_index_for_arm(arm: str, values: np.ndarray | None = None) -> int:
    if values is not None and values.size <= 7:
        return 6
    return 6 if arm == "left" else 13


def before_after_for_arm(entry: dict[str, Any], arm: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    before = entry.get(f"before_{arm}")
    after = entry.get(f"after_{arm}")
    if before is None or after is None:
        return None, None
    return arr(before), arr(after)


def command_order_metrics(trace: dict[str, Any], expected_commands: list[str] | None = None) -> dict[str, Any]:
    commands = [str(item).lower() for item in trace.get("commands", [])]
    observed = [str(item.get("command", "")).lower() for item in trace.get("command_log", [])]
    expected = [str(item).lower() for item in (expected_commands or commands)]
    exact = observed == expected
    prefix = observed[: len(expected)] == expected if expected else True
    return {
        "expected_commands": expected,
        "observed_commands": observed,
        "command_count": len(observed),
        "expected_command_count": len(expected),
        "command_order_exact": int(exact),
        "command_order_prefix": int(prefix),
    }


def evaluate_screen_command(entry: dict[str, Any], min_primary_px: float, max_off_axis_ratio: float) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    selected = selected_eval(entry)
    if selected.get("translation_frame") == "world_table" or entry.get("translation_frame") == "world_table":
        return {"applicable": False, "pass": False, "reason": "world_table_translation"}
    uv_delta = screen_uv_delta_from_entry(entry, selected)
    primary = selected.get("primary_px")
    off_axis = selected.get("off_axis_px")
    magnitude = selected.get("magnitude_px")
    direction_ok = selected.get("direction_ok")
    if uv_delta is not None:
        computed_direction_ok, computed_primary, computed_off_axis, computed_magnitude = screen_direction_from_delta(
            command,
            uv_delta,
        )
        if primary is None:
            primary = computed_primary
        if off_axis is None:
            off_axis = computed_off_axis
        if magnitude is None:
            magnitude = computed_magnitude
        if direction_ok is None:
            direction_ok = computed_direction_ok
    if primary is None or off_axis is None:
        return {"applicable": False, "pass": False, "reason": "missing_screen_projection"}
    primary = float(primary)
    off_axis = float(off_axis)
    direction_ok = bool(direction_ok)
    ratio = off_axis / max(abs(primary), 1e-6)
    passed = direction_ok and primary >= min_primary_px and ratio <= max_off_axis_ratio
    return {
        "applicable": True,
        "pass": bool(passed),
        "command": command,
        "direction_ok": direction_ok,
        "primary_px": primary,
        "off_axis_px": off_axis,
        "magnitude_px": float(magnitude) if magnitude is not None else None,
        "off_axis_ratio": ratio,
        "min_primary_px": min_primary_px,
        "max_off_axis_ratio": max_off_axis_ratio,
        "uv_delta": uv_delta,
    }


def evaluate_planar_command(
    entry: dict[str, Any],
    min_abs_primary_m: float,
    max_off_axis_ratio: float,
) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    selected = selected_eval(entry)
    if selected.get("translation_frame") != "world_table" and entry.get("translation_frame") != "world_table":
        return {"applicable": False, "pass": False, "reason": "not_world_table_translation"}
    axis = selected.get("world_axis") or entry.get("world_axis")
    world_delta = selected.get("world_delta") or entry.get("world_delta")
    primary = selected.get("primary_world_delta") or entry.get("primary_world_delta")
    off_axis = selected.get("off_axis_world_delta") or entry.get("off_axis_world_delta")
    direction_ok = selected.get("direction_ok")
    if axis is None:
        axis = {
            "up": [0.0, 1.0, 0.0],
            "down": [0.0, -1.0, 0.0],
            "left": [-1.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0],
        }.get(command)
    if world_delta is None:
        controller_eval = entry.get("controller_eval") or {}
        before = controller_eval.get("before_projection") or {}
        after = controller_eval.get("after_projection") or {}
        before_pose = before.get("tcp_pose")
        after_pose = after.get("tcp_pose")
        if before_pose is not None and after_pose is not None:
            world_delta = (arr(after_pose)[:3] - arr(before_pose)[:3]).tolist()
    if axis is None or world_delta is None:
        return {"applicable": False, "pass": False, "reason": "missing_world_delta"}
    axis_vec = arr(axis)
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-12:
        return {"applicable": False, "pass": False, "reason": "invalid_world_axis"}
    axis_vec = axis_vec / axis_norm
    delta_vec = arr(world_delta)
    if primary is None:
        primary = float(delta_vec @ axis_vec)
    primary = float(primary)
    if off_axis is None:
        off_axis = float(np.linalg.norm(delta_vec - axis_vec * primary))
    off_axis = float(off_axis)
    if direction_ok is None:
        direction_ok = primary > 0.0
    ratio = off_axis / max(abs(primary), 1e-6)
    passed = bool(direction_ok and primary >= min_abs_primary_m and ratio <= max_off_axis_ratio)
    return {
        "applicable": True,
        "pass": passed,
        "command": command,
        "direction_ok": bool(direction_ok),
        "primary_world_delta": primary,
        "off_axis_world_delta": off_axis,
        "off_axis_ratio": ratio,
        "min_abs_primary_m": float(min_abs_primary_m),
        "max_off_axis_ratio": float(max_off_axis_ratio),
        "world_axis": axis_vec.tolist(),
        "world_delta": delta_vec.tolist(),
    }


def evaluate_vertical_command(entry: dict[str, Any], min_abs_z_delta: float) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    selected = selected_eval(entry)
    z_delta = selected.get("z_delta")
    if z_delta is None:
        return {"applicable": False, "pass": False, "reason": "missing_z_delta"}
    z_delta = float(z_delta)
    sign_ok = z_delta > 0 if command == "lift" else z_delta < 0
    passed = sign_ok and abs(z_delta) >= min_abs_z_delta
    return {
        "applicable": True,
        "pass": bool(passed),
        "command": command,
        "z_delta": z_delta,
        "sign_ok": bool(sign_ok),
        "min_abs_z_delta": min_abs_z_delta,
    }


def evaluate_gripper_command(entry: dict[str, Any], tolerance: float) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    arm = arm_from_entry(entry)
    before, after = before_after_for_arm(entry, arm)
    if before is None or after is None:
        return {"applicable": False, "pass": False, "reason": f"missing_{arm}_state"}
    idx = gripper_index_for_arm(arm, before)
    before_value = float(before[idx])
    after_value = float(after[idx])
    delta = after_value - before_value
    if command == "open":
        at_boundary = before_value >= 1.0 - tolerance
        direction_ok = delta > tolerance or (at_boundary and abs(delta) <= tolerance)
        boundary = 1.0
    else:
        at_boundary = before_value <= tolerance
        direction_ok = delta < -tolerance or (at_boundary and abs(delta) <= tolerance)
        boundary = 0.0
    in_range = -tolerance <= after_value <= 1.0 + tolerance
    return {
        "applicable": True,
        "pass": bool(direction_ok and in_range),
        "command": command,
        "arm": arm,
        "before_gripper": before_value,
        "after_gripper": after_value,
        "delta": float(delta),
        "boundary": boundary,
        "at_boundary": bool(at_boundary),
        "direction_ok": bool(direction_ok),
        "in_range": bool(in_range),
        "tolerance": tolerance,
    }


def evaluate_orientation_command(
    entry: dict[str, Any],
    min_abs_primary_rad: float,
    max_off_axis_ratio: float,
) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    selected = selected_eval(entry)
    axis = selected.get("orientation_axis_world") or entry.get("orientation_axis_world")
    rot_delta = selected.get("orientation_delta") or entry.get("orientation_delta")
    primary = selected.get("primary_rad") or entry.get("primary_rad")
    off_axis = selected.get("off_axis_rad")
    direction_ok = selected.get("direction_ok")
    if rot_delta is None:
        controller_eval = entry.get("controller_eval") or {}
        before = controller_eval.get("before_projection") or {}
        after = controller_eval.get("after_projection") or {}
        before_pose = before.get("tcp_pose")
        after_pose = after.get("tcp_pose")
        if before_pose is not None and after_pose is not None:
            rot_delta = quat_delta_rotvec(arr(before_pose)[3:7], arr(after_pose)[3:7]).tolist()
    if axis is None or rot_delta is None:
        return {"applicable": False, "pass": False, "reason": "missing_orientation_delta"}
    axis_vec = arr(axis)
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-12:
        return {"applicable": False, "pass": False, "reason": "invalid_orientation_axis"}
    axis_vec = axis_vec / axis_norm
    rot_vec = arr(rot_delta)
    if primary is None:
        primary = float(rot_vec @ axis_vec)
    primary = float(primary)
    if off_axis is None:
        off_axis = float(np.linalg.norm(rot_vec - axis_vec * primary))
    off_axis = float(off_axis)
    if direction_ok is None:
        direction_ok = primary > 0.0
    ratio = off_axis / max(abs(primary), 1e-6)
    passed = bool(direction_ok and abs(primary) >= min_abs_primary_rad and ratio <= max_off_axis_ratio)
    return {
        "applicable": True,
        "pass": passed,
        "command": command,
        "direction_ok": bool(direction_ok),
        "primary_rad": primary,
        "primary_deg": float(np.degrees(primary)),
        "off_axis_rad": off_axis,
        "off_axis_ratio": ratio,
        "min_abs_primary_rad": float(min_abs_primary_rad),
        "max_off_axis_ratio": float(max_off_axis_ratio),
        "orientation_axis_world": axis_vec.tolist(),
        "orientation_delta": rot_vec.tolist(),
    }


def evaluate_inactive_arm_lock(trace: dict[str, Any], entry: dict[str, Any], tolerance: float) -> dict[str, Any]:
    arm = arm_from_entry(entry)
    if arm == "left" and entry.get("right_action") is not None and trace.get("right_locked_action") is not None:
        right_action = arr(entry["right_action"])
        locked = arr(trace["right_locked_action"])
        max_abs_error = float(np.max(np.abs(right_action - locked))) if right_action.size else 0.0
        return {
            "applicable": True,
            "pass": bool(max_abs_error <= tolerance),
            "active_arm": arm,
            "inactive_arm": "right",
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
        }

    inactive = "left" if arm == "right" else "right"
    before, after = before_after_for_arm(entry, inactive)
    if before is None or after is None:
        return {"applicable": False, "pass": True, "reason": "no_inactive_arm_state"}
    max_abs_error = float(np.max(np.abs(after - before))) if before.size else 0.0
    return {
        "applicable": True,
        "pass": bool(max_abs_error <= tolerance),
        "active_arm": arm,
        "inactive_arm": inactive,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
    }


def evaluate_action_effect(entry: dict[str, Any], joint_tolerance: float) -> dict[str, Any]:
    command = str(entry.get("command", "")).lower()
    arm = arm_from_entry(entry)
    before, after = before_after_for_arm(entry, arm)
    if before is None or after is None:
        return {"applicable": False, "pass": False, "reason": f"missing_{arm}_state"}
    max_abs_delta = float(np.max(np.abs(after - before))) if before.size else 0.0
    expected_change = command in SCREEN_COMMANDS or command in VERTICAL_COMMANDS or command in ROTATION_COMMANDS
    if command in GRIPPER_COMMANDS:
        before_gripper = float(before[gripper_index_for_arm(arm, before)])
        if command == "open":
            expected_change = before_gripper < 1.0 - joint_tolerance
        else:
            expected_change = before_gripper > joint_tolerance
    return {
        "applicable": True,
        "pass": bool((max_abs_delta > joint_tolerance) if expected_change else True),
        "command": command,
        "arm": arm,
        "expected_change": bool(expected_change),
        "max_abs_delta": max_abs_delta,
        "joint_tolerance": joint_tolerance,
    }


def video_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        import cv2
    except Exception:
        return {"path": str(path), "exists": True, "opencv_available": False}
    cap = cv2.VideoCapture(str(path))
    info = {
        "path": str(path),
        "exists": True,
        "opencv_available": True,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def load_sampled_frames(path: Path, samples: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError(f"video has no frames: {path}")
    indices = np.linspace(0, frame_count - 1, min(samples, frame_count)).round().astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode sampled frames: {path}")
    return np.stack(frames, axis=0)


def video_alignment_metrics(
    reference_video: Path | None,
    candidate_video: Path | None,
    samples: int,
    threshold: float,
) -> dict[str, Any] | None:
    if reference_video is None or candidate_video is None:
        return None
    import cv2

    ref = load_sampled_frames(reference_video, samples)
    cand = load_sampled_frames(candidate_video, samples)
    n = min(len(ref), len(cand))
    ref = ref[:n]
    cand = cand[:n]
    if cand.shape[1:3] != ref.shape[1:3]:
        cand = np.stack(
            [cv2.resize(frame, (ref.shape[2], ref.shape[1]), interpolation=cv2.INTER_LINEAR) for frame in cand],
            axis=0,
        )
    mae = np.mean(np.abs(cand.astype(np.float32) - ref.astype(np.float32)), axis=(1, 2, 3))
    return {
        "reference_video": str(reference_video),
        "candidate_video": str(candidate_video),
        "frames_compared": int(n),
        "mean_frame_mae": float(np.mean(mae)),
        "max_frame_mae": float(np.max(mae)),
        "threshold": threshold,
        "reference_alignment_pass": int(float(np.mean(mae)) < threshold),
    }


def evaluate_task3_trace(
    trace_path: Path,
    expected_commands: list[str] | None = None,
    reference_video: Path | None = None,
    candidate_video: Path | None = None,
    min_screen_primary_px: float = 12.0,
    max_screen_off_axis_ratio: float = 0.35,
    min_abs_planar_delta: float = 0.005,
    max_planar_off_axis_ratio: float = 1.25,
    min_abs_z_delta: float = 0.015,
    min_abs_orientation_rad: float = float(np.deg2rad(1.0)),
    max_orientation_off_axis_ratio: float = 2.0,
    gripper_tolerance: float = 1e-4,
    lock_tolerance: float = 1e-6,
    joint_change_tolerance: float = 1e-6,
    video_samples: int = 24,
    video_mae_threshold: float = 12.0,
) -> dict[str, Any]:
    trace = load_json(trace_path)
    order = command_order_metrics(trace, expected_commands=expected_commands)
    command_log = trace.get("command_log", [])

    rows = []
    screen_passes: list[bool] = []
    planar_passes: list[bool] = []
    vertical_passes: list[bool] = []
    gripper_passes: list[bool] = []
    orientation_passes: list[bool] = []
    lock_passes: list[bool] = []
    effect_passes: list[bool] = []
    errors = []

    for entry in command_log:
        command = str(entry.get("command", "")).lower()
        try:
            screen = evaluate_screen_command(entry, min_screen_primary_px, max_screen_off_axis_ratio) if command in SCREEN_COMMANDS else None
            planar = evaluate_planar_command(entry, min_abs_planar_delta, max_planar_off_axis_ratio) if command in SCREEN_COMMANDS else None
            vertical = evaluate_vertical_command(entry, min_abs_z_delta) if command in VERTICAL_COMMANDS else None
            gripper = evaluate_gripper_command(entry, gripper_tolerance) if command in GRIPPER_COMMANDS else None
            orientation = (
                evaluate_orientation_command(entry, min_abs_orientation_rad, max_orientation_off_axis_ratio)
                if command in ROTATION_COMMANDS
                else None
            )
            lock = evaluate_inactive_arm_lock(trace, entry, lock_tolerance)
            effect = evaluate_action_effect(entry, joint_change_tolerance)
            if screen and screen.get("applicable"):
                screen_passes.append(bool(screen["pass"]))
            if planar and planar.get("applicable"):
                planar_passes.append(bool(planar["pass"]))
            if vertical and vertical.get("applicable"):
                vertical_passes.append(bool(vertical["pass"]))
            if gripper and gripper.get("applicable"):
                gripper_passes.append(bool(gripper["pass"]))
            if orientation and orientation.get("applicable"):
                orientation_passes.append(bool(orientation["pass"]))
            if lock.get("applicable"):
                lock_passes.append(bool(lock["pass"]))
            if effect.get("applicable"):
                effect_passes.append(bool(effect["pass"]))
            rows.append(
                {
                    "index": entry.get("index"),
                    "command": command,
                    "arm": arm_from_entry(entry),
                    "screen": screen,
                    "planar": planar,
                    "vertical": vertical,
                    "gripper": gripper,
                    "orientation": orientation,
                    "inactive_arm_lock": lock,
                    "action_effect": effect,
                    "planner": entry.get("planner"),
                    "path_steps": entry.get("path_steps"),
                    "plan_error": entry.get("plan_error"),
                }
            )
        except Exception as exc:
            errors.append({"index": entry.get("index"), "command": command, "error": repr(exc)})

    reference_video = reference_video or (Path(trace["video"]) if trace.get("video") else None)
    video = None
    if reference_video and candidate_video:
        video = video_alignment_metrics(reference_video, candidate_video, video_samples, video_mae_threshold)

    summary = {
        "command_order_accuracy": float(order["command_order_exact"]),
        "screen_direction_accuracy": mean_bool(screen_passes),
        "planar_direction_accuracy": mean_bool(planar_passes),
        "vertical_direction_accuracy": mean_bool(vertical_passes),
        "gripper_accuracy": mean_bool(gripper_passes),
        "orientation_direction_accuracy": mean_bool(orientation_passes),
        "inactive_arm_lock_accuracy": mean_bool(lock_passes),
        "action_effect_accuracy": mean_bool(effect_passes),
        "trace_contract_pass": None,
    }
    core_values = [
        order["command_order_exact"] == 1,
        all(screen_passes) if screen_passes else True,
        all(planar_passes) if planar_passes else True,
        all(vertical_passes) if vertical_passes else True,
        all(gripper_passes) if gripper_passes else True,
        all(orientation_passes) if orientation_passes else True,
        all(lock_passes) if lock_passes else True,
        all(effect_passes) if effect_passes else True,
        not errors,
    ]
    summary["trace_contract_pass"] = int(all(core_values))

    return json_safe(
        {
            "mode": "task3_control_trace_contract",
            "trace_path": trace_path,
            "trace_type": trace.get("type"),
            "task": trace.get("task"),
            "episode_id": trace.get("episode_id"),
            "control_mode": trace.get("control_mode"),
            "video": video_info(reference_video),
            "candidate_video": video_info(candidate_video),
            "order": order,
            "thresholds": {
                "min_screen_primary_px": min_screen_primary_px,
                "max_screen_off_axis_ratio": max_screen_off_axis_ratio,
                "min_abs_planar_delta": min_abs_planar_delta,
                "max_planar_off_axis_ratio": max_planar_off_axis_ratio,
                "min_abs_z_delta": min_abs_z_delta,
                "min_abs_orientation_rad": min_abs_orientation_rad,
                "max_orientation_off_axis_ratio": max_orientation_off_axis_ratio,
                "gripper_tolerance": gripper_tolerance,
                "lock_tolerance": lock_tolerance,
                "joint_change_tolerance": joint_change_tolerance,
                "video_mae_threshold": video_mae_threshold,
            },
            "summary": summary,
            "video_alignment": video,
            "errors": errors,
            "rows": rows,
        }
    )
