from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from worldsimprobe.evaluation.task3_action_coverage.teleoperation.trace import (
    evaluate_task3_trace,
    json_safe,
)
from worldsimprobe.submission.video_config import default_video_fps


TRANSLATION_COMMANDS = ["up", "down", "left", "right", "lift", "delift"]
ROTATION_COMMANDS = ["roll_neg", "roll_pos", "pitch_pos", "pitch_neg", "yaw_neg", "yaw_pos"]
GRIPPER_COMMANDS = ["open", "close"]
DEFAULT_COMMANDS = TRANSLATION_COMMANDS + ROTATION_COMMANDS + GRIPPER_COMMANDS
SUPPORTED_COMMANDS = set(DEFAULT_COMMANDS)
COMMAND_DELTA_TYPES = {
    **{cmd: "translation" for cmd in TRANSLATION_COMMANDS},
    **{cmd: "rotation" for cmd in ROTATION_COMMANDS},
    **{cmd: "gripper" for cmd in GRIPPER_COMMANDS},
}
REFERENCE_VIDEO_FPS = 30.0
TASK_OBJECT_FIELDS = {
    "stack_blocks_three": ("block1", "block2", "block3"),
    "stack_blocks_two": ("block1", "block2"),
    "stack_bowls_two": ("bowl1", "bowl2"),
    "place_empty_cup": ("cup", "coaster"),
    "place_phone_stand": ("phone", "stand"),
    "place_mouse_pad": ("mouse", "target"),
    "move_pillbottle_pad": ("pillbottle", "pad"),
    "move_stapler_pad": ("stapler", "pad"),
    "open_laptop": ("laptop",),
    "open_microwave": ("microwave",),
    "hanging_mug": ("mug", "rack"),
    "click_bell": ("bell",),
}


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _robotwin_root_from_env() -> Path:
    raw = os.environ.get("ROBOTWIN_ROOT")
    if not raw:
        raise RuntimeError(
            "RoboTwin root is required; pass --robotwin-root or set ROBOTWIN_ROOT"
        )
    path = Path(raw).expanduser()
    if not path.exists():
        raise RuntimeError(f"RoboTwin root does not exist: {path}")
    return path.resolve()


def _annotation_root_from_env() -> Path:
    raw = os.environ.get("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT")
    if not raw:
        raise RuntimeError(
            "Task 3 annotation root is required; pass --annotation-root or set "
            "WORLDSIMPROBE_TASK3_ANNOTATION_ROOT"
        )
    return Path(raw).expanduser().resolve()


@contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _insert_once(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


def _load_replay_module(robotwin_root: Path) -> ModuleType:
    helper_dir = _module_dir()
    replay_path = helper_dir / "robotwin_replay.py"
    if not replay_path.exists():
        raise RuntimeError(f"Task3 replay helper not found: {replay_path}")
    _insert_once(helper_dir)
    _insert_once(robotwin_root)
    _insert_once(robotwin_root / "script")
    module_name = "_worldsimprobe_task3_replay_live"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, replay_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Task3 replay helper: {replay_path}")
    module = importlib.util.module_from_spec(spec)
    with _cwd(robotwin_root):
        spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _round_float(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _round_list(value: Any, digits: int = 4) -> list[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), digits) for item in arr.tolist()]


def _pose_to_dict(pose: Any) -> dict[str, Any]:
    return {
        "xyz": _round_list(getattr(pose, "p", [])[:3]),
        "quat": _round_list(getattr(pose, "q", [])[:4]),
    }


def _safe_call(obj: Any, name: str) -> Any:
    method = getattr(obj, name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


@dataclass
class Task3SessionConfig:
    robotwin_root: Path = field(default_factory=_robotwin_root_from_env)
    annotation_root: Path = field(default_factory=_annotation_root_from_env)
    output_dir: Path = field(
        default_factory=lambda: (Path.cwd() / "outputs" / "task3_operator").resolve()
    )
    episode_id: str = "stack_blocks_three__000000"
    frame_id: int = 0
    down_sample: int = 3
    task_config: str = "cross_clean_50"
    planar_step: float = 0.06
    vertical_step: float = 0.02
    lateral_step_mult: float = 1.0
    rotate_deg: float = 8.0
    fps: float = field(default_factory=default_video_fps)
    record_every_path_step: int = 6
    hold_frames: int = 2
    screen_step_px: float = 24.0
    gripper_step: float = 0.2
    gripper_interpolation_steps: int = 12
    runtime_execution_mode: str = "physical_drive_target"
    show_overlay: bool = False
    max_joint_delta: float = 0.72
    fd_joint_eps: float = 0.01
    control_mode: str = "calibrated"
    fixed_delta_source: Path | None = None
    prefix_replay_steps_per_action: int = 15
    record_artifacts: bool = True


class Task3TeleopSession:
    def __init__(self, config: Task3SessionConfig | None = None):
        self.config = config or Task3SessionConfig()
        self.robotwin_root = Path(self.config.robotwin_root).expanduser().resolve()
        self.annotation_root = Path(self.config.annotation_root).expanduser().resolve()
        self.output_dir = Path(self.config.output_dir).expanduser().resolve()
        if self.config.record_artifacts:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / "teleoperation.mp4"
        self.raw_video_path = self.output_dir / "teleoperation_raw.mp4"
        self.first_frame_path = self.output_dir / "first_frame.png"
        self.manifest_path = self.output_dir / "teleoperation_trace.json"
        self.dataset_row_path = self.output_dir / "dataset_row.json"
        self.eval_path = self.output_dir / "task3_trace_validation.json"
        self._replay = _load_replay_module(self.robotwin_root)
        self._adapter = None
        self._frame_callback: Callable[[np.ndarray], None] | None = None
        self._started = 0.0
        self._closed = False
        self._writer_closed = False
        self.ann: dict[str, Any] | None = None
        self.initial_action: np.ndarray | None = None
        self.raw_action_id: int | None = None
        self.reset_state: dict[str, Any] | None = None
        self.initial_left: np.ndarray | None = None
        self.fixed_right: np.ndarray | None = None
        self.fixed_deltas: dict[str, np.ndarray] = {}
        self.command_log: list[dict[str, Any]] = []

    def set_frame_callback(self, callback: Callable[[np.ndarray], None] | None) -> None:
        self._frame_callback = callback
        if self._adapter is not None:
            self._adapter.set_frame_callback(callback)

    def emit_frame(self) -> None:
        if self._adapter is None:
            raise RuntimeError("Task 3 session is not ready")
        self._adapter.record_frame("start", "hold", -1)

    def reference_video_path(self) -> Path | None:
        ann = self.ann or {}
        task = ann.get("task")
        episode = ann.get("source_episode_index")
        if not task or episode is None:
            parts = self.config.episode_id.rsplit("__", 1)
            if len(parts) != 2:
                return None
            task, episode = parts[0], int(parts[1])
        return (
            self.robotwin_root
            / "data"
            / str(task)
            / str(self.config.task_config)
            / "video"
            / f"episode{int(episode)}.mp4"
        )

    def reference_metadata(self) -> dict[str, Any]:
        path = self.reference_video_path()
        raw_frame = int(self.raw_action_id if self.raw_action_id is not None else -1)
        return {
            "reference_video": str(path) if path is not None else None,
            "reference_fps": float(REFERENCE_VIDEO_FPS),
            "reference_start_frame": raw_frame,
            "reference_start_time_sec": round(raw_frame / REFERENCE_VIDEO_FPS, 4) if raw_frame >= 0 else None,
        }

    def _arm_telemetry(self, arm: str) -> dict[str, Any] | None:
        adapter = self._adapter
        if adapter is None:
            return None
        action = np.asarray(adapter.current_action, dtype=np.float64)
        grip_idx = adapter.left_dim if arm == "left" else adapter.left_dim + 1 + adapter.right_dim
        try:
            projection = adapter.project_tcp(arm)
        except Exception:
            projection = {}
        try:
            qpos = adapter.arm_qpos(arm)
        except Exception:
            qpos = []
        try:
            gripper_state = adapter.gripper_runtime_state(arm)
        except Exception:
            gripper_state = None
        tcp_pose = projection.get("tcp_pose") or []
        return {
            "tcp_xyz": _round_list(tcp_pose[:3]),
            "tcp_quat": _round_list(tcp_pose[3:7]),
            "screen_uv": _round_list(projection.get("uv") or [], digits=2),
            "depth": _round_float(projection.get("depth"), digits=4),
            "qpos": _round_list(qpos),
            "gripper": _round_float(action[grip_idx], digits=4) if action.size > grip_idx else None,
            "gripper_runtime": gripper_state,
        }

    def _object_telemetry(self) -> list[dict[str, Any]]:
        adapter = self._adapter
        task = getattr(adapter, "task", None) if adapter is not None else None
        task_name = (self.ann or {}).get("task")
        if task is None or not task_name:
            return []
        rows = []
        for role in TASK_OBJECT_FIELDS.get(str(task_name), ()):
            obj = getattr(task, role, None)
            if obj is None:
                continue
            pose = _safe_call(obj, "get_pose")
            row: dict[str, Any] = {
                "role": role,
                "name": str(_safe_call(obj, "get_name") or getattr(obj, "name", role)),
            }
            if pose is not None:
                row.update(_pose_to_dict(pose))
            qpos = _safe_call(obj, "get_qpos")
            if qpos is not None:
                row["qpos"] = _round_list(qpos)
            qlimits = _safe_call(obj, "get_qlimits")
            if qlimits is not None:
                try:
                    row["qlimits"] = np.asarray(qlimits, dtype=np.float64).round(4).tolist()
                except Exception:
                    pass
            rows.append(row)
        return rows

    def telemetry(self) -> dict[str, Any]:
        adapter = self._adapter
        return {
            "task": (self.ann or {}).get("task"),
            "episode_id": self.config.episode_id,
            "frame_id": int(self.config.frame_id),
            "raw_action_index": int(self.raw_action_id if self.raw_action_id is not None else -1),
            "active_arm": getattr(adapter, "active_arm", None) if adapter is not None else None,
            "active_command": getattr(adapter, "active_command", None) if adapter is not None else None,
            "robot": {
                "left": self._arm_telemetry("left"),
                "right": self._arm_telemetry("right"),
            },
            "objects": self._object_telemetry(),
        }

    def reset(self, episode_id: str | None = None, frame_id: int | None = None) -> dict[str, Any]:
        self.close()
        self._closed = False
        self._writer_closed = False
        if episode_id is not None:
            self.config.episode_id = episode_id
        if frame_id is not None:
            self.config.frame_id = int(frame_id)
        if self.config.record_artifacts:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.time()
        with _cwd(self.robotwin_root):
            ann, initial_action, raw_id, action_sequence = self._replay.load_initial_action(
                self.config.episode_id,
                self.config.frame_id,
                self.annotation_root,
                self.config.down_sample,
            )
            fixed_deltas = self._replay.load_fixed_deltas(self.config.fixed_delta_source)
            if self.config.control_mode == "fixed_delta" and not fixed_deltas:
                raise RuntimeError("--control-mode fixed_delta requires --fixed-delta-source with q_delta entries")
            adapter = self._replay.RecordingTeleopAdapter(
                self.config.task_config,
                self.config.planar_step,
                self.config.vertical_step,
                self.config.rotate_deg,
                self.config.lateral_step_mult,
                video_path=self.video_path if self.config.record_artifacts else None,
                raw_video_path=self.raw_video_path if self.config.record_artifacts else None,
                first_frame_path=self.first_frame_path if self.config.record_artifacts else None,
                fps=self.config.fps,
                record_every_path_step=self.config.record_every_path_step,
                hold_frames=self.config.hold_frames,
                screen_step_px=self.config.screen_step_px,
                gripper_step=self.config.gripper_step,
                gripper_interpolation_steps=self.config.gripper_interpolation_steps,
                max_joint_delta=self.config.max_joint_delta,
                fd_joint_eps=self.config.fd_joint_eps,
                control_mode=self.config.control_mode,
                fixed_deltas=fixed_deltas,
                fixed_delta_source=str(self.config.fixed_delta_source) if self.config.fixed_delta_source else None,
                runtime_execution_mode=self.config.runtime_execution_mode,
                show_overlay=bool(self.config.show_overlay),
            )
            adapter.set_frame_callback(self._frame_callback)
            reset_state = adapter.reset(
                ann["task"],
                int(ann["source_episode_index"]),
                initial_action,
                action_sequence=action_sequence,
                raw_action_id=raw_id,
                prefix_steps_per_action=self.config.prefix_replay_steps_per_action,
            )
            adapter.record_frame("start", "hold", -1)
            adapter.hold("start", -1)

        self.ann = ann
        self.initial_action = np.asarray(initial_action, dtype=np.float64)
        self.raw_action_id = int(raw_id)
        self._adapter = adapter
        self.reset_state = reset_state
        self.initial_left = self.initial_action[:7].copy()
        self.fixed_right = self.initial_action[7:].copy()
        self.fixed_deltas = fixed_deltas
        self.command_log = []
        if self.config.record_artifacts:
            self.write_manifest()
        return self.status()

    def _arm_slice(self, arm: str) -> slice:
        if arm == "left":
            return slice(0, 7)
        if arm == "right":
            return slice(7, 14)
        raise ValueError(f"Unsupported Task3 arm: {arm}")

    def step_command(
        self,
        command: str,
        arm: str = "left",
        input_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._adapter is None:
            self.reset()
        if self._adapter is None or self.fixed_right is None:
            raise RuntimeError("Task3 session is not initialized")
        cmd = str(command).lower()
        if cmd not in SUPPORTED_COMMANDS:
            raise ValueError(f"Unsupported Task3 command: {cmd}")
        arm = str(arm or "left").lower()
        if arm not in {"left", "right"}:
            raise ValueError(f"Unsupported Task3 arm: {arm}")
        inactive_arm = "right" if arm == "left" else "left"
        active_slice = self._arm_slice(arm)
        inactive_slice = self._arm_slice(inactive_arm)
        idx = len(self.command_log)
        with _cwd(self.robotwin_root):
            self._adapter.active_command = cmd
            self._adapter.active_mode = "action"
            self._adapter.active_step_idx = idx
            before = np.asarray(self._adapter.current_action, dtype=np.float64).copy()
            state = self._adapter.step({"command": cmd, "arm": arm})
            after = np.asarray(state["action"], dtype=np.float64)
            if not np.allclose(after[inactive_slice], before[inactive_slice], atol=1e-8):
                raise RuntimeError(
                    f"Inactive {inactive_arm} arm moved on {arm}:{cmd}: "
                    f"{after[inactive_slice].tolist()} vs {before[inactive_slice].tolist()}"
                )
            self._adapter.hold(cmd, idx)
        input_event = dict(input_event or {"source": "unknown"})
        input_event.setdefault("arm", arm)
        input_event.setdefault("command", cmd)
        telemetry_after = self.telemetry()
        selected = ((state.get("controller_eval") or {}).get("selected") or {})
        eef_delta_type = state.get("eef_delta_type") or selected.get("eef_delta_type") or COMMAND_DELTA_TYPES.get(cmd)
        entry = {
            "index": idx + 1,
            "command": cmd,
            "arm": arm,
            "active_arm": arm,
            "inactive_arm": inactive_arm,
            "control_mode": "eef_6d",
            "eef_delta_type": eef_delta_type,
            "input_event": input_event,
            "before_action": before.tolist(),
            "after_action": after.tolist(),
            "action_14d": after.tolist(),
            "q_delta": (after - before).tolist(),
            "before_left": before[:7].tolist(),
            "after_left": after[:7].tolist(),
            "before_right": before[7:].tolist(),
            "after_right": after[7:].tolist(),
            "left_action": after[:7].tolist(),
            "right_action": after[7:].tolist(),
            "left_changed": bool(not np.allclose(before[:7], after[:7], atol=1e-8)),
            "right_changed": bool(not np.allclose(before[7:], after[7:], atol=1e-8)),
            "active_changed": bool(not np.allclose(before[active_slice], after[active_slice], atol=1e-8)),
            "inactive_max_abs_error": float(
                np.max(np.abs(after[inactive_slice] - before[inactive_slice]))
            ),
            "planner": state.get("planner"),
            "execution_mode": state.get("execution_mode"),
            "plan_scale": state.get("plan_scale"),
            "path_steps": state.get("path_steps"),
            "plan_error": state.get("plan_error"),
            "left_tcp_pose": state.get("left_tcp_pose"),
            "right_tcp_pose": state.get("right_tcp_pose"),
            "active_tcp_pose": state.get("active_tcp_pose"),
            "active_gripper_state": state.get("active_gripper_state"),
            "gripper_per_step": state.get("gripper_per_step"),
            "controller_eval": state.get("controller_eval"),
            "telemetry_after": telemetry_after,
            "orientation_axis": selected.get("orientation_axis"),
            "orientation_axis_world": selected.get("orientation_axis_world"),
            "orientation_delta": selected.get("orientation_delta"),
            "primary_rad": selected.get("primary_rad"),
            "angle_delta_deg": selected.get("angle_delta_deg"),
            "translation_frame": selected.get("translation_frame"),
            "world_delta": selected.get("world_delta"),
            "world_axis": selected.get("world_axis"),
            "primary_world_delta": selected.get("primary_world_delta"),
            "off_axis_world_delta": selected.get("off_axis_world_delta"),
        }
        self.command_log.append(entry)
        if self.config.record_artifacts:
            self.write_manifest()
        selected = ((entry.get("controller_eval") or {}).get("selected") or {})
        return {
            "type": "trace_update",
            "step": idx + 1,
            "command": cmd,
            "arm": arm,
            "active_arm": arm,
            "planner": entry["planner"],
            "execution_mode": entry.get("execution_mode"),
            "path_steps": entry["path_steps"],
            "gripper_per_step": entry.get("gripper_per_step"),
            "active_gripper_state": entry.get("active_gripper_state"),
            "direction_ok": selected.get("direction_ok"),
            "uv_delta": selected.get("uv_delta"),
            "eef_delta_type": eef_delta_type,
            "translation_frame": selected.get("translation_frame"),
            "world_delta": selected.get("world_delta"),
            "world_axis": selected.get("world_axis"),
            "primary_world_delta": selected.get("primary_world_delta"),
            "off_axis_world_delta": selected.get("off_axis_world_delta"),
            "primary_px": selected.get("primary_px"),
            "off_axis_px": selected.get("off_axis_px"),
            "z_delta": selected.get("z_delta"),
            "action_14d": after.tolist(),
            "input_event": entry["input_event"],
            "telemetry": telemetry_after,
            "orientation_delta": selected.get("orientation_delta"),
            "angle_delta_deg": selected.get("angle_delta_deg"),
            "manifest": str(self.manifest_path),
        }

    def capture_frame(self, overlay: bool = True) -> np.ndarray:
        if self._adapter is None:
            self.reset()
        if self._adapter is None:
            raise RuntimeError("Task3 session is not initialized")
        with _cwd(self.robotwin_root):
            raw = self._adapter.capture_rgb()
        if not overlay:
            return raw
        return self._replay.draw_overlay(
            raw,
            self._adapter.active_command,
            self._adapter.active_mode,
            self._adapter.active_step_idx,
            right_locked=True,
        )

    def build_manifest(self) -> dict[str, Any]:
        ann = self.ann or {}
        adapter = self._adapter
        arms_used = sorted({str(row.get("arm") or "left") for row in self.command_log})
        reference = self.reference_metadata()
        return {
            "task_id": "task3",
            "source_type": "human_teleoperation",
            "type": "robotwin_simulator_bimanual_controls_replay" if "right" in arms_used else "robotwin_simulator_left_controls_replay",
            "sample_type": "operator_console_user_teleop",
            "episode_id": self.config.episode_id,
            "task": ann.get("task"),
            "source_episode_index": int(ann.get("source_episode_index", -1)),
            "instruction": (ann.get("texts") or [""])[0],
            "task_config": self.config.task_config,
            "frame_id": int(self.config.frame_id),
            "raw_action_index": int(self.raw_action_id if self.raw_action_id is not None else -1),
            **reference,
            "commands": [row["command"] for row in self.command_log],
            "command_arms": [row.get("arm", "left") for row in self.command_log],
            "action_14d": [row["after_action"] for row in self.command_log],
            "arms_used": arms_used,
            "operator_control_mode": "eef_6d",
            "control_mode": self.config.control_mode,
            "fixed_delta_source": str(self.config.fixed_delta_source) if self.config.fixed_delta_source else None,
            "fixed_delta_commands": sorted(self.fixed_deltas),
            "screen_step_px": float(self.config.screen_step_px),
            "prefix_replay_steps_per_action": int(self.config.prefix_replay_steps_per_action),
            "gripper_step": float(self.config.gripper_step),
            "gripper_interpolation_steps": int(self.config.gripper_interpolation_steps),
            "runtime_execution_mode": self.config.runtime_execution_mode,
            "hold_frames": int(self.config.hold_frames),
            "record_every_path_step": int(self.config.record_every_path_step),
            "initial_left_action": self.initial_left.tolist() if self.initial_left is not None else None,
            "initial_right_action": self.fixed_right.tolist() if self.fixed_right is not None else None,
            "right_locked_action": (
                self.fixed_right.tolist()
                if self.fixed_right is not None and (not arms_used or arms_used == ["left"])
                else None
            ),
            "video": str(self.video_path),
            "raw_video": str(self.raw_video_path),
            "first_frame": str(self.first_frame_path),
            "fps": float(self.config.fps),
            "frames": int(adapter.frames if adapter is not None else 0),
            "raw_frames": int(adapter.raw_frames if adapter is not None else 0),
            "size": adapter.video_size if adapter is not None else None,
            "raw_size": adapter.raw_video_size if adapter is not None else None,
            "reset_state": self.reset_state,
            "telemetry": self.telemetry(),
            "command_log": self.command_log,
            "elapsed_sec": round(time.time() - self._started, 3) if self._started else 0.0,
        }

    def write_manifest(self) -> dict[str, Any]:
        manifest = json_safe(self.build_manifest())
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def build_dataset_row(self, manifest: dict[str, Any]) -> dict[str, Any]:
        eval_path = str(self.eval_path) if self.eval_path.exists() else None
        return {
            "task_id": "task3",
            "source_type": "human_teleoperation",
            "task": manifest.get("task"),
            "episode": manifest.get("source_episode_index"),
            "episode_id": manifest.get("episode_id"),
            "instruction": manifest.get("instruction"),
            "output_video": str(self.video_path),
            "output_trace": str(self.manifest_path),
            "output_raw_video": str(self.raw_video_path),
            "first_frame": str(self.first_frame_path),
            "reference_video": manifest.get("reference_video"),
            "metadata": {
                "type": "worldsimprobe_task3_operator_sample_v1",
                "sample_type": manifest.get("sample_type"),
                "frame_id": manifest.get("frame_id"),
                "raw_action_index": manifest.get("raw_action_index"),
                "reference_video": manifest.get("reference_video"),
                "reference_fps": manifest.get("reference_fps"),
                "reference_start_frame": manifest.get("reference_start_frame"),
                "reference_start_time_sec": manifest.get("reference_start_time_sec"),
                "control_mode": manifest.get("control_mode"),
                "screen_step_px": float(self.config.screen_step_px),
                "prefix_replay_steps_per_action": int(self.config.prefix_replay_steps_per_action),
                "gripper_step": float(self.config.gripper_step),
                "gripper_interpolation_steps": int(self.config.gripper_interpolation_steps),
                "runtime_execution_mode": self.config.runtime_execution_mode,
                "command_count": len(manifest.get("commands") or []),
                "commands": manifest.get("commands") or [],
                "command_arms": manifest.get("command_arms") or [],
                "arms_used": manifest.get("arms_used") or [],
                "operator_control_mode": manifest.get("operator_control_mode"),
                "frames": manifest.get("frames"),
                "raw_frames": manifest.get("raw_frames"),
                "eval_json": eval_path,
            },
        }

    def write_dataset_row(self, manifest: dict[str, Any]) -> dict[str, Any]:
        row = json_safe(self.build_dataset_row(manifest))
        self.dataset_row_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        return row

    def export_trace(self, finalize_video: bool = False) -> dict[str, Any]:
        if finalize_video:
            self.close_writer()
        manifest = self.write_manifest()
        dataset_row = self.write_dataset_row(manifest)
        return {
            "trace": str(self.manifest_path),
            "sample": str(self.dataset_row_path),
            "video": str(self.video_path),
            "raw_video": str(self.raw_video_path),
            "first_frame": str(self.first_frame_path),
            "manifest": manifest,
            "dataset_row": dataset_row,
        }

    def evaluate_trace(self, expected_commands: list[str] | None = None) -> dict[str, Any]:
        self.write_manifest()
        min_screen_primary_px = min(12.0, max(1.5, float(self.config.screen_step_px) * 0.6))
        result = evaluate_task3_trace(
            trace_path=self.manifest_path,
            expected_commands=expected_commands or [row["command"] for row in self.command_log],
            min_screen_primary_px=min_screen_primary_px,
        )
        self.eval_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    def close_writer(self) -> None:
        if self._adapter is not None and not self._writer_closed:
            self._adapter.close_writer()
            self._writer_closed = True

    def close(self) -> None:
        if self._adapter is not None:
            try:
                self.close_writer()
            finally:
                self._adapter.close()
        self._adapter = None
        self._closed = True

    def status(self) -> dict[str, Any]:
        adapter = self._adapter
        return {
            "state": "closed" if self._closed else "ready" if adapter is not None else "idle",
            "episode_id": self.config.episode_id,
            "frame_id": int(self.config.frame_id),
            "task": (self.ann or {}).get("task"),
            "source_episode_index": (self.ann or {}).get("source_episode_index"),
            "control_mode": self.config.control_mode,
            "operator_control_mode": "eef_6d",
            "screen_step_px": float(self.config.screen_step_px),
            "prefix_replay_steps_per_action": int(self.config.prefix_replay_steps_per_action),
            "gripper_step": float(self.config.gripper_step),
            "gripper_interpolation_steps": int(self.config.gripper_interpolation_steps),
            "runtime_execution_mode": self.config.runtime_execution_mode,
            "hold_frames": int(self.config.hold_frames),
            "record_every_path_step": int(self.config.record_every_path_step),
            "show_overlay": bool(self.config.show_overlay),
            "fps": float(self.config.fps),
            "command_count": len(self.command_log),
            "frames": int(adapter.frames if adapter is not None else 0),
            **self.reference_metadata(),
            "telemetry": self.telemetry(),
            "active_arm": getattr(adapter, "active_arm", None) if adapter is not None else None,
            "active_command": (
                f"{getattr(adapter, 'active_arm', 'left')} {adapter.active_command}"
                if adapter is not None and adapter.active_command
                else None
            ),
            "active_mode": adapter.active_mode if adapter is not None else None,
            "output_dir": str(self.output_dir),
            "trace": str(self.manifest_path),
            "video": str(self.video_path),
            "raw_video": str(self.raw_video_path),
            "first_frame": str(self.first_frame_path),
        }
