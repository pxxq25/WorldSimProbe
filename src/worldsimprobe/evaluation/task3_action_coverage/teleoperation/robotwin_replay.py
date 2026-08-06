from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from worldsimprobe.evaluation.task3_action_coverage.teleoperation.robotwin_adapter import (
    LeftCameraTeleopAdapter,
)


TRANSLATION_COMMANDS = ["up", "down", "left", "right", "lift", "delift"]
ROTATION_COMMANDS = ["roll_neg", "roll_pos", "pitch_pos", "pitch_neg", "yaw_neg", "yaw_pos"]
GRIPPER_COMMANDS = ["open", "close"]
COMMANDS = TRANSLATION_COMMANDS + ROTATION_COMMANDS + GRIPPER_COMMANDS
LABELS = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "lift": "LIFT",
    "open": "OPEN",
    "close": "CLOSE",
    "delift": "DELIFT",
    "roll_neg": "ROLL-",
    "roll_pos": "ROLL+",
    "pitch_pos": "PITCH+",
    "pitch_neg": "PITCH-",
    "yaw_neg": "YAW-",
    "yaw_pos": "YAW+",
    "start": "START",
}


def capture_head_camera_rgb(camera_manager):
    """Render and read only the camera used by the live operator stream."""
    for camera, camera_name in zip(
        camera_manager.static_camera_list,
        camera_manager.static_camera_name,
    ):
        if camera_name != "head_camera":
            continue
        camera.take_picture()
        rgba = camera.get_picture("Color")
        rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
        return np.ascontiguousarray(rgb)

    # Preserve compatibility with camera managers that do not expose the
    # static-camera collection used by current RoboTwin releases.
    camera_manager.update_picture()
    rgb = camera_manager.get_rgb()["head_camera"]["rgb"][:, :, :3]
    return np.ascontiguousarray(rgb.astype(np.uint8, copy=False))


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found and imageio_ffmpeg unavailable") from exc


def start_writer(path, rgb, fps):
    h, w = rgb.shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        "21",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE), f"{w}x{h}"


def draw_overlay(rgb, command, mode, step_idx, right_locked=True):
    import cv2

    frame = np.ascontiguousarray(rgb.copy())
    h, w = frame.shape[:2]
    scale = max(0.55, min(1.5, w / 960.0))
    pad = int(8 * scale)
    panel_w = int(286 * scale)
    panel_h = int(166 * scale)
    x0, y0 = pad, pad

    overlay = frame.copy()
    panel_color = (18, 14, 10) if mode == "hold" else (18, 14, 10)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), panel_color, -1)
    alpha = 0.72
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 1)

    title = "LEFT ARM CONTROLS"
    status = "RIGHT ARM LOCKED" if right_locked else "RIGHT ARM UNLOCKED"
    label = LABELS.get(command, str(command).upper())
    active = f"ACTIVE: {label}" if mode == "action" else f"PAUSE AFTER: {label}"
    cv2.putText(frame, title, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (x0 + 10, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (135, 235, 135), 1, cv2.LINE_AA)
    cv2.putText(frame, active, (x0 + 10, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (80, 225, 255), 1, cv2.LINE_AA)

    btn_w = int(62 * scale)
    btn_h = int(25 * scale)
    gap = int(6 * scale)
    bx = x0 + 10
    by = y0 + 76
    positions = {
        "up": (bx + btn_w + gap, by),
        "left": (bx, by + btn_h + gap),
        "right": (bx + 2 * (btn_w + gap), by + btn_h + gap),
        "down": (bx + btn_w + gap, by + 2 * (btn_h + gap)),
        "lift": (bx, by + 3 * (btn_h + gap)),
        "delift": (bx + btn_w + gap, by + 3 * (btn_h + gap)),
        "open": (x0 + panel_w - btn_w - 10, by),
        "close": (x0 + panel_w - btn_w - 10, by + btn_h + gap),
    }
    for key, (x, y) in positions.items():
        is_active = key == command
        fill = (69, 202, 255) if is_active and mode == "action" else (104, 226, 255) if is_active else (58, 48, 38)
        text = (20, 18, 16) if is_active else (245, 240, 235)
        cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), fill, -1)
        cv2.rectangle(frame, (x, y), (x + btn_w, y + btn_h), (230, 230, 230), 1)
        name = LABELS[key]
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, 1)
        cv2.putText(
            frame,
            name,
            (x + max(2, (btn_w - tw) // 2), y + max(th + 2, (btn_h + th) // 2 - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36 * scale,
            text,
            1,
            cv2.LINE_AA,
        )

    strip_h = int(28 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    progress = "START" if step_idx < 0 else f"{step_idx + 1}/9  {label}"
    if mode == "hold" and step_idx >= 0:
        progress = f"HOLD {step_idx + 1}/9 AFTER {label}"
    cv2.putText(frame, progress, (pad, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    if mode == "hold":
        cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (104, 226, 255), max(1, int(2 * scale)))
    return frame


class RecordingTeleopAdapter(LeftCameraTeleopAdapter):
    def __init__(
        self,
        *args,
        video_path,
        raw_video_path,
        first_frame_path,
        fps,
        record_every_path_step,
        hold_frames,
        screen_step_px,
        gripper_step,
        gripper_interpolation_steps,
        max_joint_delta,
        fd_joint_eps,
        control_mode,
        fixed_deltas,
        fixed_delta_source,
        runtime_execution_mode="physical_drive_target",
        show_overlay=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.video_path = Path(video_path) if video_path else None
        self.raw_video_path = Path(raw_video_path) if raw_video_path else None
        self.first_frame_path = Path(first_frame_path) if first_frame_path else None
        self.fps = float(fps)
        self.record_every_path_step = max(1, int(record_every_path_step))
        self.hold_frames = max(0, int(hold_frames))
        self.screen_step_px = float(screen_step_px)
        self.gripper_step = float(gripper_step)
        self.gripper_interpolation_steps = max(1, int(gripper_interpolation_steps))
        self.max_joint_delta = float(max_joint_delta)
        self.fd_joint_eps = float(fd_joint_eps)
        self.control_mode = str(control_mode)
        self.runtime_execution_mode = str(runtime_execution_mode)
        if self.runtime_execution_mode not in {"physical_drive_target", "force_qpos"}:
            raise ValueError(f"Unsupported runtime execution mode: {self.runtime_execution_mode}")
        self.show_overlay = bool(show_overlay)
        self.fixed_deltas = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in (fixed_deltas or {}).items()
        }
        self.fixed_delta_source = fixed_delta_source
        self.writer = None
        self.raw_writer = None
        self.video_size = None
        self.raw_video_size = None
        self.frames = 0
        self.raw_frames = 0
        self.first_frame_saved = False
        self.active_command = "start"
        self.active_arm = "left"
        self.active_mode = "hold"
        self.active_step_idx = -1
        self._last_command_eval = None
        self.frame_callback = None

    def close_writer(self):
        if self.writer is not None:
            self.writer.stdin.close()
            rc = self.writer.wait()
            self.writer = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with code {rc}: {self.video_path}")
        if self.raw_writer is not None:
            self.raw_writer.stdin.close()
            rc = self.raw_writer.wait()
            self.raw_writer = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with code {rc}: {self.raw_video_path}")

    def set_frame_callback(self, callback):
        self.frame_callback = callback

    def _apply_prefix_drive_action(self, action, steps_per_action):
        action = np.asarray(action, dtype=np.float64)
        robot = self.task.robot
        left = action[: self.left_dim]
        left_gripper = float(action[self.left_dim])
        right_start = self.left_dim + 1
        right = action[right_start : right_start + self.right_dim]
        right_gripper = float(action[right_start + self.right_dim])
        robot.set_arm_joints(left, np.zeros(self.left_dim), "left")
        robot.set_arm_joints(right, np.zeros(self.right_dim), "right")
        robot.set_gripper(left_gripper, "left", gripper_eps=0)
        robot.set_gripper(right_gripper, "right", gripper_eps=0)
        for _ in range(max(0, int(steps_per_action))):
            self.task.scene.step()
        self.current_action = action.copy()

    def _replay_prefix_to_action(self, action_sequence, raw_action_id, steps_per_action):
        actions = np.asarray(action_sequence, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D action sequence for prefix replay, got {actions.shape}")
        if actions.shape[0] == 0:
            raise ValueError("Cannot prefix replay an empty action sequence")
        raw_idx = min(max(0, int(raw_action_id)), actions.shape[0] - 1)
        replay_count = 0
        if raw_idx > 0:
            for action in actions[:raw_idx]:
                self._apply_prefix_drive_action(action, steps_per_action)
                replay_count += 1
        self.current_action = actions[raw_idx].copy()
        self.force_set_action_qpos(self.current_action, step=False)
        return {
            "enabled": bool(raw_idx > 0),
            "raw_action_index": int(raw_idx),
            "replayed_actions": int(replay_count),
            "steps_per_action": int(steps_per_action),
        }

    def reset(self, task_name, episode, action, action_sequence=None, raw_action_id=0, prefix_steps_per_action=15):
        if action_sequence is not None:
            actions = np.asarray(action_sequence, dtype=np.float64)
            raw_idx = min(max(0, int(raw_action_id)), actions.shape[0] - 1)
            state = super().reset(task_name, episode, actions[0])
            if raw_idx > 0:
                state["pre_prefix_alignment"] = self.settle_current_action()
            state["prefix_replay"] = self._replay_prefix_to_action(
                actions,
                raw_idx,
                prefix_steps_per_action,
            )
            # Prefix replay preserves object/fixture state; this final force-set
            # makes robot qpos exactly match the selected reference action.
            self.current_action = np.asarray(action, dtype=np.float64).copy()
            self.force_set_action_qpos(self.current_action, step=False)
            state["prefix_start_action"] = state.get("action")
            state["action"] = self.current_action.tolist()
            state["left_tcp_pose"] = self.task.robot.get_left_tcp_pose()
            state["right_tcp_pose"] = self.task.robot.get_right_tcp_pose()
            alignment = self.action_alignment(self.current_action)
            alignment["settle_steps"] = 0
            alignment["tolerance"] = 1e-6
            alignment["settled"] = bool(alignment["max_abs_error"] <= alignment["tolerance"])
            if not alignment["settled"]:
                raise RuntimeError(f"Initial action did not align before first frame: {alignment}")
            state["initial_alignment"] = alignment
        else:
            state = super().reset(task_name, episode, action)
            state["prefix_replay"] = {"enabled": False}
            state["initial_alignment"] = self.settle_current_action()
        return state

    def _active_joint_index(self, entity, joint):
        active = list(entity.get_active_joints())
        return active.index(joint)

    def _arm_qpos(self, arm_tag):
        entity = self.task.robot.left_entity if arm_tag == "left" else self.task.robot.right_entity
        joints = self.task.robot.left_arm_joints if arm_tag == "left" else self.task.robot.right_arm_joints
        entity_qpos = entity.get_qpos()
        return np.asarray([entity_qpos[self._active_joint_index(entity, joint)] for joint in joints], dtype=np.float64)

    def _set_gripper_joint_qpos(self, arm_tag, gripper_val):
        robot = self.task.robot
        entity = robot.left_entity if arm_tag == "left" else robot.right_entity
        joints = robot.left_gripper if arm_tag == "left" else robot.right_gripper
        scale = robot.left_gripper_scale if arm_tag == "left" else robot.right_gripper_scale
        normalized = float(np.clip(gripper_val, 0.0, 1.0))
        base_qpos = float(scale[0]) + normalized * (float(scale[1]) - float(scale[0]))
        entity_qpos = entity.get_qpos()
        for joint, multiplier, offset in joints:
            if joint is None:
                continue
            entity_qpos[self._active_joint_index(entity, joint)] = base_qpos * float(multiplier) + float(offset)
        entity.set_qpos(entity_qpos)
        robot.set_gripper(normalized, arm_tag, gripper_eps=0)

    def force_set_current_grippers(self):
        action = np.asarray(self.current_action, dtype=np.float64)
        self._set_gripper_joint_qpos("left", float(action[self.left_dim]))
        self._set_gripper_joint_qpos("right", float(action[self.left_dim + 1 + self.right_dim]))

    def set_current_gripper_drive_targets(self):
        action = np.asarray(self.current_action, dtype=np.float64)
        robot = self.task.robot
        robot.set_gripper(float(action[self.left_dim]), "left", gripper_eps=0)
        robot.set_gripper(float(action[self.left_dim + 1 + self.right_dim]), "right", gripper_eps=0)

    def _gripper_joint_values(self, arm_tag, source):
        robot = self.task.robot
        entity = robot.left_entity if arm_tag == "left" else robot.right_entity
        joints = robot.left_gripper if arm_tag == "left" else robot.right_gripper
        scale = robot.left_gripper_scale if arm_tag == "left" else robot.right_gripper_scale
        denom = float(scale[1]) - float(scale[0])
        values = []
        for joint, multiplier, offset in joints:
            if joint is None or abs(float(multiplier)) < 1e-12 or abs(denom) < 1e-12:
                continue
            if source == "qpos":
                raw_value = float(entity.get_qpos()[self._active_joint_index(entity, joint)])
            elif source == "drive_target":
                raw_value = float(joint.get_drive_target()[0])
            else:
                raise ValueError(f"Unsupported gripper value source: {source}")
            base_qpos = (raw_value - float(offset)) / float(multiplier)
            values.append((base_qpos - float(scale[0])) / denom)
        return np.asarray(values, dtype=np.float64)

    def gripper_runtime_state(self, arm_tag):
        action = np.asarray(self.current_action, dtype=np.float64)
        _, gripper_idx = self._arm_action_slice(arm_tag)
        qpos = self._gripper_joint_values(arm_tag, "qpos")
        drive = self._gripper_joint_values(arm_tag, "drive_target")
        return {
            "target": float(action[gripper_idx]),
            "qpos": qpos.tolist(),
            "drive_target": drive.tolist(),
            "qpos_mean": float(np.mean(qpos)) if qpos.size else None,
            "drive_target_mean": float(np.mean(drive)) if drive.size else None,
        }

    def force_set_action_qpos(self, action, step=False):
        action = np.asarray(action, dtype=np.float64)
        robot = self.task.robot
        left = action[: self.left_dim]
        left_gripper = float(action[self.left_dim])
        right_start = self.left_dim + 1
        right = action[right_start : right_start + self.right_dim]
        right_gripper = float(action[right_start + self.right_dim])

        self._set_arm_qpos(robot.left_entity, robot.left_arm_joints, left)
        self._set_arm_qpos(robot.right_entity, robot.right_arm_joints, right)
        robot.set_arm_joints(left, np.zeros(self.left_dim), "left")
        robot.set_arm_joints(right, np.zeros(self.right_dim), "right")
        self._set_gripper_joint_qpos("left", left_gripper)
        self._set_gripper_joint_qpos("right", right_gripper)
        if step:
            self.task.scene.step()

    def action_alignment(self, action):
        action = np.asarray(action, dtype=np.float64)
        left_target = action[: self.left_dim]
        left_gripper_target = float(action[self.left_dim])
        right_start = self.left_dim + 1
        right_target = action[right_start : right_start + self.right_dim]
        right_gripper_target = float(action[right_start + self.right_dim])

        left_qpos = self._arm_qpos("left")
        right_qpos = self._arm_qpos("right")
        left_gripper_qpos = self._gripper_joint_values("left", "qpos")
        right_gripper_qpos = self._gripper_joint_values("right", "qpos")
        left_gripper_drive = self._gripper_joint_values("left", "drive_target")
        right_gripper_drive = self._gripper_joint_values("right", "drive_target")

        def max_abs_error(values, target):
            if values.size == 0:
                return 0.0
            return float(np.max(np.abs(values - float(target))))

        errors = {
            "left_arm_max_abs_error": float(np.max(np.abs(left_qpos - left_target))) if left_qpos.size else 0.0,
            "right_arm_max_abs_error": float(np.max(np.abs(right_qpos - right_target))) if right_qpos.size else 0.0,
            "left_gripper_qpos_max_abs_error": max_abs_error(left_gripper_qpos, left_gripper_target),
            "right_gripper_qpos_max_abs_error": max_abs_error(right_gripper_qpos, right_gripper_target),
            "left_gripper_drive_max_abs_error": max_abs_error(left_gripper_drive, left_gripper_target),
            "right_gripper_drive_max_abs_error": max_abs_error(right_gripper_drive, right_gripper_target),
        }
        errors["max_abs_error"] = max(errors.values()) if errors else 0.0
        return {
            **errors,
            "left_gripper_target": left_gripper_target,
            "right_gripper_target": right_gripper_target,
            "left_gripper_qpos": left_gripper_qpos.tolist(),
            "right_gripper_qpos": right_gripper_qpos.tolist(),
            "left_gripper_drive": left_gripper_drive.tolist(),
            "right_gripper_drive": right_gripper_drive.tolist(),
        }

    def settle_current_action(self, settle_steps=12, tolerance=1e-6):
        if self.current_action is None:
            raise RuntimeError("Cannot settle before adapter reset")
        action = np.asarray(self.current_action, dtype=np.float64).copy()
        for _ in range(max(0, int(settle_steps))):
            self.force_set_action_qpos(action, step=True)
        self.force_set_action_qpos(action, step=False)
        alignment = self.action_alignment(action)
        alignment["settle_steps"] = int(settle_steps)
        alignment["tolerance"] = float(tolerance)
        alignment["settled"] = bool(alignment["max_abs_error"] <= float(tolerance))
        if not alignment["settled"]:
            raise RuntimeError(f"Initial action did not align before first frame: {alignment}")
        return alignment

    def capture_rgb(self):
        self.task._update_render()
        return capture_head_camera_rgb(self.task.cameras)

    def _arm_entity_joints(self, arm_tag):
        if arm_tag == "left":
            return self.task.robot.left_entity, self.task.robot.left_arm_joints
        if arm_tag == "right":
            return self.task.robot.right_entity, self.task.robot.right_arm_joints
        raise ValueError(f"Unsupported arm: {arm_tag}")

    def _arm_dim(self, arm_tag):
        return self.left_dim if arm_tag == "left" else self.right_dim

    def _arm_action_slice(self, arm_tag):
        if arm_tag == "left":
            return slice(0, self.left_dim), self.left_dim
        right_start = self.left_dim + 1
        return slice(right_start, right_start + self.right_dim), right_start + self.right_dim

    def _arm_active_joint_indices(self, arm_tag):
        entity, joints = self._arm_entity_joints(arm_tag)
        active = list(entity.get_active_joints())
        return [active.index(joint) for joint in joints]

    def _left_active_joint_indices(self):
        return self._arm_active_joint_indices("left")

    def left_arm_qpos(self):
        return self.arm_qpos("left")

    def arm_qpos(self, arm_tag):
        entity, _ = self._arm_entity_joints(arm_tag)
        entity_qpos = entity.get_qpos()
        return np.asarray([entity_qpos[idx] for idx in self._arm_active_joint_indices(arm_tag)], dtype=np.float64)

    def direct_set_left_arm(self, qpos, step=True):
        self.direct_set_arm("left", qpos, step=step)

    def direct_set_arm(self, arm_tag, qpos, step=True):
        qpos = np.asarray(qpos, dtype=np.float64)
        entity, _ = self._arm_entity_joints(arm_tag)
        dim = self._arm_dim(arm_tag)
        entity_qpos = entity.get_qpos()
        for joint_idx, value in zip(self._arm_active_joint_indices(arm_tag), qpos[:dim]):
            entity_qpos[joint_idx] = float(value)
        entity.set_qpos(entity_qpos)
        self.task.robot.set_arm_joints(qpos[:dim], np.zeros(dim), arm_tag)
        if step:
            self.task.scene.step()

    def project_tcp(self, arm_tag="left"):
        cfg = self.task.cameras.get_config()["head_camera"]
        k_mat = np.asarray(cfg["intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(cfg["extrinsic_cv"], dtype=np.float64)
        if arm_tag == "left":
            tcp = np.asarray(self.task.robot.get_left_tcp_pose(), dtype=np.float64)
        elif arm_tag == "right":
            tcp = np.asarray(self.task.robot.get_right_tcp_pose(), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported arm: {arm_tag}")
        camera_xyz = extrinsic @ np.r_[tcp[:3], 1.0]
        uvw = k_mat @ camera_xyz[:3]
        return {
            "uv": [float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])],
            "depth": float(camera_xyz[2]),
            "tcp_pose": tcp.tolist(),
        }

    def _evaluate_final_qpos(self, final_qpos, arm_tag="left"):
        original = self.arm_qpos(arm_tag)
        self.direct_set_arm(arm_tag, final_qpos, step=False)
        projection = self.project_tcp(arm_tag)
        self.direct_set_arm(arm_tag, original, step=False)
        return projection

    def _estimate_qpos_jacobian(self, base_qpos, arm_tag="left"):
        base_qpos = np.asarray(base_qpos, dtype=np.float64).copy()
        eps = max(self.fd_joint_eps, 1e-4)
        dim = self._arm_dim(arm_tag)
        self.direct_set_arm(arm_tag, base_qpos, step=False)
        before = self.project_tcp(arm_tag)
        uv0 = np.asarray(before["uv"], dtype=np.float64)
        tcp0 = np.asarray(before["tcp_pose"], dtype=np.float64)
        uv_jac = np.zeros((2, dim), dtype=np.float64)
        xyz_jac = np.zeros((3, dim), dtype=np.float64)
        for joint_idx in range(dim):
            plus = base_qpos.copy()
            minus = base_qpos.copy()
            plus[joint_idx] += eps
            minus[joint_idx] -= eps
            self.direct_set_arm(arm_tag, plus, step=False)
            plus_proj = self.project_tcp(arm_tag)
            self.direct_set_arm(arm_tag, minus, step=False)
            minus_proj = self.project_tcp(arm_tag)
            uv_jac[:, joint_idx] = (
                np.asarray(plus_proj["uv"], dtype=np.float64)
                - np.asarray(minus_proj["uv"], dtype=np.float64)
            ) / (2.0 * eps)
            xyz_jac[:, joint_idx] = (
                np.asarray(plus_proj["tcp_pose"][:3], dtype=np.float64)
                - np.asarray(minus_proj["tcp_pose"][:3], dtype=np.float64)
            ) / (2.0 * eps)
        self.direct_set_arm(arm_tag, base_qpos, step=False)
        return before, uv_jac, xyz_jac, uv0, tcp0

    def _regularized_delta(self, jacobian, target):
        jacobian = np.asarray(jacobian, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        reg = 1e-2
        system = jacobian @ jacobian.T + reg * np.eye(jacobian.shape[0])
        delta = jacobian.T @ np.linalg.solve(system, target)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs > self.max_joint_delta:
            delta *= self.max_joint_delta / max_abs
        return delta

    @staticmethod
    def _normalize_quat(quat):
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / norm

    @classmethod
    def _quat_to_mat(cls, quat):
        w, x, y, z = cls._normalize_quat(quat)
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _quat_delta_rotvec(cls, before_quat, after_quat):
        before_mat = cls._quat_to_mat(before_quat)
        after_mat = cls._quat_to_mat(after_quat)
        rel = after_mat @ before_mat.T
        cos_angle = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        skew_vec = np.array(
            [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]],
            dtype=np.float64,
        )
        if angle < 1e-8:
            return 0.5 * skew_vec
        denom = 2.0 * np.sin(angle)
        if abs(denom) < 1e-8:
            return 0.5 * skew_vec
        return (skew_vec / denom) * angle

    def _qpos_candidates(self, base_delta):
        candidates = []
        for scale in (1.5, 1.25, 1.0, 0.75, 0.5, 0.25):
            delta = np.asarray(base_delta, dtype=np.float64) * scale
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            if max_abs > self.max_joint_delta:
                delta *= self.max_joint_delta / max_abs
            candidates.append(delta)

        basis_step = min(self.max_joint_delta, 0.08)
        dim = len(np.asarray(base_delta, dtype=np.float64))
        for joint_idx in range(dim):
            for sign in (-1.0, 1.0):
                delta = np.zeros(dim, dtype=np.float64)
                delta[joint_idx] = sign * basis_step
                candidates.append(delta)

        unique = []
        for delta in candidates:
            if np.linalg.norm(delta) < 1e-9:
                continue
            if not any(np.linalg.norm(delta - old) < 1e-6 for old in unique):
                unique.append(delta)
        return unique

    def _path_from_delta(self, start_qpos, delta):
        start_qpos = np.asarray(start_qpos, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        steps = max(10, min(60, int(np.ceil(max_abs / 0.010)) + 1))
        position = np.linspace(start_qpos, start_qpos + delta, steps)
        velocity = np.gradient(position, axis=0) if len(position) > 1 else np.zeros_like(position)
        return {
            "status": "Success",
            "position": position,
            "velocity": velocity,
        }

    def _choose_joint_table_plan(self, command, arm_tag="left"):
        base_qpos = self.arm_qpos(arm_tag)
        before, uv_jac, xyz_jac, _, _ = self._estimate_qpos_jacobian(base_qpos, arm_tag)
        before_tcp = np.asarray(before["tcp_pose"], dtype=np.float64)
        desired_axis = {
            "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "down": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        }[command]
        target_m = abs(self.planar_step) * (self.lateral_step_mult if command in {"left", "right"} else 1.0)
        target_delta = desired_axis * target_m
        z_hold_weight = 30.0
        servo_jac = np.vstack([xyz_jac[:2, :], z_hold_weight * xyz_jac[2:3, :]])
        servo_target = np.r_[target_delta[:2], 0.0]
        base_delta = self._regularized_delta(servo_jac, servo_target)
        best = None
        for delta in self._qpos_candidates(base_delta):
            projection = self._evaluate_final_qpos(base_qpos + delta, arm_tag)
            after_tcp = np.asarray(projection["tcp_pose"], dtype=np.float64)
            world_delta = after_tcp[:3] - before_tcp[:3]
            primary = float(world_delta @ desired_axis)
            off_axis = float(np.linalg.norm(world_delta - desired_axis * primary))
            z_delta = float(world_delta[2])
            uv_delta = np.asarray(projection["uv"], dtype=np.float64) - np.asarray(before["uv"], dtype=np.float64)
            q_norm = float(np.linalg.norm(delta))
            overshoot = max(0.0, primary - 1.75 * target_m)
            score = primary - 0.85 * off_axis - 0.50 * abs(z_delta) - 0.002 * q_norm - 0.25 * overshoot
            candidate = {
                "score": float(score),
                "eef_delta_type": "table_translation",
                "translation_frame": "world_table",
                "world_axis": desired_axis.tolist(),
                "target_world_delta": target_delta.tolist(),
                "world_delta": world_delta.tolist(),
                "primary_world_delta": primary,
                "off_axis_world_delta": off_axis,
                "z_delta": z_delta,
                "xy_delta": float(np.linalg.norm(world_delta[:2])),
                "uv_delta": uv_delta.tolist(),
                "direction_ok": bool(primary > 0.0),
                "q_delta_norm": q_norm,
                "q_delta": delta.tolist(),
                "target_qpos": (base_qpos + delta).tolist(),
                "projection": projection,
                "jacobian_uv": uv_jac.tolist(),
                "jacobian_xyz": xyz_jac.tolist(),
                "plan": self._path_from_delta(base_qpos, delta),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        min_primary = max(0.002, 0.20 * target_m)
        if best is None or best["primary_world_delta"] < min_primary:
            raise RuntimeError(f"No verified joint table-space plan for {command}: best={best}")
        return best, before

    def _choose_joint_vertical_plan(self, command, arm_tag="left"):
        base_qpos = self.arm_qpos(arm_tag)
        before, uv_jac, xyz_jac, _, _ = self._estimate_qpos_jacobian(base_qpos, arm_tag)
        sign = 1.0 if command == "lift" else -1.0
        target_z = max(abs(self.vertical_step), 0.006)
        base_delta = self._regularized_delta(xyz_jac[2:3, :], np.array([sign * target_z], dtype=np.float64))
        best = None
        before_tcp = np.asarray(before["tcp_pose"], dtype=np.float64)
        for delta in self._qpos_candidates(base_delta):
            projection = self._evaluate_final_qpos(base_qpos + delta, arm_tag)
            after_tcp = np.asarray(projection["tcp_pose"], dtype=np.float64)
            dz = float(after_tcp[2] - before_tcp[2])
            xy = float(np.linalg.norm(after_tcp[:2] - before_tcp[:2]))
            uv_delta = np.asarray(projection["uv"], dtype=np.float64) - np.asarray(before["uv"], dtype=np.float64)
            q_norm = float(np.linalg.norm(delta))
            score = sign * dz - 0.45 * xy - 0.0005 * float(np.linalg.norm(uv_delta)) - 0.002 * q_norm
            candidate = {
                "score": float(score),
                "z_delta": dz,
                "xy_delta": xy,
                "uv_delta": uv_delta.tolist(),
                "direction_ok": bool(sign * dz > 0.0),
                "q_delta_norm": q_norm,
                "q_delta": delta.tolist(),
                "target_qpos": (base_qpos + delta).tolist(),
                "projection": projection,
                "jacobian_uv": uv_jac.tolist(),
                "jacobian_xyz": xyz_jac.tolist(),
                "plan": self._path_from_delta(base_qpos, delta),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None or sign * best["z_delta"] < 0.002:
            raise RuntimeError(f"No verified joint vertical plan for {command}: best={best}")
        return best, before

    def _estimate_qpos_orientation_jacobian(self, base_qpos, arm_tag="left"):
        base_qpos = np.asarray(base_qpos, dtype=np.float64).copy()
        eps = max(self.fd_joint_eps, 1e-4)
        dim = self._arm_dim(arm_tag)
        self.direct_set_arm(arm_tag, base_qpos, step=False)
        before = self.project_tcp(arm_tag)
        tcp0 = np.asarray(before["tcp_pose"], dtype=np.float64)
        quat0 = tcp0[3:7]
        xyz_jac = np.zeros((3, dim), dtype=np.float64)
        rot_jac = np.zeros((3, dim), dtype=np.float64)
        for joint_idx in range(dim):
            plus = base_qpos.copy()
            minus = base_qpos.copy()
            plus[joint_idx] += eps
            minus[joint_idx] -= eps
            self.direct_set_arm(arm_tag, plus, step=False)
            plus_proj = self.project_tcp(arm_tag)
            self.direct_set_arm(arm_tag, minus, step=False)
            minus_proj = self.project_tcp(arm_tag)
            plus_tcp = np.asarray(plus_proj["tcp_pose"], dtype=np.float64)
            minus_tcp = np.asarray(minus_proj["tcp_pose"], dtype=np.float64)
            xyz_jac[:, joint_idx] = (plus_tcp[:3] - minus_tcp[:3]) / (2.0 * eps)
            rot_plus = self._quat_delta_rotvec(quat0, plus_tcp[3:7])
            rot_minus = self._quat_delta_rotvec(quat0, minus_tcp[3:7])
            rot_jac[:, joint_idx] = (rot_plus - rot_minus) / (2.0 * eps)
        self.direct_set_arm(arm_tag, base_qpos, step=False)
        return before, xyz_jac, rot_jac, tcp0

    def _choose_joint_rotation_plan(self, command, arm_tag="left"):
        base_qpos = self.arm_qpos(arm_tag)
        before, xyz_jac, rot_jac, before_tcp = self._estimate_qpos_orientation_jacobian(base_qpos, arm_tag)
        axis_name, sign = {
            "roll_neg": ("roll", -1.0),
            "roll_pos": ("roll", 1.0),
            "pitch_neg": ("pitch", -1.0),
            "pitch_pos": ("pitch", 1.0),
            "yaw_neg": ("yaw", -1.0),
            "yaw_pos": ("yaw", 1.0),
        }[command]
        local_axis = {
            "roll": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "pitch": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "yaw": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }[axis_name]
        world_axis = self._quat_to_mat(before_tcp[3:7]) @ local_axis
        world_axis = world_axis / max(float(np.linalg.norm(world_axis)), 1e-12)
        signed_axis = sign * world_axis
        target_rad = max(abs(float(getattr(self, "rotate_rad", math.radians(15.0)))), math.radians(1.0))
        target_rotvec = signed_axis * target_rad
        xyz_hold_weight = 30.0
        servo_jac = np.vstack([rot_jac, xyz_hold_weight * xyz_jac])
        servo_target = np.r_[target_rotvec, np.zeros(3, dtype=np.float64)]
        base_delta = self._regularized_delta(servo_jac, servo_target)
        best = None
        for delta in self._qpos_candidates(base_delta):
            projection = self._evaluate_final_qpos(base_qpos + delta, arm_tag)
            after_tcp = np.asarray(projection["tcp_pose"], dtype=np.float64)
            rotvec = self._quat_delta_rotvec(before_tcp[3:7], after_tcp[3:7])
            primary = float(rotvec @ signed_axis)
            off_axis = float(np.linalg.norm(rotvec - signed_axis * primary))
            xyz_drift = float(np.linalg.norm(after_tcp[:3] - before_tcp[:3]))
            q_norm = float(np.linalg.norm(delta))
            score = primary - 0.65 * off_axis - 20.0 * xyz_drift - 0.002 * q_norm - 0.10 * max(0.0, primary - 1.6 * target_rad)
            candidate = {
                "score": float(score),
                "eef_delta_type": "rotation",
                "orientation_axis": axis_name,
                "orientation_sign": float(sign),
                "orientation_axis_world": signed_axis.tolist(),
                "orientation_delta": rotvec.tolist(),
                "target_angle_rad": float(target_rad),
                "primary_rad": primary,
                "off_axis_rad": off_axis,
                "angle_delta_deg": float(math.degrees(primary)),
                "xyz_drift": xyz_drift,
                "direction_ok": bool(primary > 0.0),
                "q_delta_norm": q_norm,
                "q_delta": delta.tolist(),
                "target_qpos": (base_qpos + delta).tolist(),
                "projection": projection,
                "jacobian_orientation": rot_jac.tolist(),
                "jacobian_xyz": xyz_jac.tolist(),
                "plan": self._path_from_delta(base_qpos, delta),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        min_primary = max(math.radians(1.0), min(0.15 * target_rad, math.radians(4.0)))
        if best is None or best["primary_rad"] < min_primary:
            raise RuntimeError(f"No verified joint orientation plan for {command}: best={best}")
        return best, before

    def _choose_fixed_delta_plan(self, command, arm_tag="left"):
        if command not in self.fixed_deltas:
            raise RuntimeError(f"No fixed qpos delta available for {command}")
        base_qpos = self.arm_qpos(arm_tag)
        before = self.project_tcp(arm_tag)
        delta = np.asarray(self.fixed_deltas[command], dtype=np.float64)
        dim = self._arm_dim(arm_tag)
        if delta.shape[0] != dim:
            raise RuntimeError(
                f"Fixed qpos delta for {command} has dim {delta.shape[0]}, expected {dim}"
            )
        projection = self._evaluate_final_qpos(base_qpos + delta, arm_tag)
        after_uv = np.asarray(projection["uv"], dtype=np.float64)
        before_uv = np.asarray(before["uv"], dtype=np.float64)
        after_tcp = np.asarray(projection["tcp_pose"], dtype=np.float64)
        before_tcp = np.asarray(before["tcp_pose"], dtype=np.float64)
        selected = {
            "q_delta": delta.tolist(),
            "target_qpos": (base_qpos + delta).tolist(),
            "projection": projection,
            "q_delta_norm": float(np.linalg.norm(delta)),
            "source": self.fixed_delta_source,
            "plan": self._path_from_delta(base_qpos, delta),
        }
        if command in {"up", "down", "left", "right"}:
            score, primary, off_axis, magnitude = self._score_screen_candidate(command, before_uv, after_uv)
            selected.update({
                "score": float(score),
                "primary_px": float(primary),
                "off_axis_px": float(off_axis),
                "magnitude_px": float(magnitude),
                "z_drift": float(abs(after_tcp[2] - before_tcp[2])),
                "uv_delta": (after_uv - before_uv).tolist(),
                "z_delta": float(after_tcp[2] - before_tcp[2]),
                "direction_ok": bool(primary > 0.0),
            })
        elif command in {"lift", "delift"}:
            sign = 1.0 if command == "lift" else -1.0
            selected.update({
                "z_delta": float(after_tcp[2] - before_tcp[2]),
                "xy_delta": float(np.linalg.norm(after_tcp[:2] - before_tcp[:2])),
                "uv_delta": (after_uv - before_uv).tolist(),
                "direction_ok": bool(sign * (after_tcp[2] - before_tcp[2]) > 0.0),
            })
        return selected, before

    def _score_screen_candidate(self, command, before_uv, after_uv):
        desired = {
            "up": np.array([0.0, -1.0]),
            "down": np.array([0.0, 1.0]),
            "left": np.array([-1.0, 0.0]),
            "right": np.array([1.0, 0.0]),
        }[command]
        delta = np.asarray(after_uv, dtype=np.float64) - np.asarray(before_uv, dtype=np.float64)
        primary = float(delta @ desired)
        off_axis = float(abs(delta @ np.array([-desired[1], desired[0]])))
        magnitude = float(np.linalg.norm(delta))
        return primary - 0.45 * off_axis - 0.015 * max(0.0, magnitude - 80.0), primary, off_axis, magnitude

    def record_frame(self, command=None, mode=None, step_idx=None):
        if command is not None:
            self.active_command = command
        if mode is not None:
            self.active_mode = mode
        if step_idx is not None:
            self.active_step_idx = step_idx
        raw_rgb = self.capture_rgb()
        if self.first_frame_path is not None and not self.first_frame_saved:
            import imageio.v2 as imageio

            self.first_frame_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.first_frame_path, raw_rgb)
            self.first_frame_saved = True
        if self.raw_video_path is not None:
            if self.raw_writer is None:
                self.raw_writer, self.raw_video_size = start_writer(self.raw_video_path, raw_rgb, self.fps)
            self.raw_writer.stdin.write(raw_rgb.tobytes())
            self.raw_frames += 1
        if self.show_overlay:
            rgb = draw_overlay(
                raw_rgb,
                self.active_command,
                self.active_mode,
                self.active_step_idx,
                right_locked=True,
            )
        else:
            rgb = raw_rgb.copy()
        if self.video_path is not None:
            if self.writer is None:
                self.writer, self.video_size = start_writer(self.video_path, rgb, self.fps)
            self.writer.stdin.write(rgb.tobytes())
        self.frames += 1
        if self.frame_callback is not None:
            try:
                self.frame_callback(rgb.copy())
            except Exception:
                self.frame_callback = None

    def _execute_left_path(self, result):
        return self._execute_arm_path("left", result)

    def _execute_arm_path(self, arm_tag, result):
        position = np.asarray(result["position"], dtype=np.float64)
        velocity = np.asarray(result.get("velocity", np.zeros_like(position)), dtype=np.float64)
        if velocity.shape != position.shape:
            velocity = np.zeros_like(position)
        dim = self._arm_dim(arm_tag)
        for idx, (qpos, qvel) in enumerate(zip(position, velocity)):
            if self.runtime_execution_mode == "force_qpos":
                self.direct_set_arm(arm_tag, qpos[:dim], step=False)
                self.force_set_current_grippers()
                self.task.scene.step()
                self.direct_set_arm(arm_tag, qpos[:dim], step=False)
                self.force_set_current_grippers()
            else:
                self.task.robot.set_arm_joints(qpos[:dim], qvel[:dim], arm_tag)
                self.set_current_gripper_drive_targets()
                self.task.scene.step()
            if idx % self.record_every_path_step == 0 or idx == len(position) - 1:
                self.record_frame()
        return position[-1, :dim]

    def _execute_gripper_path(self, start_action, target_gripper, arm_tag="left"):
        start_action = np.asarray(start_action, dtype=np.float64)
        _, gripper_idx = self._arm_action_slice(arm_tag)
        start_gripper = float(start_action[gripper_idx])
        target_gripper = float(np.clip(target_gripper, 0.0, 1.0))
        if abs(target_gripper - start_gripper) < 1e-12:
            self.current_action = start_action.copy()
            if self.runtime_execution_mode == "force_qpos":
                self.force_set_action_qpos(start_action, step=True)
                self.force_set_action_qpos(start_action, step=False)
            else:
                self.task.robot.set_gripper(start_gripper, arm_tag, gripper_eps=0)
                self.task.scene.step()
            self.record_frame()
            return start_gripper, 1

        steps = self.gripper_interpolation_steps
        per_step = float((target_gripper - start_gripper) / steps)
        for gripper_value in np.linspace(start_gripper, target_gripper, steps + 1)[1:]:
            action = start_action.copy()
            action[gripper_idx] = float(gripper_value)
            self.current_action = action
            if self.runtime_execution_mode == "force_qpos":
                self.force_set_action_qpos(action, step=True)
                self.force_set_action_qpos(action, step=False)
            else:
                self.task.robot.set_gripper(float(gripper_value), arm_tag, gripper_eps=per_step)
                self.task.scene.step()
            self.record_frame()
        return target_gripper, steps

    def step(self, command):
        cmd = str(command.get("command", "")).lower()
        arm_tag = str(command.get("arm") or command.get("active_arm") or "left").lower()
        if arm_tag not in {"left", "right"}:
            raise ValueError(f"Unsupported Task3 arm: {arm_tag}")
        self.active_arm = arm_tag
        arm_slice, gripper_idx = self._arm_action_slice(arm_tag)
        if cmd in {"open", "close"}:
            action = self.current_action.copy()
            before_gripper = float(action[gripper_idx])
            sign = 1.0 if cmd == "open" else -1.0
            target_gripper = np.clip(before_gripper + sign * self.gripper_step, 0.0, 1.0)
            self.current_action = action
            after_gripper, path_steps = self._execute_gripper_path(action, target_gripper, arm_tag=arm_tag)
            action = self.current_action.copy()
            action[gripper_idx] = after_gripper
            self.current_action = action
            state = self._state(
                "step",
                command=cmd,
                arm=arm_tag,
                active_arm=arm_tag,
                active_tcp_pose=self.project_tcp(arm_tag)["tcp_pose"],
                planner=(
                    "gripper_physical_servo"
                    if self.runtime_execution_mode == "physical_drive_target"
                    else "gripper_delta_servo"
                ),
                execution_mode=self.runtime_execution_mode,
                gripper_delta=float(after_gripper - before_gripper),
                gripper_step=float(self.gripper_step),
                gripper_per_step=float((after_gripper - before_gripper) / max(1, int(path_steps))),
                active_gripper_state=self.gripper_runtime_state(arm_tag),
                path_steps=int(path_steps),
            )
            self._last_command_eval = None
            return state

        action = self.current_action.copy()
        fixed_cmds = {"up", "down", "left", "right", "lift", "delift"}
        rotation_cmds = {"roll_neg", "roll_pos", "pitch_pos", "pitch_neg", "yaw_neg", "yaw_pos"}
        if self.control_mode == "fixed_delta" and cmd in fixed_cmds:
            best, before = self._choose_fixed_delta_plan(cmd, arm_tag)
            plan = best.pop("plan")
            planner_name = "fixed_delta_joint_servo"
        elif cmd in {"up", "down", "left", "right"}:
            best, before = self._choose_joint_table_plan(cmd, arm_tag)
            plan = best.pop("plan")
            planner_name = "calibrated_joint_table_servo"
        elif cmd in {"lift", "delift"}:
            best, before = self._choose_joint_vertical_plan(cmd, arm_tag)
            plan = best.pop("plan")
            planner_name = "calibrated_joint_vertical_servo"
        elif cmd in rotation_cmds:
            best, before = self._choose_joint_rotation_plan(cmd, arm_tag)
            plan = best.pop("plan")
            planner_name = "calibrated_joint_orientation_servo"
        else:
            return super().step(command)

        action[arm_slice] = self._execute_arm_path(arm_tag, plan)
        self.current_action = action
        after = self.project_tcp(arm_tag)
        self._last_command_eval = {
            "before_projection": before,
            "after_projection": after,
            "selected": best,
        }
        return self._state(
            "step",
            command=cmd,
            arm=arm_tag,
            active_arm=arm_tag,
            **{f"{arm_tag}_tcp_pose": after["tcp_pose"]},
            active_tcp_pose=after["tcp_pose"],
            planner=planner_name,
            execution_mode=self.runtime_execution_mode,
            active_gripper_state=self.gripper_runtime_state(arm_tag),
            eef_delta_type=best.get("eef_delta_type"),
            target_pose=best.get("target_pose"),
            target_qpos=best.get("target_qpos"),
            plan_scale=best.get("scale"),
            plan_error=None,
            path_steps=int(np.asarray(plan["position"]).shape[0]),
            controller_eval=self._last_command_eval,
        )

    def hold(self, command, step_idx):
        for _ in range(self.hold_frames):
            if self.runtime_execution_mode == "force_qpos":
                self.force_set_action_qpos(self.current_action, step=True)
                self.force_set_action_qpos(self.current_action, step=False)
            else:
                action = np.asarray(self.current_action, dtype=np.float64)
                left = action[: self.left_dim]
                right_start = self.left_dim + 1
                right = action[right_start : right_start + self.right_dim]
                self.task.robot.set_arm_joints(left, np.zeros(self.left_dim), "left")
                self.task.robot.set_arm_joints(right, np.zeros(self.right_dim), "right")
                self.set_current_gripper_drive_targets()
                self.task.scene.step()
            self.record_frame(command, "hold", step_idx)


def load_initial_action(episode_id, frame_id, annotation_root, down_sample):
    ann_path = Path(annotation_root) / f"{episode_id}.json"
    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    action_seq = ann.get("robotwin_action") or ann.get("action.joint_position") or ann.get("action")
    action_seq = np.asarray(action_seq, dtype=np.float64)
    raw_id = min(int(frame_id) * int(down_sample), len(action_seq) - 1)
    return ann, action_seq[raw_id].copy(), raw_id, action_seq


def load_fixed_deltas(path):
    if path is None:
        return {}
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    deltas = {}
    for row in manifest.get("command_log", []):
        command = row.get("command")
        selected = ((row.get("controller_eval") or {}).get("selected") or {})
        q_delta = selected.get("q_delta")
        if command and q_delta is not None:
            deltas[str(command).lower()] = np.asarray(q_delta, dtype=np.float64)
    return deltas
