from __future__ import annotations

import importlib
import math
import os
from functools import lru_cache

import numpy as np
import transforms3d as t3d
import yaml


def _missing_robotwin_dependency(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        "RoboTwin runtime modules are unavailable. Launch the operator with an "
        "official RoboTwin checkout configured through --robotwin-root."
    )


@lru_cache(maxsize=1)
def _robotwin_runtime():
    try:
        envs_module = importlib.import_module("envs")
        test_render_module = importlib.import_module("test_render")
        contacts_module = importlib.import_module("script.annotate_official_contacts")
    except ModuleNotFoundError as exc:
        raise _missing_robotwin_dependency(exc) from exc
    return (
        envs_module.CONFIGS_PATH,
        test_render_module.Sapien_TEST,
        contacts_module.tolerate_official_seed_instability,
    )


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise RuntimeError(f"No such RoboTwin task: {task_name}") from exc
    return env_class()


def get_embodiment_config(robot_file):
    config_path = os.path.join(robot_file, "config.yml")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


def load_task_args(task_name, task_config, render_freq=0):
    configs_path, _, _ = _robotwin_runtime()
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    embodiment_type = args.get("embodiment")
    with open(
        os.path.join(configs_path, "_embodiment_config.yml"),
        "r",
        encoding="utf-8",
    ) as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def embodiment_file(name):
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise RuntimeError(f"Missing embodiment files for {name}")
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise RuntimeError("embodiment config should contain one or three items")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
    args["need_plan"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["render_freq"] = render_freq
    args["save_freq"] = None
    return args


def load_seed(save_path, episode):
    seed_path = os.path.join(save_path, "seed.txt")
    with open(seed_path, "r", encoding="utf-8") as handle:
        seeds = [int(item) for item in handle.read().split()]
    if episode >= len(seeds):
        raise IndexError(
            f"episode {episode} requested, but {seed_path} has {len(seeds)} seeds"
        )
    return seeds[episode]


def _unit(vec, fallback):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _project_xy(vec, fallback):
    arr = np.asarray(vec, dtype=np.float64).copy()
    arr[2] = 0.0
    return _unit(arr, fallback)


def _quat_roll(q, angle_rad):
    rot = t3d.quaternions.quat2mat(q)
    # RoboTwin TCP/tool forward axis is local +X in robot._trans_endpose().
    roll = t3d.axangles.axangle2mat([1.0, 0.0, 0.0], angle_rad)
    return t3d.quaternions.mat2quat(rot @ roll)


class LeftCameraTeleopAdapter:
    def __init__(self, task_config, planar_step, vertical_step, rotate_deg, lateral_step_mult):
        _, sapien_test, tolerate_seed_instability = _robotwin_runtime()
        sapien_test()
        tolerate_seed_instability()
        self.task_config = task_config
        self.planar_step = float(planar_step)
        self.vertical_step = float(vertical_step)
        self.lateral_step_mult = float(lateral_step_mult)
        self.rotate_rad = math.radians(float(rotate_deg))
        self.task = None
        self.task_name = None
        self.episode = None
        self.current_action = None
        self.left_dim = 6
        self.right_dim = 6

    def close(self):
        if self.task is not None:
            try:
                self.task.close_env(clear_cache=True)
            except Exception:
                pass
        self.task = None

    def reset(self, task_name, episode, action):
        self.close()
        args = load_task_args(task_name, self.task_config, render_freq=0)
        args["need_plan"] = True
        args["save_data"] = False
        args["collect_data"] = False
        args["render_freq"] = 0
        args["save_freq"] = None
        seed = load_seed(args["save_path"], int(episode))
        task = class_decorator(task_name)
        task.setup_demo(now_ep_num=int(episode), seed=int(seed), **args)
        self.task = task
        self.task_name = task_name
        self.episode = int(episode)
        self.current_action = np.asarray(action, dtype=np.float64).copy()
        self.left_dim = len(task.robot.left_arm_joints)
        self.right_dim = len(task.robot.right_arm_joints)
        self._apply_action(self.current_action)
        return self._state("reset", seed=seed)

    def _state(self, event, **extra):
        left_pose = self.task.robot.get_left_tcp_pose() if self.task is not None else None
        data = {
            "event": event,
            "task": self.task_name,
            "episode": self.episode,
            "action": self.current_action.tolist() if self.current_action is not None else None,
            "left_tcp_pose": left_pose,
        }
        data.update(extra)
        return data

    def _set_arm_qpos(self, entity, arm_joints, values):
        qpos = entity.get_qpos()
        active = list(entity.get_active_joints())
        for joint, value in zip(arm_joints, values):
            idx = active.index(joint)
            qpos[idx] = float(value)
        entity.set_qpos(qpos)

    def _apply_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        zero_left = np.zeros(self.left_dim)
        zero_right = np.zeros(self.right_dim)
        left = action[:self.left_dim]
        left_gripper = float(action[self.left_dim])
        right_start = self.left_dim + 1
        right = action[right_start : right_start + self.right_dim]
        right_gripper = float(action[right_start + self.right_dim])

        self._set_arm_qpos(self.task.robot.left_entity, self.task.robot.left_arm_joints, left)
        self._set_arm_qpos(self.task.robot.right_entity, self.task.robot.right_arm_joints, right)
        self.task.robot.set_arm_joints(left, zero_left, "left")
        self.task.robot.set_arm_joints(right, zero_right, "right")
        self.task.robot.set_gripper(left_gripper, "left", gripper_eps=0)
        self.task.robot.set_gripper(right_gripper, "right", gripper_eps=0)
        self.task.scene.step()

    def _head_camera_axes(self):
        cfg = self.task.cameras.get_config()
        head = cfg.get("head_camera")
        if head is None:
            raise RuntimeError("head_camera config is unavailable")
        mat = np.asarray(head["cam2world_gl"], dtype=np.float64)
        cam_forward = _unit(mat[:3, 0], [1, 0, 0])
        cam_left = _unit(mat[:3, 1], [0, 1, 0])
        cam_up = _unit(mat[:3, 2], [0, 0, 1])
        screen_left = _project_xy(cam_left, [0, 1, 0])
        # RoboTwin's OpenGL camera matrix reports the displayed vertical axis
        # with the opposite sign from screen-space "up".
        screen_up = -_project_xy(cam_up, -cam_forward)
        return {
            "left": screen_left,
            "right": -screen_left,
            "up": screen_up,
            "down": -screen_up,
            "lift": np.array([0.0, 0.0, 1.0]),
            "delift": np.array([0.0, 0.0, -1.0]),
        }

    def _plan_left_path(self, target_pose):
        result = self.task.robot.left_plan_path(target_pose)
        if not isinstance(result, dict) or result.get("status") != "Success":
            raise RuntimeError(f"left_plan_path failed: {result}")
        position = np.asarray(result["position"], dtype=np.float64)
        if position.ndim != 2 or position.shape[0] == 0:
            raise RuntimeError(f"left_plan_path returned bad position shape: {position.shape}")
        result["position"] = position
        if "velocity" in result:
            result["velocity"] = np.asarray(result["velocity"], dtype=np.float64)
        return result

    def _execute_left_path(self, result):
        position = np.asarray(result["position"], dtype=np.float64)
        velocity = np.asarray(result.get("velocity", np.zeros_like(position)), dtype=np.float64)
        if velocity.shape != position.shape:
            velocity = np.zeros_like(position)
        for qpos, qvel in zip(position, velocity):
            self.task.robot.set_arm_joints(
                qpos[: self.left_dim],
                qvel[: self.left_dim],
                "left",
            )
            self.task.scene.step()
        return position[-1, : self.left_dim]

    def _plan_left_delta_with_retries(self, pose, delta):
        last_error = None
        for scale in [1.0, 0.5, 0.25, 0.125]:
            target_pose = pose.copy()
            target_pose[:3] = pose[:3] + delta * scale
            try:
                return self._plan_left_path(target_pose.tolist()), target_pose, scale, None
            except Exception as exc:
                last_error = repr(exc)
        return None, pose.copy(), 0.0, last_error

    def step(self, command):
        if self.task is None or self.current_action is None:
            raise RuntimeError("Adapter has not been reset")

        cmd = str(command.get("command", "")).lower()
        action = self.current_action.copy()
        pose = np.asarray(self.task.robot.get_left_tcp_pose(), dtype=np.float64)
        target_pose = pose.copy()

        if cmd in {"open", "close"}:
            action[self.left_dim] = 1.0 if cmd == "open" else 0.0
            self.current_action = action
            self._apply_action(action)
            return self._state("step", command=cmd, planner="gripper")

        plan_scale = 1.0
        plan_error = None
        if cmd in {"left", "right", "up", "down", "lift", "delift"}:
            step = self.vertical_step if cmd in {"lift", "delift"} else self.planar_step
            if cmd in {"left", "right"}:
                step *= self.lateral_step_mult
            plan_result, target_pose, plan_scale, plan_error = self._plan_left_delta_with_retries(
                pose,
                self._head_camera_axes()[cmd] * step,
            )
            if plan_result is None:
                action[: self.left_dim] = self.current_action[: self.left_dim]
                path_steps = 0
            else:
                action[: self.left_dim] = self._execute_left_path(plan_result)
                path_steps = int(np.asarray(plan_result["position"]).shape[0])
        elif cmd in {"rotate_cw", "rotate_ccw"}:
            sign = -1.0 if cmd == "rotate_cw" else 1.0
            target_pose[3:7] = _quat_roll(pose[3:7], sign * self.rotate_rad)
            plan_result = self._plan_left_path(target_pose.tolist())
            action[: self.left_dim] = self._execute_left_path(plan_result)
            path_steps = int(np.asarray(plan_result["position"]).shape[0])
        else:
            raise ValueError(f"Unsupported left teleop command: {cmd}")

        self.current_action = action
        return self._state(
            "step",
            command=cmd,
            planner="left_plan_path",
            target_pose=target_pose.tolist(),
            plan_scale=plan_scale,
            plan_error=plan_error,
            path_steps=path_steps,
        )
