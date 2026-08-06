from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import numpy as np

from worldsimprobe.evaluation.task3_action_coverage.teleoperation.session import (
    DEFAULT_COMMANDS,
    Task3SessionConfig,
    Task3TeleopSession,
)
from worldsimprobe.submission.video_config import default_video_fps


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
KEY_BINDINGS = {
    "ArrowUp": "up",
    "KeyW": "up",
    "w": "up",
    "W": "up",
    "ArrowDown": "down",
    "KeyS": "down",
    "s": "down",
    "S": "down",
    "ArrowLeft": "left",
    "KeyA": "left",
    "a": "left",
    "A": "left",
    "ArrowRight": "right",
    "KeyD": "right",
    "d": "right",
    "D": "right",
    "KeyQ": "lift",
    "q": "lift",
    "Q": "lift",
    "KeyE": "delift",
    "e": "delift",
    "E": "delift",
    "KeyU": "roll_neg",
    "u": "roll_neg",
    "U": "roll_neg",
    "KeyO": "roll_pos",
    "o": "roll_pos",
    "O": "roll_pos",
    "KeyI": "pitch_pos",
    "i": "pitch_pos",
    "I": "pitch_pos",
    "KeyK": "pitch_neg",
    "k": "pitch_neg",
    "K": "pitch_neg",
    "KeyJ": "yaw_neg",
    "j": "yaw_neg",
    "J": "yaw_neg",
    "KeyL": "yaw_pos",
    "l": "yaw_pos",
    "L": "yaw_pos",
    "KeyZ": "close",
    "z": "close",
    "Z": "close",
    "KeyX": "open",
    "x": "open",
    "X": "open",
}
SELECTED_SCENE_TASKS = (
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_two",
    "place_empty_cup",
    "place_phone_stand",
    "place_mouse_pad",
    "move_pillbottle_pad",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "hanging_mug",
    "click_bell",
)
SCENE_EPISODES = tuple(range(50))
SCENE_EPISODES_PER_TASK = 5
REFERENCE_VIDEO_FPS = 30.0
EPISODE_ID_RE = re.compile(r"^(?P<task>.+)__(?P<episode>\d{6})$")
TASK3_MODEL_VIDEO_RE = re.compile(
    r"^/api/task3/model-video/(?P<video_id>[a-f0-9]{16})$"
)
TASK3_MODELS = {
    "lingbot-va": "LingBot-VA",
    "ctrl-world": "Ctrl-World",
}
TASK1_JOB_RE = re.compile(
    r"^/api/task1/jobs/(?P<job_id>[a-f0-9]{12})(?:/video/(?P<variant>original|small|large))?$"
)
TASK1_DEMOS: dict[str, dict[str, Any]] = {
    "adjust_bottle__000000": {
        "task": "adjust_bottle",
        "task_label": "Adjust Bottle",
        "episode": 0,
        "window": (42, 72),
        "perturb_dim": "auto",
        "magnitude": {
            "min": 0.02,
            "max": 0.18,
            "step": 0.005,
            "default": 0.14,
            "small_ratio": 0.25,
        },
    },
    "place_bread_skillet__000044": {
        "task": "place_bread_skillet",
        "task_label": "Place Bread in Skillet",
        "episode": 44,
        "window": (13, 16),
        "perturb_dim": 8,
        "magnitude": {
            "min": 0.005,
            "max": 0.04,
            "step": 0.001,
            "default": 0.02,
            "small_ratio": 0.25,
        },
    },
    "open_microwave__000006": {
        "task": "open_microwave",
        "task_label": "Open Microwave",
        "episode": 6,
        "window": (15, 18),
        "perturb_dim": 1,
        "magnitude": {
            "min": 0.005,
            "max": 0.04,
            "step": 0.001,
            "default": 0.02,
            "small_ratio": 0.25,
        },
    },
    "shake_bottle__000000": {
        "task": "shake_bottle",
        "task_label": "Shake Bottle",
        "episode": 0,
        "window": (18, 21),
        "perturb_dim": 1,
        "magnitude": {
            "min": 0.005,
            "max": 0.04,
            "step": 0.001,
            "default": 0.02,
            "small_ratio": 0.25,
        },
    },
}
TASK1_DEFAULT_DEMO_ID = "adjust_bottle__000000"
TASK1_VARIANTS = ("original", "small", "large")
TASK1_CACHE_VERSION = 2


class Task1BusyError(RuntimeError):
    pass


def resolve_task1_demo(demo_id: Any) -> tuple[str, dict[str, Any]]:
    key = TASK1_DEFAULT_DEMO_ID if demo_id is None else str(demo_id)
    demo = TASK1_DEMOS.get(key)
    if demo is None:
        raise ValueError("Unknown Task 1 demo.")
    return key, demo


def validate_task1_magnitude(value: Any, demo: dict[str, Any]) -> float:
    try:
        magnitude = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Perturbation magnitude must be numeric.") from exc
    if not np.isfinite(magnitude):
        raise ValueError("Perturbation magnitude must be finite.")
    magnitude_config = demo["magnitude"]
    minimum = float(magnitude_config["min"])
    maximum = float(magnitude_config["max"])
    if not minimum <= magnitude <= maximum:
        raise ValueError(
            f"Perturbation magnitude must be between {minimum:g} "
            f"and {maximum:g}."
        )
    return round(magnitude, 4)


def task1_small_magnitude(demo: dict[str, Any]) -> float:
    magnitude_config = demo["magnitude"]
    return round(
        float(magnitude_config["default"]) * float(magnitude_config["small_ratio"]),
        4,
    )


def static_root() -> Path:
    return Path(__file__).resolve().parent / "static"


def encode_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode returned false")
        return bytes(buf)
    except Exception:
        from PIL import Image
        from io import BytesIO

        out = BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(out, format="JPEG", quality=quality)
        return out.getvalue()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def episode_id_for(task: str, episode: int) -> str:
    return f"{task}__{int(episode):06d}"


def parse_episode_id(episode_id: str) -> tuple[str, int] | None:
    match = EPISODE_ID_RE.match(str(episode_id))
    if not match:
        return None
    return match.group("task"), int(match.group("episode"))


def validate_scene_selection(task: str, episode: int) -> tuple[str, int]:
    episode = int(episode)
    if task not in SELECTED_SCENE_TASKS or episode not in SCENE_EPISODES:
        raise ValueError(f"Scene is not available in the Task 3 operator: {task} episode {episode}")
    return task, episode


def validate_task3_model(model_id: Any) -> str:
    model = str(model_id or "").strip().lower()
    if model not in TASK3_MODELS:
        raise ValueError("Select LingBot-VA or Ctrl-World before sending a command.")
    return model


def normalize_model_endpoint(raw: Any) -> str | None:
    value = str(raw or "").strip().rstrip("/")
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid Task 3 model endpoint: {value}")
    return value


def first_instruction(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("seen", "unseen"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
    for value in data.values():
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
    return ""


def read_ws_frame(rfile) -> tuple[int, bytes] | None:
    header = rfile.read(2)
    if not header or len(header) < 2:
        return None
    first, second = header
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", rfile.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", rfile.read(8))[0]
    mask = rfile.read(4) if masked else b""
    payload = rfile.read(length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
    return opcode, payload


def make_ws_frame(opcode: int, payload: bytes) -> bytes:
    size = len(payload)
    header = bytearray([0x80 | opcode])
    if size < 126:
        header.append(size)
    elif size < (1 << 16):
        header.extend([126])
        header.extend(struct.pack("!H", size))
    else:
        header.extend([127])
        header.extend(struct.pack("!Q", size))
    return bytes(header) + payload


class OperatorRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.public_demo = bool(getattr(args, "public_demo", False))
        self.output_root = Path(args.output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task3-sim")
        self.task1_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task1-sim")
        self.session: Task3TeleopSession | None = None
        self.model_endpoints = {
            "lingbot-va": normalize_model_endpoint(
                getattr(args, "task3_lingbot_endpoint", None)
            ),
            "ctrl-world": normalize_model_endpoint(
                getattr(args, "task3_ctrlworld_endpoint", None)
            ),
        }
        self.model_timeout = float(getattr(args, "task3_model_timeout", 900.0))
        self._local_http_opener = build_opener(ProxyHandler({}))
        self._model_initial_jpeg: bytes | None = None
        self._model_videos: dict[str, dict[str, Any]] = {}
        self._model_lock = threading.Lock()
        self.task1_output_root = self.output_root / "task1_live"
        self.task1_output_root.mkdir(parents=True, exist_ok=True)
        self.task1_cache_root = self.task1_output_root / "cache"
        self.task1_cache_root.mkdir(parents=True, exist_ok=True)
        if self.public_demo:
            self._clear_task1_transient_outputs()
        self._task1_jobs: dict[str, dict[str, Any]] = {}
        self._task1_lock = threading.Lock()
        self._task1_running = False
        self._state_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._scene_inventory_lock = threading.Lock()
        self._scene_inventory_cache: dict[str, Any] | None = None
        self._active_client: object | None = None
        self._prewarm_lock = threading.Lock()
        self._prewarm_future = None

    def _clear_task1_transient_outputs(self) -> None:
        for target in self.task1_output_root.iterdir():
            if target == self.task1_cache_root:
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)

    def public_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.public_demo:
            return payload
        if payload.get("type") == "error":
            safe_messages = {
                "Artifact export is disabled.",
                "Artifact operations are disabled in public demo mode.",
                "The live simulator is currently in use.",
                "The simulator is already running another live probe.",
                "Another Task 1 generation is already running.",
                "Trace evaluation is disabled.",
            }
            if payload.get("message") in safe_messages:
                return payload
            return {"type": "error", "message": "The simulator session could not complete the request."}
        hidden_keys = {
            "annotation",
            "annotation_root",
            "dataset_row",
            "first_frame",
            "fixed_delta_source",
            "manifest",
            "output_dir",
            "raw_video",
            "reference_video",
            "reference_video_path",
            "reference_video_url",
            "robotwin_root",
            "sample",
            "trace",
            "video",
        }

        def scrub(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: scrub(item)
                    for key, item in value.items()
                    if key not in hidden_keys
                }
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value

        return scrub(payload)

    def acquire_client(self, token: object) -> bool:
        if not self.public_demo:
            return True
        with self._client_lock:
            if self._active_client is not None:
                return False
            self._active_client = token
            return True

    def _remove_session_output(self, output_dir: Path) -> None:
        if not self.public_demo:
            return
        target = output_dir.expanduser().resolve()
        try:
            target.relative_to(self.output_root)
        except ValueError:
            return
        shutil.rmtree(target, ignore_errors=True)

    def release_client(self, token: object) -> None:
        if not self.public_demo:
            return
        with self._client_lock:
            if self._active_client is not token:
                return

        def park_or_close_session() -> bool:
            with self._state_lock:
                session = self.session
            if session is None:
                return False
            if bool(getattr(self.args, "prewarm_default", False)) and self._session_matches(
                session,
                str(self.args.episode_id),
                int(self.args.frame_id),
                require_pristine=True,
            ):
                session.set_frame_callback(None)
                return True
            with self._state_lock:
                if self.session is session:
                    self.session = None
            output_dir = session.output_dir
            try:
                session.close()
            finally:
                self._remove_session_output(output_dir)
            return False

        parked = False
        try:
            parked = bool(self.sim_call(park_or_close_session))
        finally:
            self.clear_model_video()
            self._model_initial_jpeg = None
            with self._client_lock:
                if self._active_client is token:
                    self._active_client = None
        if not parked:
            self.prewarm_default_session()

    def annotation_root(self) -> Path:
        return Path(self.args.annotation_root).expanduser().resolve()

    def robotwin_scene_dir(self, task: str) -> Path:
        return Path(self.args.robotwin_root).expanduser().resolve() / "data" / task / str(self.args.task_config)

    def scene_source_paths(self, task: str, episode: int) -> dict[str, Path]:
        scene_dir = self.robotwin_scene_dir(task)
        stem = f"episode{int(episode)}"
        return {
            "hdf5": scene_dir / "data" / f"{stem}.hdf5",
            "instruction": scene_dir / "instructions" / f"{stem}.json",
            "video": scene_dir / "video" / f"{stem}.mp4",
        }

    def load_scene_action_count(self, task: str, episode: int) -> int:
        import h5py

        with h5py.File(self.scene_source_paths(task, episode)["hdf5"], "r") as handle:
            return int(handle["joint_action/vector"].shape[0])

    def load_reference_video_info(self, task: str, episode: int, fallback_count: int = 0) -> dict[str, Any]:
        video_path = self.scene_source_paths(task, episode)["video"]
        frame_count = int(fallback_count)
        fps = float(REFERENCE_VIDEO_FPS)
        if video_path.exists():
            try:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                observed_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                observed_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                cap.release()
                if observed_count > 0:
                    frame_count = observed_count
                if observed_fps > 0:
                    fps = observed_fps
            except Exception:
                pass
        return {
            "reference_video_path": str(video_path),
            "reference_video_url": "/reference-video?"
            + urlencode({"task": task, "episode": int(episode)}),
            "reference_fps": fps,
            "reference_frame_count": frame_count,
        }

    def resolve_reference_video(self, task: str, episode: int) -> Path:
        try:
            task, episode = validate_scene_selection(task, episode)
        except ValueError as exc:
            raise PermissionError(str(exc)) from exc
        path = self.scene_source_paths(task, int(episode))["video"].resolve()
        root = Path(self.args.robotwin_root).expanduser().resolve() / "data" / task / str(self.args.task_config) / "video"
        path.relative_to(root.resolve())
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def default_frame_for(self, action_count: int) -> int:
        return 0

    def _build_scene_inventory(self) -> dict[str, Any]:
        default_parsed = parse_episode_id(self.args.episode_id)
        default_task = default_parsed[0] if default_parsed else SELECTED_SCENE_TASKS[0]
        default_episode = default_parsed[1] if default_parsed else SCENE_EPISODES[0]
        tasks = []
        for task in SELECTED_SCENE_TASKS:
            episodes = []
            for episode in SCENE_EPISODES:
                source = self.scene_source_paths(task, episode)
                episode_id = episode_id_for(task, episode)
                exists = all(path.exists() for path in source.values())
                if not exists:
                    continue
                if self.public_demo:
                    episodes.append(
                        {
                            "episode": int(episode),
                            "episode_id": episode_id,
                            "default_frame": self.default_frame_for(1),
                            "instruction": first_instruction(source["instruction"])
                            if source["instruction"].exists() else "",
                            "data_exists": True,
                        }
                    )
                    if len(episodes) >= SCENE_EPISODES_PER_TASK:
                        break
                    continue
                annotation_path = self.annotation_root() / f"{episode_id}.json"
                action_count = self.load_scene_action_count(task, episode)
                reference_info = self.load_reference_video_info(task, episode, action_count) if source["video"].exists() else {}
                episodes.append(
                    {
                        "episode": int(episode),
                        "episode_id": episode_id,
                        "default_frame": self.default_frame_for(action_count) if action_count else int(self.args.frame_id),
                        "action_count": action_count,
                        **reference_info,
                        "instruction": first_instruction(source["instruction"]) if source["instruction"].exists() else "",
                        "annotation_exists": annotation_path.exists(),
                        "data_exists": True,
                    }
                )
                if len(episodes) >= SCENE_EPISODES_PER_TASK:
                    break
            tasks.append({"task": task, "label": task.replace("_", " "), "episodes": episodes})
        return {
            "tasks": tasks,
            "episodes": list(SCENE_EPISODES),
            "default_task": default_task,
            "default_episode": int(default_episode),
            "annotation_root": str(self.annotation_root()),
            "robotwin_root": str(Path(self.args.robotwin_root).expanduser().resolve()),
            "down_sample": int(self.args.down_sample),
            "reference_fps": float(REFERENCE_VIDEO_FPS),
        }

    def scene_inventory(self) -> dict[str, Any]:
        with self._scene_inventory_lock:
            if self._scene_inventory_cache is None:
                self._scene_inventory_cache = self._build_scene_inventory()
            return self._scene_inventory_cache

    def _model_health(self, model_id: str, timeout: float = 1.5) -> dict[str, Any]:
        model = validate_task3_model(model_id)
        endpoint = self.model_endpoints.get(model)
        if endpoint is None:
            return {"available": False, "state": "unconfigured"}
        try:
            with self._local_http_opener.open(f"{endpoint}/health", timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ready = response.status == 200 and payload.get("state") == "ready"
            return {
                "available": bool(ready),
                "state": str(payload.get("state") or ("ready" if ready else "offline")),
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
            return {"available": False, "state": "offline"}

    def task3_model_config(self) -> dict[str, Any]:
        models = []
        for model_id, label in TASK3_MODELS.items():
            health = self._model_health(model_id)
            models.append({"id": model_id, "label": label, **health})
        return {"models": models}

    def require_task3_model(self, model_id: Any) -> str:
        model = validate_task3_model(model_id)
        health = self._model_health(model, timeout=2.0)
        if not health["available"]:
            raise RuntimeError(f"{TASK3_MODELS[model]} is not ready.")
        return model

    def clear_model_video(self) -> None:
        with self._model_lock:
            self._model_videos.clear()

    def resolve_model_video(self, video_id: str) -> tuple[bytes, str]:
        with self._model_lock:
            record = self._model_videos.get(str(video_id))
            if record is None:
                raise FileNotFoundError(video_id)
            return bytes(record["data"]), str(record["content_type"])

    def _model_snapshot(self, client_session_id: Any) -> dict[str, Any]:
        def snapshot() -> dict[str, Any]:
            if self.session is None:
                raise RuntimeError("No active Task 3 session")
            if self._model_initial_jpeg is None:
                raise RuntimeError("The model start frame is unavailable.")
            manifest = self.session.build_manifest()
            return {
                "session_id": str(client_session_id or manifest.get("episode_id") or "task3"),
                "episode_id": manifest.get("episode_id"),
                "instruction": manifest.get("instruction") or manifest.get("task") or "",
                "initial_frame_jpeg": base64.b64encode(self._model_initial_jpeg).decode("ascii"),
                "trace": manifest,
            }

        return self.sim_call(snapshot)

    def infer_task3_model(self, model_id: Any, client_session_id: Any) -> dict[str, Any]:
        model = self.require_task3_model(model_id)
        endpoint = self.model_endpoints[model]
        if endpoint is None:
            raise RuntimeError(f"{TASK3_MODELS[model]} is not configured.")
        payload = self._model_snapshot(client_session_id)
        request = Request(
            f"{endpoint}/infer",
            data=json_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._local_http_opener.open(request, timeout=self.model_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"{TASK3_MODELS[model]} rejected the action.") from exc
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"{TASK3_MODELS[model]} did not complete inference.") from exc
        try:
            video = base64.b64decode(result.pop("video_base64"), validate=True)
        except Exception as exc:
            raise RuntimeError(f"{TASK3_MODELS[model]} returned an invalid video.") from exc
        if len(video) < 32 or b"ftyp" not in video[:32]:
            raise RuntimeError(f"{TASK3_MODELS[model]} returned an invalid video.")
        video_id = uuid.uuid4().hex[:16]
        with self._model_lock:
            self._model_videos[video_id] = {
                "data": video,
                "content_type": "video/mp4",
                "model": model,
                "created_at": time.time(),
            }
            while len(self._model_videos) > 2:
                self._model_videos.pop(next(iter(self._model_videos)))
        return {
            "type": "model_result",
            "model": model,
            "model_label": TASK3_MODELS[model],
            "video_url": f"/api/task3/model-video/{video_id}",
            "command_count": int(result.get("command_count", 0)),
            "frame_count": int(result.get("frame_count", 0)),
            "segment_frame_count": int(result.get("segment_frame_count", 0)),
            "continued": bool(result.get("continued", False)),
            "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
            "inference_mode": str(result.get("inference_mode") or "incremental"),
        }

    def ensure_scene_annotation(self, task: str, episode: int) -> dict[str, Any]:
        episode_id = episode_id_for(task, episode)
        annotation_path = self.annotation_root() / f"{episode_id}.json"
        if annotation_path.exists():
            return {"episode_id": episode_id, "annotation": str(annotation_path), "created": False}
        source = self.scene_source_paths(task, episode)
        missing = [str(path) for path in source.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing RoboTwin scene source files for {episode_id}: {missing}")

        import h5py

        with h5py.File(source["hdf5"], "r") as handle:
            actions = np.asarray(handle["joint_action/vector"][()], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected 14D joint_action/vector for {episode_id}, got {actions.shape}")
        action_list = actions.tolist()
        annotation = {
            "episode_id": episode_id,
            "task": task,
            "source_episode_index": int(episode),
            "texts": [first_instruction(source["instruction"])],
            "action": action_list,
            "robotwin_action": action_list,
            "action.joint_position": action_list,
            "videos": [
                {
                    "camera": "head_camera",
                    "video_path": str(source["video"]),
                }
            ],
        }
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = annotation_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(annotation_path)
        with self._scene_inventory_lock:
            self._scene_inventory_cache = None
        return {"episode_id": episode_id, "annotation": str(annotation_path), "created": True}

    def task1_generator_path(self) -> Path | None:
        raw = getattr(self.args, "task1_generator_script", None)
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.exists() and path.is_file() else None

    def _task1_demo_payload(self, demo_id: str, demo: dict[str, Any]) -> dict[str, Any]:
        source = self.scene_source_paths(str(demo["task"]), int(demo["episode"]))
        available = self.task1_generator_path() is not None and all(path.exists() for path in source.values())
        return {
            "id": demo_id,
            "available": available,
            "task": demo["task"],
            "task_label": demo["task_label"],
            "episode": demo["episode"],
            "instruction": first_instruction(source["instruction"]) if source["instruction"].exists() else "",
            "magnitude": {
                **demo["magnitude"],
                "small": task1_small_magnitude(demo),
            },
        }

    def task1_demo_config(self) -> dict[str, Any]:
        demos = [
            self._task1_demo_payload(demo_id, demo)
            for demo_id, demo in TASK1_DEMOS.items()
        ]
        default = next(item for item in demos if item["id"] == TASK1_DEFAULT_DEMO_ID)
        return {
            **default,
            "available": any(bool(item["available"]) for item in demos),
            "default_demo_id": TASK1_DEFAULT_DEMO_ID,
            "demos": demos,
        }

    def _task1_cache_spec(
        self,
        demo_id: str,
        demo: dict[str, Any],
        magnitude: float,
    ) -> dict[str, Any]:
        return {
            "version": TASK1_CACHE_VERSION,
            "demo_id": demo_id,
            "task": demo["task"],
            "task_config": str(self.args.task_config),
            "episode": demo["episode"],
            "window": list(demo["window"]),
            "perturb_dim": demo["perturb_dim"],
            "small_magnitude": task1_small_magnitude(demo),
            "large_magnitude": round(float(magnitude), 4),
            "steps_per_action": 3,
            "fps": 10,
        }

    def _task1_cache_dir(self, demo_id: str, magnitude: float) -> Path:
        return self.task1_cache_root / demo_id / f"{float(magnitude):.4f}"

    def _load_task1_cache(
        self,
        demo_id: str,
        demo: dict[str, Any],
        magnitude: float,
    ) -> tuple[Path, dict[str, Any]] | None:
        cache_dir = self._task1_cache_dir(demo_id, magnitude)
        manifest_path = cache_dir / "cache.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if payload.get("spec") != self._task1_cache_spec(demo_id, demo, magnitude):
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        if not all((cache_dir / "web" / f"{variant}.mp4").is_file() for variant in TASK1_VARIANTS):
            return None
        return cache_dir, result

    def _store_task1_cache(
        self,
        demo_id: str,
        demo: dict[str, Any],
        magnitude: float,
        result: dict[str, Any],
        output_dir: Path,
    ) -> Path:
        cache_dir = self._task1_cache_dir(demo_id, magnitude)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = cache_dir.parent / f".{cache_dir.name}.{uuid.uuid4().hex[:8]}.tmp"
        shutil.rmtree(staging_dir, ignore_errors=True)
        web_dir = staging_dir / "web"
        web_dir.mkdir(parents=True, exist_ok=False)
        for variant in TASK1_VARIANTS:
            shutil.copy2(output_dir / "web" / f"{variant}.mp4", web_dir / f"{variant}.mp4")
        (staging_dir / "cache.json").write_text(
            json.dumps(
                {"spec": self._task1_cache_spec(demo_id, demo, magnitude), "result": result},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        staging_dir.replace(cache_dir)
        return cache_dir

    def _remove_task1_output(self, output_dir: Path) -> None:
        target = output_dir.expanduser().resolve()
        try:
            target.relative_to(self.task1_cache_root.resolve())
            return
        except ValueError:
            pass
        try:
            target.relative_to(self.task1_output_root.resolve())
        except ValueError:
            return
        shutil.rmtree(target, ignore_errors=True)

    @staticmethod
    def _video_mse(first_path: Path, second_path: Path) -> float:
        import cv2

        first = cv2.VideoCapture(str(first_path))
        second = cv2.VideoCapture(str(second_path))
        squared_error = 0.0
        value_count = 0
        try:
            while True:
                ok_first, frame_first = first.read()
                ok_second, frame_second = second.read()
                if not ok_first or not ok_second:
                    break
                if frame_first.shape != frame_second.shape:
                    frame_second = cv2.resize(
                        frame_second,
                        (frame_first.shape[1], frame_first.shape[0]),
                    )
                delta = frame_first.astype(np.float64) - frame_second.astype(np.float64)
                squared_error += float(np.square(delta).sum())
                value_count += int(delta.size)
        finally:
            first.release()
            second.release()
        if value_count == 0:
            raise RuntimeError("Generated Task 1 videos contain no aligned frames.")
        return squared_error / value_count

    @staticmethod
    def _transcode_task1_video(source: Path, target: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            shutil.copy2(source, target)
            return
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if completed.returncode != 0:
            raise RuntimeError("Could not prepare a browser-compatible Task 1 video.")

    def _task1_public_status(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "type": "task1_job",
            "job_id": job["job_id"],
            "state": job["state"],
            "demo_id": job["demo_id"],
            "task": job["task"],
            "task_label": job["task_label"],
            "episode": job["episode"],
            "instruction": job["instruction"],
            "small_magnitude": job["small_magnitude"],
            "large_magnitude": job["large_magnitude"],
            "created_at": job["created_at"],
            "cache_hit": bool(job.get("cache_hit", False)),
            "generated_variants": list(job.get("generated_variants", TASK1_VARIANTS)),
        }
        if job["state"] == "ready":
            payload["result"] = job["result"]
            payload["videos"] = {
                variant: f"/api/task1/jobs/{job['job_id']}/video/{variant}"
                for variant in TASK1_VARIANTS
            }
        elif job["state"] == "error":
            payload["message"] = "The Task 1 rollout could not be generated."
        return payload

    def task1_job_status(self, job_id: str) -> dict[str, Any]:
        with self._task1_lock:
            job = self._task1_jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            snapshot = dict(job)
        return self._task1_public_status(snapshot)

    def resolve_task1_video(self, job_id: str, variant: str) -> Path:
        if variant not in TASK1_VARIANTS:
            raise FileNotFoundError(variant)
        with self._task1_lock:
            job = self._task1_jobs.get(job_id)
            if job is None or job.get("state") != "ready":
                raise FileNotFoundError(job_id)
            output_dir = Path(job["output_dir"]).expanduser().resolve()
        output_dir.relative_to(self.task1_output_root.resolve())
        target = (output_dir / "web" / f"{variant}.mp4").resolve()
        target.relative_to(output_dir)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(str(target))
        return target

    def start_task1_job(self, magnitude_value: Any, demo_id_value: Any = None) -> dict[str, Any]:
        generator = self.task1_generator_path()
        if generator is None:
            raise RuntimeError("Task 1 live generation is not configured.")
        demo_id, demo = resolve_task1_demo(demo_id_value)
        magnitude = validate_task1_magnitude(magnitude_value, demo)
        small_magnitude = task1_small_magnitude(demo)
        config = self._task1_demo_payload(demo_id, demo)
        if not config["available"]:
            raise RuntimeError("Task 1 demo assets are unavailable.")

        cached = self._load_task1_cache(demo_id, demo, magnitude)
        if cached is not None:
            cache_dir, result = cached
            job_id = uuid.uuid4().hex[:12]
            job = {
                "job_id": job_id,
                "state": "ready",
                "demo_id": demo_id,
                "task": demo["task"],
                "task_label": config["task_label"],
                "episode": demo["episode"],
                "instruction": config["instruction"],
                "small_magnitude": small_magnitude,
                "large_magnitude": magnitude,
                "created_at": time.time(),
                "output_dir": str(cache_dir),
                "result": result,
                "cache_hit": True,
                "generated_variants": [],
            }
            with self._task1_lock:
                stale = list(self._task1_jobs.values())
                self._task1_jobs = {job_id: job}
            for previous in stale:
                if not previous.get("cache_hit", False):
                    self._remove_task1_output(Path(previous["output_dir"]))
            return self._task1_public_status(dict(job))

        default_magnitude = round(float(demo["magnitude"]["default"]), 4)
        baseline = self._load_task1_cache(demo_id, demo, default_magnitude)
        large_only = baseline is not None and magnitude != default_magnitude

        with self._client_lock:
            if self._task1_running:
                raise Task1BusyError("Another Task 1 generation is already running.")
            self._task1_running = True

        job_id = uuid.uuid4().hex[:12]
        output_dir = self.task1_output_root / job_id
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except Exception:
            with self._client_lock:
                self._task1_running = False
            raise
        job = {
            "job_id": job_id,
            "state": "queued",
            "demo_id": demo_id,
            "task": demo["task"],
            "task_label": config["task_label"],
            "episode": demo["episode"],
            "instruction": config["instruction"],
            "small_magnitude": small_magnitude,
            "large_magnitude": magnitude,
            "created_at": time.time(),
            "output_dir": str(output_dir),
            "cache_hit": False,
            "generated_variants": ["large"] if large_only else list(TASK1_VARIANTS),
            "baseline_dir": str(baseline[0]) if large_only else None,
            "baseline_result": baseline[1] if large_only else None,
        }
        with self._task1_lock:
            stale = list(self._task1_jobs.values())
            self._task1_jobs = {job_id: job}
        for previous in stale:
            if not previous.get("cache_hit", False):
                self._remove_task1_output(Path(previous["output_dir"]))

        try:
            self.task1_executor.submit(self._run_task1_job, job_id, generator)
        except Exception:
            with self._client_lock:
                self._task1_running = False
            self._remove_task1_output(output_dir)
            raise
        return self._task1_public_status(dict(job))

    def _run_task1_job(self, job_id: str, generator: Path) -> None:
        with self._task1_lock:
            job = self._task1_jobs[job_id]
            job["state"] = "running"
            output_dir = Path(job["output_dir"])
            demo_id = str(job["demo_id"])
            demo = TASK1_DEMOS[demo_id]
            small_magnitude = float(job["small_magnitude"])
            large_magnitude = float(job["large_magnitude"])
            generated_variants = tuple(job.get("generated_variants", TASK1_VARIANTS))
            baseline_dir = Path(job["baseline_dir"]) if job.get("baseline_dir") else None
            baseline_result = job.get("baseline_result") or {}

        command = [
            sys.executable,
            str(generator),
            "--robotwin-root",
            str(Path(self.args.robotwin_root).expanduser().resolve()),
            "--out-dir",
            str(output_dir),
            "--task",
            str(demo["task"]),
            "--task-config",
            str(self.args.task_config),
            "--episode",
            str(demo["episode"]),
            "--window-start",
            str(demo["window"][0]),
            "--window-end",
            str(demo["window"][1]),
            "--small-magnitude",
            str(small_magnitude),
            "--large-magnitude",
            str(large_magnitude),
            "--arm",
            "auto",
            "--perturb-dim",
            str(demo["perturb_dim"]),
            "--steps-per-action",
            "3",
            "--fps",
            "10",
            "--variants",
            *generated_variants,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(self.args.task1_timeout),
                cwd=str(generator.parent),
            )
            if completed.returncode != 0:
                raise RuntimeError("Task 1 generator exited unsuccessfully.")
            summary = json.loads(completed.stdout)
            raw_video_dir = output_dir / "video"
            raw_videos = {
                variant: raw_video_dir / f"{variant}.mp4"
                for variant in generated_variants
            }
            if not all(path.exists() for path in raw_videos.values()):
                raise RuntimeError("Task 1 generator did not produce every requested rollout.")

            web_dir = output_dir / "web"
            web_dir.mkdir(parents=True, exist_ok=True)
            if baseline_dir is not None:
                for variant in ("original", "small"):
                    shutil.copy2(
                        baseline_dir / "web" / f"{variant}.mp4",
                        web_dir / f"{variant}.mp4",
                    )
            for variant, source in raw_videos.items():
                self._transcode_task1_video(source, web_dir / f"{variant}.mp4")
            web_videos = {
                variant: web_dir / f"{variant}.mp4"
                for variant in TASK1_VARIANTS
            }
            if not all(path.exists() for path in web_videos.values()):
                raise RuntimeError("Task 1 result is missing a baseline or generated rollout.")

            small_mse = self._video_mse(web_videos["original"], web_videos["small"])
            large_mse = self._video_mse(web_videos["original"], web_videos["large"])

            generated = {
                str(item.get("variant")): bool(item.get("success"))
                for item in summary.get("results", [])
            }
            baseline_success = baseline_result.get("task_success", {})
            task_success = {
                variant: bool(generated.get(variant, baseline_success.get(variant, False)))
                for variant in TASK1_VARIANTS
            }
            all_task_success = all(task_success.values())
            response_order_pass = bool(large_mse > small_mse)
            oracle_ratio = float(small_mse / large_mse) if large_mse > 0 else None
            result = {
                "task_success": task_success,
                "all_task_success": all_task_success,
                "small_response_mse": small_mse,
                "large_response_mse": large_mse,
                "oracle_ratio": oracle_ratio,
                "response_order_pass": response_order_pass,
                "valid_probe": bool(all_task_success and response_order_pass),
                "elapsed_sec": float(summary.get("elapsed_sec", 0.0)),
                "generated_variants": list(generated_variants),
            }
            cached_output_dir = None
            try:
                cached_output_dir = self._store_task1_cache(
                    demo_id,
                    demo,
                    large_magnitude,
                    result,
                    output_dir,
                )
            except Exception as cache_error:
                print(f"Task 1 cache write failed: {cache_error!r}", file=sys.stderr, flush=True)
            with self._task1_lock:
                current = self._task1_jobs.get(job_id)
                if current is not None:
                    current["state"] = "ready"
                    current["result"] = result
                    if cached_output_dir is not None:
                        current["output_dir"] = str(cached_output_dir)
            if cached_output_dir is not None:
                self._remove_task1_output(output_dir)
        except Exception as exc:
            self._remove_task1_output(output_dir)
            with self._task1_lock:
                current = self._task1_jobs.get(job_id)
                if current is not None:
                    current["state"] = "error"
                    current["error"] = repr(exc)
        finally:
            with self._client_lock:
                self._task1_running = False
            self.prewarm_default_session()

    def shutdown(self) -> None:
        def close_session() -> None:
            with self._state_lock:
                session = self.session
                self.session = None
            if session is None:
                return
            output_dir = session.output_dir
            try:
                session.close()
            finally:
                self._remove_session_output(output_dir)

        try:
            self.executor.submit(close_session).result(timeout=120)
        finally:
            self.task1_executor.shutdown(wait=False, cancel_futures=True)
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.clear_model_video()
            if self.public_demo:
                self._clear_task1_transient_outputs()

    def sim_call(self, func: Callable[[], Any]) -> Any:
        return self.executor.submit(func).result()

    def session_status(self) -> dict[str, Any]:
        with self._state_lock:
            session = self.session
        status = session.status() if session is not None else {
            "state": "idle",
            "episode_id": self.args.episode_id,
            "frame_id": self.args.frame_id,
            "command_count": 0,
            "active_arm": "left",
            "output_dir": str(self.output_root),
        }
        status.update(
            {
                "server": "task3_operator",
                "supported_commands": DEFAULT_COMMANDS,
            }
        )
        return status

    def _episode_id_from_payload(self, payload: dict[str, Any]) -> str:
        if payload.get("episode_id"):
            return str(payload["episode_id"])
        if payload.get("task") and payload.get("episode") is not None:
            return f"{payload['task']}__{int(payload['episode']):06d}"
        return str(self.args.episode_id)

    @staticmethod
    def _session_matches(
        session: Task3TeleopSession | None,
        episode_id: str,
        frame_id: int,
        *,
        require_pristine: bool,
    ) -> bool:
        if session is None:
            return False
        config = getattr(session, "config", None)
        if config is None:
            return False
        if str(getattr(config, "episode_id", "")) != str(episode_id):
            return False
        if int(getattr(config, "frame_id", -1)) != int(frame_id):
            return False
        if require_pristine and bool(getattr(session, "command_log", [])):
            return False
        try:
            return session.status().get("state") == "ready"
        except Exception:
            return False

    def _build_session(
        self,
        payload: dict[str, Any],
        frame_callback: Callable[[np.ndarray], None] | None,
    ) -> tuple[Task3TeleopSession, dict[str, Any], dict[str, Any] | None]:
        episode_id = self._episode_id_from_payload(payload)
        parsed = parse_episode_id(episode_id)
        if parsed is None:
            raise ValueError(f"Invalid Task 3 episode ID: {episode_id}")
        task, episode = validate_scene_selection(*parsed)
        annotation_info = self.ensure_scene_annotation(task, episode)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        output_dir = self.output_root / f"{episode_id}_{stamp}"
        cfg = Task3SessionConfig(
            robotwin_root=Path(self.args.robotwin_root).expanduser().resolve(),
            annotation_root=Path(self.args.annotation_root).expanduser().resolve(),
            output_dir=output_dir,
            episode_id=episode_id,
            frame_id=int(payload.get("frame_id", self.args.frame_id)),
            down_sample=int(self.args.down_sample),
            task_config=self.args.task_config,
            planar_step=float(self.args.planar_step),
            vertical_step=float(self.args.vertical_step),
            lateral_step_mult=float(self.args.lateral_step_mult),
            rotate_deg=float(self.args.rotate_deg),
            fps=float(self.args.fps),
            record_every_path_step=int(self.args.record_every_path_step),
            hold_frames=int(self.args.hold_frames),
            screen_step_px=float(self.args.screen_step_px),
            gripper_step=float(self.args.gripper_step),
            gripper_interpolation_steps=int(self.args.gripper_interpolation_steps),
            runtime_execution_mode=self.args.runtime_execution_mode,
            show_overlay=bool(self.args.show_video_overlay),
            max_joint_delta=float(self.args.max_joint_delta),
            fd_joint_eps=float(self.args.fd_joint_eps),
            control_mode=self.args.control_mode,
            fixed_delta_source=Path(self.args.fixed_delta_source).expanduser().resolve()
            if self.args.fixed_delta_source
            else None,
            prefix_replay_steps_per_action=int(self.args.prefix_replay_steps_per_action),
            record_artifacts=not self.public_demo,
        )
        session = Task3TeleopSession(cfg)
        session.set_frame_callback(frame_callback)
        try:
            status = session.reset()
        except Exception:
            session.close()
            raise
        return session, status, annotation_info

    def prewarm_default_session(self):
        if not bool(getattr(self.args, "prewarm_default", False)):
            return None
        episode_id = str(self.args.episode_id)
        frame_id = int(self.args.frame_id)
        with self._state_lock:
            existing_session = self.session
        if self._session_matches(
            existing_session,
            episode_id,
            frame_id,
            require_pristine=True,
        ):
            return None

        with self._prewarm_lock:
            if self._prewarm_future is not None and not self._prewarm_future.done():
                return self._prewarm_future

            def run() -> None:
                with self._state_lock:
                    if self.session is not None:
                        return
                session = None
                try:
                    session, _, _ = self._build_session(
                        {"episode_id": episode_id, "frame_id": frame_id},
                        frame_callback=None,
                    )
                    with self._state_lock:
                        if self.session is None:
                            self.session = session
                            session = None
                except Exception as exc:
                    print(f"Task 3 default prewarm failed: {exc!r}", file=sys.stderr, flush=True)
                finally:
                    if session is not None:
                        output_dir = session.output_dir
                        try:
                            session.close()
                        finally:
                            self._remove_session_output(output_dir)

            self._prewarm_future = self.executor.submit(run)
            return self._prewarm_future

    def start_session(self, payload: dict[str, Any], frame_callback: Callable[[np.ndarray], None] | None) -> dict[str, Any]:
        episode_id = self._episode_id_from_payload(payload)
        frame_id = int(payload.get("frame_id", self.args.frame_id))
        self.clear_model_video()
        self._model_initial_jpeg = None

        def model_frame_callback(frame: np.ndarray) -> None:
            if self._model_initial_jpeg is None:
                self._model_initial_jpeg = encode_jpeg(frame, quality=92)
            if frame_callback is not None:
                frame_callback(frame)

        def run() -> dict[str, Any]:
            with self._state_lock:
                old_session = self.session
                self.session = None
            if self._session_matches(
                old_session,
                episode_id,
                frame_id,
                require_pristine=True,
            ):
                old_session.set_frame_callback(model_frame_callback)
                old_session.emit_frame()
                status = old_session.status()
                status["prewarmed"] = True
                with self._state_lock:
                    self.session = old_session
                if payload.get("client_session_id") is not None:
                    status["client_session_id"] = payload.get("client_session_id")
                return status
            if old_session is not None:
                old_output_dir = old_session.output_dir
                try:
                    old_session.close()
                finally:
                    self._remove_session_output(old_output_dir)
            session, status, annotation_info = self._build_session(
                payload,
                model_frame_callback,
            )
            with self._state_lock:
                self.session = session
            if annotation_info is not None:
                status["annotation"] = annotation_info["annotation"]
                status["annotation_created"] = annotation_info["created"]
            if payload.get("client_session_id") is not None:
                status["client_session_id"] = payload.get("client_session_id")
            return status

        return self.sim_call(run)

    def step_command(
        self,
        command: str,
        arm: str,
        frame_callback: Callable[[np.ndarray], None] | None,
        input_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if self.session is None:
                raise RuntimeError("No active Task3 session; send start_session first")
            self.session.set_frame_callback(frame_callback)
            return self.session.step_command(command, arm=arm, input_event=input_event)

        return self.sim_call(run)

    def export_trace(self, finalize_video: bool) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if self.session is None:
                raise RuntimeError("No active Task3 session")
            return self.session.export_trace(finalize_video=finalize_video)

        return self.sim_call(run)

    def evaluate_trace(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if self.session is None:
                raise RuntimeError("No active Task3 session")
            return self.session.evaluate_trace()

        return self.sim_call(run)


class Task3RequestHandler(BaseHTTPRequestHandler):
    server_version = "Task3Operator/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def runtime(self) -> OperatorRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[task3-ui] {self.address_string()} {fmt % args}", flush=True)

    def send_json_response(self, status: int, payload: dict[str, Any]) -> None:
        payload = self.runtime.public_payload(payload)
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/ws":
            self.handle_ws()
            return
        if self.runtime.public_demo and parsed.path in {"/status", "/reference-video", "/artifact"}:
            self.send_error(404, "not found")
            return
        if parsed.path == "/status":
            self.send_json_response(200, self.runtime.session_status())
            return
        if parsed.path == "/api/scenes":
            self.send_json_response(200, self.runtime.scene_inventory())
            return
        if parsed.path == "/api/task3/models":
            self.send_json_response(200, self.runtime.task3_model_config())
            return
        model_video_match = TASK3_MODEL_VIDEO_RE.match(parsed.path)
        if model_video_match:
            self.handle_model_video(model_video_match.group("video_id"))
            return
        if parsed.path == "/api/task1/config":
            self.send_json_response(200, self.runtime.task1_demo_config())
            return
        task1_match = TASK1_JOB_RE.match(parsed.path)
        if task1_match:
            job_id = task1_match.group("job_id")
            variant = task1_match.group("variant")
            if variant is not None:
                self.handle_task1_video(job_id, variant)
                return
            try:
                self.send_json_response(200, self.runtime.task1_job_status(job_id))
            except KeyError:
                self.send_error(404, "Task 1 job not found")
            return
        if parsed.path == "/reference-video":
            self.handle_reference_video(parsed.query)
            return
        if parsed.path == "/artifact":
            self.handle_artifact(parsed.query)
            return
        if parsed.path in {"/", "/index.html"}:
            data = (static_root() / "operator.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "not found")

    def handle_model_video(self, video_id: str) -> None:
        try:
            data, content_type = self.runtime.resolve_model_video(video_id)
        except FileNotFoundError:
            self.send_error(404, "Task 3 model video not found")
            return

        size = len(data)
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            try:
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200

        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self.wfile.write(body)

    def handle_reference_video(self, query: str) -> None:
        params = parse_qs(query)
        task = (params.get("task") or [""])[0]
        try:
            episode = int((params.get("episode") or [""])[0])
            target = self.runtime.resolve_reference_video(task, episode)
        except PermissionError as exc:
            self.send_error(403, str(exc))
            return
        except Exception as exc:
            self.send_error(404, str(exc))
            return

        size = target.stat().st_size
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            try:
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200

        with target.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_task1_video(self, job_id: str, variant: str) -> None:
        try:
            target = self.runtime.resolve_task1_video(job_id, variant)
        except Exception:
            self.send_error(404, "Task 1 video not found")
            return

        size = target.stat().st_size
        start = 0
        end = size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            try:
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else size - 1
                start = max(0, min(start, size - 1))
                end = max(start, min(end, size - 1))
                status = 206
            except ValueError:
                start, end, status = 0, size - 1, 200

        with target.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_artifact(self, query: str) -> None:
        raw = (parse_qs(query).get("path") or [""])[0]
        if not raw:
            self.send_error(400, "missing artifact path")
            return
        target = Path(raw).expanduser().resolve()
        root = self.runtime.output_root.resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self.send_error(403, "artifact outside output root")
            return
        if not target.exists() or not target.is_file():
            self.send_error(404, "artifact not found")
            return
        content_type = {
            ".json": "application/json",
            ".mp4": "video/mp4",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(target.suffix.lower(), "application/octet-stream")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}
        try:
            if self.path == "/api/task1/generate":
                self.send_json_response(
                    202,
                    self.runtime.start_task1_job(
                        payload.get("large_magnitude"),
                        payload.get("demo_id"),
                    ),
                )
                return
            if self.runtime.public_demo:
                self.send_json_response(
                    403,
                    {"type": "error", "message": "Artifact operations are disabled in public demo mode."},
                )
                return
            if self.path == "/api/export-trace":
                self.send_json_response(200, self.runtime.export_trace(bool(payload.get("finalize_video", False))))
            elif self.path == "/api/evaluate-trace":
                self.send_json_response(200, self.runtime.evaluate_trace())
            else:
                self.send_error(404, "not found")
        except Task1BusyError as exc:
            self.send_json_response(409, {"type": "error", "message": str(exc)})
        except ValueError as exc:
            self.send_json_response(400, {"type": "error", "message": str(exc)})
        except Exception as exc:
            self.send_json_response(500, {"type": "error", "message": repr(exc)})

    def handle_ws(self) -> None:
        client_token = object()
        if not self.runtime.acquire_client(client_token):
            self.send_json_response(
                409,
                {"type": "error", "message": "The live simulator is currently in use."},
            )
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "missing websocket key")
            self.runtime.release_client(client_token)
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        peer = WebSocketPeer(self, sanitizer=self.runtime.public_payload)
        try:
            peer.send_json({"type": "status", **self.runtime.session_status()})
            while True:
                frame = read_ws_frame(self.rfile)
                if frame is None:
                    return
                opcode, payload = frame
                if opcode == 8:
                    peer.send_close()
                    return
                if opcode == 9:
                    peer.send_pong(payload)
                    continue
                if opcode != 1:
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                    self.handle_sim_message(peer, message)
                except Exception as exc:
                    peer.send_json({"type": "error", "message": repr(exc)})
        finally:
            self.runtime.release_client(client_token)

    def handle_sim_message(self, peer: "WebSocketPeer", message: dict[str, Any]) -> None:
        msg_type = str(message.get("type", ""))
        request_started = time.perf_counter()
        frame_timing = {"count": 0, "jpeg_encode_ms": 0.0, "socket_send_ms": 0.0}

        def frame_callback(frame: np.ndarray) -> None:
            encode_started = time.perf_counter()
            payload = encode_jpeg(frame)
            frame_timing["jpeg_encode_ms"] += (time.perf_counter() - encode_started) * 1000.0
            send_started = time.perf_counter()
            peer.send_binary(payload)
            frame_timing["socket_send_ms"] += (time.perf_counter() - send_started) * 1000.0
            frame_timing["count"] += 1

        def attach_timing(update: dict[str, Any], step_started: float) -> dict[str, Any]:
            update["server_timing"] = {
                "total_ms": round((time.perf_counter() - request_started) * 1000.0, 3),
                "step_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                "frame_count": int(frame_timing["count"]),
                "jpeg_encode_ms": round(frame_timing["jpeg_encode_ms"], 3),
                "socket_send_ms": round(frame_timing["socket_send_ms"], 3),
            }
            return update

        def selected_model_or_error() -> str | None:
            try:
                return self.runtime.require_task3_model(message.get("model"))
            except Exception:
                raw_model = str(message.get("model") or "")
                model_label = TASK3_MODELS.get(raw_model, "Selected model")
                peer.send_json(
                    {
                        "type": "model_status",
                        "state": "error",
                        "model": raw_model,
                        "model_label": model_label,
                        "message": f"{model_label} is not ready.",
                    }
                )
                return None

        def infer_selected_model(model: str) -> None:
            peer.send_json(
                {
                    "type": "model_status",
                    "state": "running",
                    "model": model,
                    "model_label": TASK3_MODELS[model],
                }
            )
            try:
                result = self.runtime.infer_task3_model(
                    model,
                    message.get("client_session_id"),
                )
            except Exception:
                peer.send_json(
                    {
                        "type": "model_status",
                        "state": "error",
                        "model": model,
                        "model_label": TASK3_MODELS[model],
                        "message": f"{TASK3_MODELS[model]} could not complete this action.",
                    }
                )
                return
            peer.send_json(result)
            peer.send_json(
                {
                    "type": "model_status",
                    "state": "ready",
                    "model": model,
                    "model_label": TASK3_MODELS[model],
                    "elapsed_sec": result["elapsed_sec"],
                    "command_count": result["command_count"],
                }
            )

        if msg_type in {"start_session", "reset"}:
            episode_id = self.runtime._episode_id_from_payload(message)
            peer.send_json(
                {
                    "type": "status",
                    "state": "initializing",
                    "episode_id": episode_id,
                    "frame_id": int(message.get("frame_id", self.runtime.args.frame_id)),
                    "active_arm": "left",
                    "active_command": "start",
                    "client_session_id": message.get("client_session_id"),
                }
            )
            status = self.runtime.start_session(message, frame_callback)
            peer.send_json({"type": "status", **status})
            return
        if msg_type == "keydown":
            command = KEY_BINDINGS.get(str(message.get("code") or message.get("key") or ""))
            if command is None:
                return
            model = selected_model_or_error()
            if model is None:
                return
            arm = str(message.get("active_arm") or message.get("arm") or "left").lower()
            if arm not in {"left", "right"}:
                arm = "left"
            input_event = {
                "source": "keyboard",
                "message_type": "keydown",
                "arm": arm,
                "active_arm": arm,
                "command": command,
                "code": message.get("code"),
                "key": message.get("key"),
                "client_time_ms": message.get("client_time_ms"),
                "server_time": time.time(),
            }
            active_label = f"{arm} {command}"
            peer.send_json({"type": "status", "state": "running", "active_arm": arm, "active_command": active_label})
            step_started = time.perf_counter()
            update = self.runtime.step_command(command, arm, frame_callback, input_event=input_event)
            peer.send_json(attach_timing(update, step_started))
            peer.send_json({"type": "status", **self.runtime.session_status()})
            infer_selected_model(model)
            return
        if msg_type == "command":
            command = str(message.get("command", "")).lower()
            model = selected_model_or_error()
            if model is None:
                return
            arm = str(message.get("active_arm") or message.get("arm") or "left").lower()
            if arm not in {"left", "right"}:
                arm = "left"
            input_event = {
                "source": str(message.get("source") or "button"),
                "message_type": "command",
                "arm": arm,
                "active_arm": arm,
                "command": command,
                "client_time_ms": message.get("client_time_ms"),
                "server_time": time.time(),
            }
            active_label = f"{arm} {command}"
            peer.send_json({"type": "status", "state": "running", "active_arm": arm, "active_command": active_label})
            step_started = time.perf_counter()
            update = self.runtime.step_command(command, arm, frame_callback, input_event=input_event)
            peer.send_json(attach_timing(update, step_started))
            peer.send_json({"type": "status", **self.runtime.session_status()})
            infer_selected_model(model)
            return
        if msg_type == "export_trace":
            if self.runtime.public_demo:
                peer.send_json({"type": "error", "message": "Artifact export is disabled."})
                return
            peer.send_json({"type": "artifact", **self.runtime.export_trace(bool(message.get("finalize_video", False)))})
            return
        if msg_type == "evaluate_trace":
            if self.runtime.public_demo:
                peer.send_json({"type": "error", "message": "Trace evaluation is disabled."})
                return
            result = self.runtime.evaluate_trace()
            peer.send_json({"type": "eval_result", **result.get("summary", {}), "result": result})
            return
        peer.send_json({"type": "error", "message": f"Unsupported message type: {msg_type}"})


class WebSocketPeer:
    def __init__(
        self,
        handler: Task3RequestHandler,
        sanitizer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.handler = handler
        self.sanitizer = sanitizer
        self._lock = threading.Lock()

    def send_json(self, payload: dict[str, Any]) -> None:
        if self.sanitizer is not None:
            payload = self.sanitizer(payload)
        self._send(1, json_bytes(payload))

    def send_binary(self, payload: bytes) -> None:
        self._send(2, payload)

    def send_pong(self, payload: bytes) -> None:
        self._send(10, payload)

    def send_close(self) -> None:
        self._send(8, b"")

    def _send(self, opcode: int, payload: bytes) -> None:
        with self._lock:
            self.handler.wfile.write(make_ws_frame(opcode, payload))
            self.handler.wfile.flush()


class Task3HTTPServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], runtime: OperatorRuntime):
        super().__init__(addr, Task3RequestHandler)
        self.runtime = runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the WorldSimProbe Task 3 RoboTwin operator console."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Run a single-client ephemeral demo with artifact operations and internal paths disabled.",
    )
    robotwin_root = os.environ.get("ROBOTWIN_ROOT")
    parser.add_argument(
        "--robotwin-root",
        default=robotwin_root,
        required=robotwin_root is None,
        help="Path to an official RoboTwin checkout.",
    )
    annotation_root = os.environ.get("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT")
    parser.add_argument(
        "--annotation-root",
        default=annotation_root,
        required=annotation_root is None,
        help="Directory containing one replay annotation JSON per episode.",
    )
    parser.add_argument("--output-root", default=str(Path("outputs") / "task3_operator"))
    parser.add_argument(
        "--task1-generator-script",
        default=os.environ.get("WORLDSIMPROBE_TASK1_GENERATOR"),
        help="Optional path to the controlled RoboTwin Task 1 triplet generator.",
    )
    parser.add_argument(
        "--task1-timeout",
        type=float,
        default=900.0,
        help="Maximum seconds allowed for one Task 1 triplet generation.",
    )
    parser.add_argument(
        "--task3-lingbot-endpoint",
        default=os.environ.get("WORLDSIMPROBE_TASK3_LINGBOT_ENDPOINT"),
        help="Optional LingBot-VA worker URL for paired Task 3 inference.",
    )
    parser.add_argument(
        "--task3-ctrlworld-endpoint",
        default=os.environ.get("WORLDSIMPROBE_TASK3_CTRLWORLD_ENDPOINT"),
        help="Optional Ctrl-World worker URL for paired Task 3 inference.",
    )
    parser.add_argument(
        "--task3-model-timeout",
        type=float,
        default=900.0,
        help="Maximum seconds allowed for one selected-model Task 3 inference.",
    )
    parser.add_argument(
        "--prewarm-default",
        action="store_true",
        help="Initialize and retain the default Task 3 scene before the first client connects.",
    )
    parser.add_argument("--episode-id", default="stack_blocks_three__000000")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--down-sample", type=int, default=3)
    parser.add_argument("--task-config", default="cross_clean_50")
    parser.add_argument("--planar-step", type=float, default=0.06)
    parser.add_argument("--vertical-step", type=float, default=0.02)
    parser.add_argument("--lateral-step-mult", type=float, default=1.0)
    parser.add_argument("--rotate-deg", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=default_video_fps())
    parser.add_argument("--record-every-path-step", type=int, default=6)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--screen-step-px", type=float, default=24.0)
    parser.add_argument("--gripper-step", type=float, default=0.2)
    parser.add_argument("--gripper-interpolation-steps", type=int, default=12)
    parser.add_argument(
        "--runtime-execution-mode",
        choices=["physical_drive_target", "force_qpos"],
        default="physical_drive_target",
        help="Runtime command execution mode; force_qpos is debug-only and can inject non-physical contact impulses.",
    )
    parser.add_argument(
        "--show-video-overlay",
        action="store_true",
        help="Draw the command overlay into live and recorded frames.",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.72)
    parser.add_argument("--fd-joint-eps", type=float, default=0.01)
    parser.add_argument("--control-mode", choices=["calibrated", "fixed_delta"], default="calibrated")
    parser.add_argument("--fixed-delta-source", default=None)
    parser.add_argument(
        "--prefix-replay-steps-per-action",
        type=int,
        default=15,
        help="Physics steps per saved RoboTwin frame while replaying the reference prefix before teleop.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = OperatorRuntime(args)
    server = Task3HTTPServer((args.host, int(args.port)), runtime)
    task1_state = "enabled" if runtime.task1_generator_path() is not None else "disabled"
    task3_models = ", ".join(
        TASK3_MODELS[model]
        for model, endpoint in runtime.model_endpoints.items()
        if endpoint is not None
    ) or "disabled"
    print(
        f"WorldSimProbe live console: http://{args.host}:{args.port}/ "
        f"(Task 1 {task1_state}, Task 3 simulator enabled, models: {task3_models})",
        flush=True,
    )
    runtime.prewarm_default_session()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
