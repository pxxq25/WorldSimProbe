#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


MODEL_LABELS = {
    "lingbot-va": "LingBot-VA",
    "ctrl-world": "Ctrl-World",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve one persistent Task 3 world-model worker."
    )
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--ctrlworld-root", type=Path)
    parser.add_argument("--ctrlworld-checkpoint", type=Path)
    parser.add_argument("--ctrlworld-svd", type=Path)
    parser.add_argument("--ctrlworld-clip", type=Path)
    parser.add_argument("--ctrlworld-stat", type=Path)
    parser.add_argument("--ctrlworld-steps", type=int, default=50)
    parser.add_argument("--lingbot-repo", type=Path)
    parser.add_argument("--lingbot-transformer", type=Path)
    parser.add_argument("--lingbot-base", type=Path)
    parser.add_argument("--lingbot-steps", type=int, default=25)
    return parser.parse_args()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def decode_jpeg(encoded: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(encoded, validate=True)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), copy=True)


def encode_video(frames: list[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise RuntimeError("Model produced no frames.")
    with tempfile.TemporaryDirectory(prefix="worldsimprobe-task3-model-") as directory:
        path = Path(directory) / "rollout.mp4"
        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return path.read_bytes()


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**31)


def command_signature(entry: dict[str, Any]) -> str:
    value = {
        "command": entry.get("command"),
        "arm": entry.get("arm"),
        "action": entry.get("after_action") or entry.get("action_14d"),
    }
    return hashlib.sha256(json_bytes(value)).hexdigest()


class CtrlWorldRuntime:
    fps = 5

    def __init__(self, args: argparse.Namespace):
        required = {
            "ctrlworld_root": args.ctrlworld_root,
            "ctrlworld_checkpoint": args.ctrlworld_checkpoint,
            "ctrlworld_svd": args.ctrlworld_svd,
            "ctrlworld_clip": args.ctrlworld_clip,
            "ctrlworld_stat": args.ctrlworld_stat,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing Ctrl-World arguments: {', '.join(missing)}")
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.bfloat16
        self.root = args.ctrlworld_root.expanduser().resolve()
        sys.path.insert(0, str(self.root))

        from config import wm_args
        from models.ctrl_world import CrtlWorld

        model_args = wm_args()
        model_args.svd_model_path = str(args.ctrlworld_svd.expanduser().resolve())
        model_args.clip_model_path = str(args.ctrlworld_clip.expanduser().resolve())
        model_args.camera_mode = "head"
        model_args.ckpt_path = str(args.ctrlworld_checkpoint.expanduser().resolve())
        model_args.val_model_path = model_args.ckpt_path
        model_args.num_inference_steps = int(args.ctrlworld_steps)
        model_args.decode_chunk_size = 7
        model_args.dtype = self.dtype
        model_args.width = 320
        model_args.height = 192
        model_args.num_history = 6
        model_args.num_frames = 5
        model_args.pred_step = 5
        model_args.action_dim = 14
        model_args.text_cond = True
        model_args.frame_level_cond = True
        model_args.his_cond_zero = False
        self.model_args = model_args

        stat = json.loads(args.ctrlworld_stat.read_text(encoding="utf-8"))
        self.state_p01 = np.asarray(stat["state_01"], dtype=np.float32)[None, :]
        self.state_p99 = np.asarray(stat["state_99"], dtype=np.float32)[None, :]

        torch.backends.cuda.matmul.allow_tf32 = True
        model = CrtlWorld(model_args)
        model.load_state_dict(
            torch.load(args.ctrlworld_checkpoint, map_location="cpu")
        )
        self.model = model.to(self.device).to(self.dtype).eval()
        self.pipeline = self.model.pipeline
        self.session: dict[str, Any] | None = None

    @torch.no_grad()
    def encode_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        frame = resize_frame(frame_rgb, 320, 192)
        tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        tensor = tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        latent = self.pipeline.vae.encode(tensor).latent_dist.sample()
        return latent.mul_(self.pipeline.vae.config.scaling_factor)[0].detach()

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        decoded = []
        flat = latents.flatten(0, 1)
        for start in range(0, flat.shape[0], self.model_args.decode_chunk_size):
            chunk = flat[start : start + self.model_args.decode_chunk_size]
            chunk = chunk / self.pipeline.vae.config.scaling_factor
            decoded.append(
                self.pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample
            )
        video = torch.cat(decoded, dim=0)
        video = video.reshape(latents.shape[0], latents.shape[1], *video.shape[1:])
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
        return (
            video.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
            .astype(np.uint8)
        )

    def reset_session(self, payload: dict[str, Any]) -> None:
        trace = payload["trace"]
        frame = decode_jpeg(payload["initial_frame_jpeg"])
        first_latent = self.encode_frame(frame)
        first_decoded = self.decode_latents(
            first_latent[None, None].to(device=self.device, dtype=self.dtype)
        )[0, 0]
        current_action = np.asarray(trace["reset_state"]["action"], dtype=np.float32)
        if current_action.shape != (14,):
            raise ValueError(f"Expected a 14D initial action, got {current_action.shape}")
        self.session = {
            "id": str(payload["session_id"]),
            "instruction": str(payload.get("instruction") or ""),
            "history_latents": [first_latent for _ in range(24)],
            "history_actions": [current_action[None, :] for _ in range(24)],
            "current_action": current_action,
            "last_frame": first_decoded,
            "frame_count": 1,
            "signatures": [],
        }

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        scaled = 2 * (actions - self.state_p01) / (self.state_p99 - self.state_p01 + 1e-8) - 1
        return np.clip(scaled, -1, 1).astype(np.float32)

    @staticmethod
    def action_from_entry(entry: dict[str, Any]) -> np.ndarray:
        action = entry.get("after_action") or entry.get("action_14d")
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (14,):
            raise ValueError(f"Expected a 14D command action, got {value.shape}")
        return value

    @staticmethod
    def action_chunk(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.stack(
            [(1.0 - alpha) * current + alpha * target for alpha in np.linspace(0, 1, 5)],
            axis=0,
        ).astype(np.float32)

    @torch.no_grad()
    def process_entry(self, entry: dict[str, Any], index: int) -> list[np.ndarray]:
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        session = self.session
        if session is None:
            raise RuntimeError("Ctrl-World session is not initialized")
        target = self.action_from_entry(entry)
        chunk = self.action_chunk(session["current_action"], target)
        hidx = [-1] * 6 if len(session["history_latents"]) < 8 else [0, 0, -8, -6, -4, -2]
        history_action = np.concatenate(
            [session["history_actions"][item] for item in hidx], axis=0
        )
        action_cond = np.concatenate([history_action, chunk], axis=0)
        action_tensor = torch.tensor(
            self.normalize(action_cond), device=self.device, dtype=self.dtype
        ).unsqueeze(0)
        history = torch.stack(
            [session["history_latents"][item] for item in hidx], dim=0
        ).unsqueeze(0)
        current_latent = session["history_latents"][-1].unsqueeze(0)
        text_token = self.model.action_encoder(
            action_tensor,
            [session["instruction"]],
            self.model.tokenizer,
            self.model.text_encoder,
            self.model_args.frame_level_cond,
        )
        torch.manual_seed(stable_seed(self.args.seed, f"{session['id']}:{index}"))
        torch.cuda.manual_seed_all(stable_seed(self.args.seed, f"{session['id']}:{index}"))
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            self.pipeline,
            image=current_latent,
            text=text_token,
            width=self.model_args.width,
            height=self.model_args.height,
            num_frames=self.model_args.num_frames,
            history=history,
            num_inference_steps=self.model_args.num_inference_steps,
            decode_chunk_size=self.model_args.decode_chunk_size,
            max_guidance_scale=self.model_args.guidance_scale,
            fps=self.model_args.fps,
            motion_bucket_id=self.model_args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=self.model_args.frame_level_cond,
            his_cond_zero=self.model_args.his_cond_zero,
        )
        frames = list(self.decode_latents(pred_latents)[0])
        if not frames:
            raise RuntimeError("Ctrl-World produced no incremental frames")
        session["history_latents"].append(pred_latents[0, -1].detach())
        session["history_actions"].append(chunk[-1:])
        session["current_action"] = target
        session["last_frame"] = frames[-1]
        session["frame_count"] += len(frames)
        session["signatures"].append(command_signature(entry))
        return frames

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        entries = list(payload["trace"].get("command_log") or [])
        signatures = [command_signature(entry) for entry in entries]
        reset_required = (
            self.session is None
            or self.session["id"] != str(payload["session_id"])
            or signatures[: len(self.session["signatures"])] != self.session["signatures"]
            or len(signatures) < len(self.session["signatures"])
        )
        if reset_required:
            self.reset_session(payload)
        assert self.session is not None
        previous_command_count = len(self.session["signatures"])
        segment_frames = [self.session["last_frame"]]
        for index in range(len(self.session["signatures"]), len(entries)):
            segment_frames.extend(self.process_entry(entries[index], index + 1))
        video = encode_video(segment_frames, self.fps)
        return {
            "video_base64": base64.b64encode(video).decode("ascii"),
            "command_count": len(self.session["signatures"]),
            "frame_count": int(self.session["frame_count"]),
            "segment_frame_count": len(segment_frames),
            "continued": not reset_required and previous_command_count > 0,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "inference_mode": "stateful_incremental_14d_joint_targets",
        }


class LingBotRuntime:
    fps = 10

    def __init__(self, args: argparse.Namespace):
        required = {
            "lingbot_repo": args.lingbot_repo,
            "lingbot_transformer": args.lingbot_transformer,
            "lingbot_base": args.lingbot_base,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Missing LingBot-VA arguments: {', '.join(missing)}")
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.bfloat16
        repo = args.lingbot_repo.expanduser().resolve()
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "wan_va"))

        from wan_va.configs import VA_CONFIGS
        from wan_va.distributed.fsdp import shard_model
        from wan_va.distributed.util import _configure_model
        from wan_va.modules.utils import (
            load_text_encoder,
            load_tokenizer,
            load_transformer,
            load_vae,
        )
        from wan_va.utils import FlowMatchScheduler

        self.config = copy.deepcopy(VA_CONFIGS["robotwin_fd_train"])
        base = args.lingbot_base.expanduser().resolve()
        self.tokenizer = load_tokenizer(base / "tokenizer")
        self.text_encoder = load_text_encoder(
            base / "text_encoder",
            torch_dtype=self.dtype,
            torch_device=self.device,
        ).eval()
        self.vae = load_vae(
            base / "vae", torch_dtype=self.dtype, torch_device=self.device
        ).eval()
        transformer = load_transformer(
            args.lingbot_transformer.expanduser().resolve(),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch_masked",
        )
        self.transformer = _configure_model(
            model=transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        self.scheduler = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.prompt_cache: dict[str, torch.Tensor] = {}
        self.session: dict[str, Any] | None = None

    @staticmethod
    def get_relative_pose(pose: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_quat(pose[:, 3:7])
        first = Rotation.from_quat(np.tile(pose[:1, 3:7], (len(pose), 1)))
        relative_translation = pose[:, :3] - pose[:1, :3]
        relative_quaternion = (first.inv() * rotation).as_quat()
        return np.concatenate((relative_translation, relative_quaternion), axis=1)

    def prepare_actions(self, raw_action: np.ndarray) -> torch.Tensor:
        left = self.get_relative_pose(raw_action[:, :7])
        right = self.get_relative_pose(raw_action[:, 8:15])
        action = np.concatenate(
            (left, raw_action[:, 7:8], right, raw_action[:, 15:16]), axis=1
        )
        per_frame = int(self.config.action_per_frame)
        latent_frames = (len(action) - 1) // 4 + 1
        required = latent_frames * per_frame
        action = np.pad(action, ((per_frame, 0), (0, 0)), mode="constant")[:required]
        mask = np.ones_like(action, dtype=bool)
        action = np.pad(action, ((0, 0), (0, 1)), mode="constant")
        mask = np.pad(mask, ((0, 0), (0, 1)), mode="constant", constant_values=False)
        action = action[:, self.config.inverse_used_action_channel_ids]
        mask = mask[:, self.config.inverse_used_action_channel_ids]
        q01 = np.asarray(self.config.norm_stat["q01"], dtype=np.float32)[None]
        q99 = np.asarray(self.config.norm_stat["q99"], dtype=np.float32)[None]
        action = (action - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        action = np.clip(action, -1.5, 1.5) * mask
        action = action.reshape(latent_frames, per_frame, -1).transpose(2, 0, 1)[..., None]
        return torch.from_numpy(action).float().unsqueeze(0)

    def encode_prompt(self, text: str) -> torch.Tensor:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        cached = self.prompt_cache.get(text)
        if cached is not None:
            return cached
        inputs = self.tokenizer(
            [prompt_clean(text)],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(self.device)
        mask = inputs.attention_mask.to(self.device)
        seq_len = int(mask.gt(0).sum(dim=1)[0])
        with torch.no_grad():
            embeds = self.text_encoder(ids, mask).last_hidden_state.to(dtype=self.dtype)
        embeds = embeds[:, :seq_len]
        padding = embeds.new_zeros(1, 512 - embeds.shape[1], embeds.shape[2])
        value = torch.cat((embeds, padding), dim=1).cpu()
        self.prompt_cache[text] = value
        return value

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        with tempfile.TemporaryDirectory(prefix="lingbot-context-") as directory:
            path = Path(directory) / "context.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not initialize LingBot context preprocessing.")
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            writer.release()
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()
            capture.release()
        if not ok:
            raise RuntimeError("Could not decode LingBot context frame.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        import torch.nn.functional as functional

        video = (
            torch.from_numpy(np.ascontiguousarray(image))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(1)
        )
        video = functional.interpolate(
            video, size=(256, 320), mode="bilinear", align_corners=False
        ).unsqueeze(0)
        video = video / 255.0 * 2.0 - 1.0
        with torch.no_grad():
            posterior = self.vae.encode(
                video.to(device=self.device, dtype=self.dtype), return_dict=False
            )[0]
            mu = posterior.mode()
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(mu)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(mu)
        return ((mu - mean) / std).to(self.dtype)

    def decode_video(self, latents: torch.Tensor) -> list[np.ndarray]:
        from diffusers.video_processor import VideoProcessor

        processor = VideoProcessor(vae_scale_factor=1)
        mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents)
        denormalized = latents / inv_std + mean
        with torch.no_grad():
            video = self.vae.decode(
                denormalized.to(dtype=self.dtype), return_dict=False
            )[0].detach()
        return [
            np.asarray(frame * 255.0, dtype=np.uint8)
            for frame in processor.postprocess_video(video, output_type="np")[0]
        ]

    def make_grid_id(
        self,
        frame_count: int,
        height: int,
        width: int,
        patch_size: tuple[int, int, int],
        action: bool,
    ) -> torch.Tensor:
        from wan_va.utils import get_mesh_id

        if action:
            return get_mesh_id(frame_count, height, width, 1, 1, 0, action=True).to(
                self.device
            )
        return get_mesh_id(
            frame_count // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
            0,
            1,
            0,
        ).to(self.device)

    def denoise(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        import torch.nn.functional as functional
        from wan_va.modules.forward_dynamics import forward_dynamics_video
        from wan_va.utils import data_seq_to_patch

        frame_count = actions.shape[2]
        latents = torch.randn(
            (1, history.shape[1], frame_count, history.shape[3], history.shape[4]),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents[:, :, :1] = history
        self.scheduler.set_timesteps(int(self.args.lingbot_steps))
        timesteps = functional.pad(self.scheduler.timesteps, (0, 1), value=0)
        patch_size = tuple(self.config.patch_size)
        latent_grid = self.make_grid_id(
            frame_count, latents.shape[3], latents.shape[4], patch_size, False
        )
        action_grid = self.make_grid_id(
            actions.shape[2], actions.shape[3], actions.shape[4], patch_size, True
        )
        with torch.no_grad():
            for index, timestep in enumerate(timesteps):
                latent_timesteps = torch.ones(
                    frame_count, dtype=torch.float32, device=self.device
                ) * timestep
                latent_timesteps[:1] = 0
                prediction = forward_dynamics_video(
                    self.transformer,
                    {
                        "noisy_latents": latents,
                        "timesteps": latent_timesteps[None],
                        "grid_id": latent_grid[None],
                        "text_emb": text_emb,
                    },
                    {
                        "latent": actions,
                        "timesteps": torch.zeros(
                            (1, frame_count), dtype=torch.float32, device=self.device
                        ),
                        "grid_id": action_grid[None],
                    },
                    chunk_size=int(self.config.fd_chunk_size),
                    window_size=int(self.config.fd_window_size),
                    history_latent_frames=int(self.config.history_latent_frames),
                    action_prefix_frames=int(self.config.fd_action_prefix_frames),
                )
                prediction = data_seq_to_patch(
                    patch_size,
                    prediction,
                    frame_count,
                    latents.shape[3],
                    latents.shape[4],
                    batch_size=1,
                )
                if index != len(timesteps) - 1:
                    latents = self.scheduler.step(
                        prediction, timestep, latents, return_dict=False
                    )
                    latents[:, :, :1] = history
        return latents

    @staticmethod
    def eef_from_reset(trace: dict[str, Any]) -> np.ndarray:
        reset = trace["reset_state"]
        action = np.asarray(reset["action"], dtype=np.float32)
        left = np.asarray(reset["left_tcp_pose"], dtype=np.float32)
        right = np.asarray(reset["right_tcp_pose"], dtype=np.float32)
        value = np.concatenate((left, action[6:7], right, action[13:14]))
        if value.shape != (16,):
            raise ValueError(f"Expected a 16D initial EEF action, got {value.shape}")
        return value

    @staticmethod
    def eef_from_entry(entry: dict[str, Any], previous: np.ndarray) -> np.ndarray:
        action = np.asarray(
            entry.get("after_action") or entry.get("action_14d"), dtype=np.float32
        )
        left_pose = entry.get("left_tcp_pose")
        right_pose = entry.get("right_tcp_pose")
        left = previous[:7] if left_pose is None else np.asarray(left_pose, dtype=np.float32)
        right = previous[8:15] if right_pose is None else np.asarray(right_pose, dtype=np.float32)
        value = np.concatenate((left, action[6:7], right, action[13:14]))
        if value.shape != (16,):
            raise ValueError(f"Expected a 16D command EEF action, got {value.shape}")
        return value

    @staticmethod
    def interpolate_eef(start: np.ndarray, target: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation, Slerp

        times = np.linspace(0.0, 1.0, 9)
        output = np.zeros((9, 16), dtype=np.float32)
        for offset in (0, 8):
            output[:, offset : offset + 3] = (
                start[offset : offset + 3][None] * (1.0 - times[:, None])
                + target[offset : offset + 3][None] * times[:, None]
            )
            rotations = Rotation.from_quat(
                np.stack((start[offset + 3 : offset + 7], target[offset + 3 : offset + 7]))
            )
            output[:, offset + 3 : offset + 7] = Slerp([0.0, 1.0], rotations)(
                times
            ).as_quat()
            gripper = offset + 7
            output[:, gripper] = start[gripper] * (1.0 - times) + target[gripper] * times
        return output

    def reset_session(self, payload: dict[str, Any]) -> None:
        trace = payload["trace"]
        image = self.preprocess_image(decode_jpeg(payload["initial_frame_jpeg"]))
        latent = self.encode_image(image)
        first_frame = self.decode_video(latent)[0]
        self.session = {
            "id": str(payload["session_id"]),
            "instruction": str(payload.get("instruction") or ""),
            "history": latent,
            "current_eef": self.eef_from_reset(trace),
            "last_frame": first_frame,
            "frame_count": 1,
            "signatures": [],
        }

    def process_entry(self, entry: dict[str, Any], index: int) -> list[np.ndarray]:
        session = self.session
        if session is None:
            raise RuntimeError("LingBot-VA session is not initialized")
        target = self.eef_from_entry(entry, session["current_eef"])
        raw_actions = self.interpolate_eef(session["current_eef"], target)
        actions = self.prepare_actions(raw_actions).to(
            device=self.device, dtype=self.dtype
        )
        text = self.encode_prompt(session["instruction"]).to(
            device=self.device, dtype=self.dtype
        )
        generator = torch.Generator(device=self.device).manual_seed(
            stable_seed(self.args.seed, f"{session['id']}:{index}")
        )
        prediction = self.denoise(session["history"], actions, text, generator)
        frames = self.decode_video(prediction)
        incremental_frames = frames[1:]
        if not incremental_frames:
            raise RuntimeError("LingBot-VA produced no incremental frames")
        session["history"] = prediction[:, :, -1:].detach()
        session["current_eef"] = target
        session["last_frame"] = incremental_frames[-1]
        session["frame_count"] += len(incremental_frames)
        session["signatures"].append(command_signature(entry))
        return incremental_frames

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        entries = list(payload["trace"].get("command_log") or [])
        signatures = [command_signature(entry) for entry in entries]
        reset_required = (
            self.session is None
            or self.session["id"] != str(payload["session_id"])
            or signatures[: len(self.session["signatures"])] != self.session["signatures"]
            or len(signatures) < len(self.session["signatures"])
        )
        if reset_required:
            self.reset_session(payload)
        assert self.session is not None
        previous_command_count = len(self.session["signatures"])
        segment_frames = [self.session["last_frame"]]
        for index in range(len(self.session["signatures"]), len(entries)):
            segment_frames.extend(self.process_entry(entries[index], index + 1))
        video = encode_video(segment_frames, self.fps)
        return {
            "video_base64": base64.b64encode(video).decode("ascii"),
            "command_count": len(self.session["signatures"]),
            "frame_count": int(self.session["frame_count"]),
            "segment_frame_count": len(segment_frames),
            "continued": not reset_required and previous_command_count > 0,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "inference_mode": "stateful_incremental_16d_eef",
        }


class WorkerState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = "loading"
        self.runtime: CtrlWorldRuntime | LingBotRuntime | None = None
        self.error: str | None = None
        self.loaded_sec: float | None = None
        self.inference_lock = threading.Lock()

    def load(self) -> None:
        started = time.perf_counter()
        try:
            if self.args.model == "ctrl-world":
                self.runtime = CtrlWorldRuntime(self.args)
            else:
                self.runtime = LingBotRuntime(self.args)
            self.loaded_sec = round(time.perf_counter() - started, 3)
            self.state = "ready"
        except Exception as exc:
            self.error = repr(exc)
            self.state = "error"
            print(f"Model worker failed to load: {exc!r}", file=sys.stderr, flush=True)

    def health(self) -> dict[str, Any]:
        payload = {
            "model": self.args.model,
            "model_label": MODEL_LABELS[self.args.model],
            "state": self.state,
        }
        if self.loaded_sec is not None:
            payload["loaded_sec"] = self.loaded_sec
        return payload

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.state != "ready" or self.runtime is None:
            raise RuntimeError(f"{MODEL_LABELS[self.args.model]} is not ready")
        with self.inference_lock:
            return self.runtime.infer(payload)


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WorldSimProbeTask3Model/0.1"

    @property
    def state(self) -> WorkerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[task3-model] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, self.state.health())
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path != "/infer":
            self.send_error(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > 16 * 1024 * 1024:
            self.send_json(400, {"state": "error", "message": "Invalid request size."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.state.infer(payload)
        except Exception as exc:
            print(f"Task 3 model inference failed: {exc!r}", file=sys.stderr, flush=True)
            self.send_json(
                500,
                {
                    "state": "error",
                    "model": self.state.args.model,
                    "message": "Model inference did not complete.",
                },
            )
            return
        self.send_json(200, {"state": "ready", "model": self.state.args.model, **result})


class WorkerServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: WorkerState):
        super().__init__(address, WorkerHandler)
        self.state = state


def main() -> int:
    args = parse_args()
    state = WorkerState(args)
    threading.Thread(target=state.load, name=f"{args.model}-loader", daemon=True).start()
    server = WorkerServer((args.host, int(args.port)), state)
    print(
        f"{MODEL_LABELS[args.model]} Task 3 worker: http://{args.host}:{args.port}/health",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
