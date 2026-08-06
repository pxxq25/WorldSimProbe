from __future__ import annotations

import json
from pathlib import Path

import worldsimprobe.evaluation.task3_action_coverage.teleoperation.server as teleop_server
from worldsimprobe.evaluation.task3_action_coverage.teleoperation.robotwin_replay import (
    capture_head_camera_rgb,
)
from worldsimprobe.evaluation.task3_action_coverage.teleoperation.server import (
    OperatorRuntime,
    build_parser,
    static_root,
    validate_task1_magnitude,
    validate_scene_selection,
    validate_task3_model,
)
from worldsimprobe.evaluation.task3_action_coverage.teleoperation.trace import (
    evaluate_task3_trace,
)


def test_operator_parser_uses_explicit_external_roots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
        ]
    )
    assert args.robotwin_root == str(tmp_path / "RoboTwin")
    assert args.annotation_root == str(tmp_path / "annotations")
    assert args.host == "127.0.0.1"
    assert args.public_demo is False
    assert args.prewarm_default is False


def test_public_demo_sanitizes_paths_and_disables_persistent_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
        ]
    )
    runtime = OperatorRuntime(args)
    try:
        payload = runtime.public_payload(
            {
                "type": "status",
                "state": "ready",
                "output_dir": "/private/output",
                "trace": "/private/trace.json",
                "telemetry": {"task": "stack_blocks_three"},
                "tasks": [
                    {
                        "task": "stack_blocks_three",
                        "reference_video_path": "/private/reference.mp4",
                    }
                ],
            }
        )
    finally:
        runtime.shutdown()

    assert args.public_demo is True
    assert payload["state"] == "ready"
    assert payload["telemetry"] == {"task": "stack_blocks_three"}
    assert "output_dir" not in payload
    assert "trace" not in payload
    assert "reference_video_path" not in payload["tasks"][0]


def test_public_demo_starts_stream_only_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
        ]
    )
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, config) -> None:
            captured["record_artifacts"] = config.record_artifacts
            self.output_dir = config.output_dir

        def set_frame_callback(self, callback) -> None:
            captured["callback"] = callback

        def reset(self) -> dict[str, object]:
            return {"state": "ready"}

        def close(self) -> None:
            pass

    monkeypatch.setattr(teleop_server, "Task3TeleopSession", FakeSession)
    runtime = OperatorRuntime(args)
    monkeypatch.setattr(runtime, "ensure_scene_annotation", lambda task, episode: None)
    try:
        status = runtime.start_session(
            {"episode_id": "stack_blocks_three__000000", "frame_id": 0},
            frame_callback=None,
        )
    finally:
        runtime.shutdown()

    assert status["state"] == "ready"
    assert captured["record_artifacts"] is False


def test_public_demo_reuses_prewarmed_default_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
            "--prewarm-default",
        ]
    )
    captured: dict[str, object] = {}

    class FakeConfig:
        episode_id = "stack_blocks_three__000000"
        frame_id = 0

    class FakeSession:
        config = FakeConfig()
        command_log: list[dict[str, object]] = []
        output_dir = tmp_path / "warm-output"

        def set_frame_callback(self, callback) -> None:
            captured["callback"] = callback

        def emit_frame(self) -> None:
            captured["emitted"] = True

        def status(self) -> dict[str, object]:
            return {"state": "ready", "command_count": 0}

        def close(self) -> None:
            captured["closed"] = True

    warm_session = FakeSession()
    runtime = OperatorRuntime(args)
    monkeypatch.setattr(
        runtime,
        "_build_session",
        lambda payload, frame_callback: (warm_session, {"state": "ready"}, None),
    )
    forwarded_frames: list[object] = []
    callback = lambda frame: forwarded_frames.append(frame)
    monkeypatch.setattr(teleop_server, "encode_jpeg", lambda frame, quality=92: b"jpeg")
    try:
        future = runtime.prewarm_default_session()
        assert future is not None
        future.result(timeout=2)
        status = runtime.start_session(
            {
                "episode_id": "stack_blocks_three__000000",
                "frame_id": 0,
                "client_session_id": "test-client",
            },
            frame_callback=callback,
        )
        forwarded_frame = object()
        captured["callback"](forwarded_frame)
    finally:
        runtime.shutdown()

    assert status["state"] == "ready"
    assert status["prewarmed"] is True
    assert status["client_session_id"] == "test-client"
    assert captured["callback"] is not callback
    assert forwarded_frames == [forwarded_frame]
    assert runtime._model_initial_jpeg == b"jpeg"
    assert captured["emitted"] is True


def test_task1_cached_triplet_bypasses_live_simulator_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    generator = tmp_path / "generate_task1_triplet.py"
    generator.write_text("# cached test fixture\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--task1-generator-script",
            str(generator),
            "--public-demo",
        ]
    )
    runtime = OperatorRuntime(args)
    source_dir = tmp_path / "generated"
    web_dir = source_dir / "web"
    web_dir.mkdir(parents=True)
    for variant in teleop_server.TASK1_VARIANTS:
        (web_dir / f"{variant}.mp4").write_bytes(f"video-{variant}".encode())
    result = {
        "task_success": {variant: True for variant in teleop_server.TASK1_VARIANTS},
        "all_task_success": True,
        "small_response_mse": 1.0,
        "large_response_mse": 2.0,
        "oracle_ratio": 0.5,
        "response_order_pass": True,
        "valid_probe": True,
        "elapsed_sec": 12.0,
    }
    monkeypatch.setattr(
        runtime,
        "_task1_demo_payload",
        lambda demo_id, demo: {
            "available": True,
            "task_label": "Adjust Bottle",
            "instruction": "Adjust the bottle.",
        },
    )
    try:
        demo_id = teleop_server.TASK1_DEFAULT_DEMO_ID
        demo = teleop_server.TASK1_DEMOS[demo_id]
        cache_dir = runtime._store_task1_cache(demo_id, demo, 0.14, result, source_dir)
        runtime._task1_jobs["generated-job"] = {
            "output_dir": str(cache_dir),
            "cache_hit": False,
        }
        runtime._active_client = object()
        status = runtime.start_task1_job(0.14)
        runtime._active_client = None
    finally:
        runtime.shutdown()

    assert status["state"] == "ready"
    assert status["cache_hit"] is True
    assert status["result"] == result
    assert runtime._task1_running is False
    assert all((cache_dir / "web" / f"{variant}.mp4").is_file() for variant in teleop_server.TASK1_VARIANTS)


def test_public_scene_inventory_is_compact_and_cached(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    monkeypatch.setattr(teleop_server, "SELECTED_SCENE_TASKS", ("stack_blocks_three",))
    monkeypatch.setattr(teleop_server, "SCENE_EPISODES", (0,))
    robotwin_root = tmp_path / "RoboTwin"
    scene_root = robotwin_root / "data" / "stack_blocks_three" / "cross_clean_50"
    for subdirectory in ("data", "instructions", "video"):
        (scene_root / subdirectory).mkdir(parents=True, exist_ok=True)
    (scene_root / "data" / "episode0.hdf5").touch()
    instruction_path = scene_root / "instructions" / "episode0.json"
    instruction_path.write_text(json.dumps({"seen": ["Stack the blocks."]}), encoding="utf-8")
    (scene_root / "video" / "episode0.mp4").touch()
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(robotwin_root),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
        ]
    )
    runtime = OperatorRuntime(args)
    monkeypatch.setattr(
        runtime,
        "load_scene_action_count",
        lambda *_: (_ for _ in ()).throw(AssertionError("public inventory should not open HDF5")),
    )
    monkeypatch.setattr(
        runtime,
        "load_reference_video_info",
        lambda *_: (_ for _ in ()).throw(AssertionError("public inventory should not inspect video metadata")),
    )
    try:
        first = runtime.scene_inventory()
        instruction_path.write_text(json.dumps({"seen": ["Changed."]}), encoding="utf-8")
        second = runtime.scene_inventory()
    finally:
        runtime.shutdown()

    episode = first["tasks"][0]["episodes"][0]
    assert first is second
    assert episode == {
        "episode": 0,
        "episode_id": "stack_blocks_three__000000",
        "default_frame": 0,
        "instruction": "Stack the blocks.",
        "data_exists": True,
    }


def test_public_scene_inventory_limits_each_task_to_five_valid_episodes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    monkeypatch.setattr(teleop_server, "SELECTED_SCENE_TASKS", ("stack_blocks_three",))
    monkeypatch.setattr(teleop_server, "SCENE_EPISODES", tuple(range(8)))
    robotwin_root = tmp_path / "RoboTwin"
    scene_root = robotwin_root / "data" / "stack_blocks_three" / "cross_clean_50"
    for episode in (0, 2, 3, 4, 5, 6, 7):
        for subdirectory in ("data", "instructions", "video"):
            (scene_root / subdirectory).mkdir(parents=True, exist_ok=True)
        (scene_root / "data" / f"episode{episode}.hdf5").touch()
        (scene_root / "instructions" / f"episode{episode}.json").write_text(
            json.dumps({"seen": [f"Episode {episode}."]}),
            encoding="utf-8",
        )
        (scene_root / "video" / f"episode{episode}.mp4").touch()
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(robotwin_root),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
        ]
    )
    runtime = OperatorRuntime(args)
    try:
        episodes = runtime.scene_inventory()["tasks"][0]["episodes"]
    finally:
        runtime.shutdown()

    assert [row["episode"] for row in episodes] == [0, 2, 3, 4, 5]
    assert all(row["data_exists"] for row in episodes)


def test_task3_model_selection_is_explicit() -> None:
    assert validate_task3_model("lingbot-va") == "lingbot-va"
    assert validate_task3_model("CTRL-WORLD") == "ctrl-world"
    for value in (None, "", "cosmos-3-nano", "../../worker"):
        try:
            validate_task3_model(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Task 3 model accepted: {value}")


def test_live_capture_renders_only_head_camera() -> None:
    import numpy as np

    class FakeCamera:
        def __init__(self, value: float) -> None:
            self.value = value
            self.take_picture_calls = 0
            self.get_picture_calls = 0

        def take_picture(self) -> None:
            self.take_picture_calls += 1

        def get_picture(self, texture: str) -> np.ndarray:
            assert texture == "Color"
            self.get_picture_calls += 1
            return np.full((2, 3, 4), self.value, dtype=np.float32)

    side_camera = FakeCamera(0.1)
    head_camera = FakeCamera(0.5)

    class FakeCameraManager:
        static_camera_list = [side_camera, head_camera]
        static_camera_name = ["side_camera", "head_camera"]

        def update_picture(self) -> None:
            raise AssertionError("full camera render should not run")

        def get_rgb(self) -> dict:
            raise AssertionError("all-camera readback should not run")

    rgb = capture_head_camera_rgb(FakeCameraManager())

    assert rgb.shape == (2, 3, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == 127)
    assert side_camera.take_picture_calls == 0
    assert side_camera.get_picture_calls == 0
    assert head_camera.take_picture_calls == 1
    assert head_camera.get_picture_calls == 1


def test_operator_rejects_unlisted_scene() -> None:
    assert validate_scene_selection("stack_blocks_three", 0) == ("stack_blocks_three", 0)
    try:
        validate_scene_selection("../../private", 0)
    except ValueError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("unlisted scene should be rejected")


def test_task1_live_magnitude_is_bounded() -> None:
    demo = teleop_server.TASK1_DEMOS[teleop_server.TASK1_DEFAULT_DEMO_ID]
    assert validate_task1_magnitude(0.14, demo) == 0.14
    assert validate_task1_magnitude("0.02", demo) == 0.02
    for value in (0.019, 0.181, float("nan"), "not-a-number"):
        try:
            validate_task1_magnitude(value, demo)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Task 1 magnitude accepted: {value}")


def test_task1_live_is_disabled_without_generator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOTWIN_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK3_ANNOTATION_ROOT", raising=False)
    monkeypatch.delenv("WORLDSIMPROBE_TASK1_GENERATOR", raising=False)
    args = build_parser().parse_args(
        [
            "--robotwin-root",
            str(tmp_path / "RoboTwin"),
            "--annotation-root",
            str(tmp_path / "annotations"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--public-demo",
        ]
    )
    runtime = OperatorRuntime(args)
    try:
        config = runtime.task1_demo_config()
    finally:
        runtime.shutdown()

    assert config["available"] is False
    assert config["task"] == "adjust_bottle"
    assert config["magnitude"]["small_ratio"] == 0.25


def test_operator_html_is_packaged() -> None:
    page = static_root() / "operator.html"
    text = page.read_text(encoding="utf-8")
    assert "WorldSimProbe Task 3 Operator" in text
    assert 'id="referenceVideo"' in text
    assert 'data-arm-toggle="left"' in text
    assert 'data-arm-toggle="right"' in text
    controls_index = text.index('<section class="panel controls-panel">')
    lower_row_index = text.index('<div class="lower-row">')
    assert lower_row_index < controls_index
    lower_row = text[lower_row_index:controls_index]
    assert lower_row.index("<h2>Session</h2>") < lower_row.index("<h2>Current Action</h2>")


def test_operator_http_supports_keep_alive() -> None:
    assert teleop_server.Task3RequestHandler.protocol_version == "HTTP/1.1"


def test_trace_validator_accepts_realized_gripper_command(tmp_path: Path) -> None:
    before_left = [0.0] * 6 + [1.0]
    after_left = [0.0] * 6 + [0.8]
    right = [0.0] * 7
    trace = {
        "type": "robotwin_simulator_left_controls_replay",
        "task": "example_task",
        "episode_id": "example_task__000000",
        "control_mode": "calibrated",
        "commands": ["close"],
        "right_locked_action": right,
        "command_log": [
            {
                "index": 1,
                "command": "close",
                "arm": "left",
                "before_left": before_left,
                "after_left": after_left,
                "before_right": right,
                "after_right": right,
                "right_action": right,
            }
        ],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    result = evaluate_task3_trace(trace_path, expected_commands=["close"])

    assert result["summary"]["trace_contract_pass"] == 1
    assert result["summary"]["gripper_accuracy"] == 1.0
    assert result["summary"]["inactive_arm_lock_accuracy"] == 1.0
