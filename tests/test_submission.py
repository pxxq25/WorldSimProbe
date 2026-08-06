import json
import zipfile
from pathlib import Path

from worldsimprobe.submission.package import package_submission
from worldsimprobe.submission.validator import SUBMISSION_SCHEMA, validate_submission
from worldsimprobe.submission.video_config import VideoTimingConfig, load_video_timing_config


def test_public_example_submission_decodes() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "example_submission"
    result = validate_submission(root / "submission.jsonl", root, decode=True)
    assert result["passes"], result["errors"]
    assert result["rows"] == 5
    assert result["video_timing_config"]["fps"] == 10.0


def test_default_video_timing_config_is_10_fps() -> None:
    assert load_video_timing_config().fps == 10.0


def test_decoded_fps_must_match_evaluator_config() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "example_submission"
    result = validate_submission(
        root / "submission.jsonl",
        root,
        decode=True,
        video_config=VideoTimingConfig(fps=9.0, fps_tolerance=0.01),
    )
    assert not result["passes"]
    assert "does not match configured fps 9" in json.dumps(result)


def test_official_evaluation_accepts_one_prediction_per_sample(tmp_path: Path) -> None:
    (tmp_path / "candidate.mp4").touch()
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "sample_id": "row-1",
                "task_id": "task2",
                "model_id": "model",
                "videos": {"candidate": "candidate.mp4"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_submission(manifest, tmp_path)
    assert result["passes"], result["errors"]


def test_documented_and_packaged_schemas_match() -> None:
    path = Path(__file__).resolve().parents[1] / "schemas" / "submission.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == SUBMISSION_SCHEMA


def test_submission_rejects_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        '{"schema_version":"1.0","sample_id":"x","task_id":"task2",'
        '"model_id":"m","videos":{"candidate":"../outside.mp4"}}\n'
    )
    result = validate_submission(manifest, tmp_path)
    assert not result["passes"]
    assert "escapes" in json.dumps(result)


def test_submission_rejects_schema_violation(tmp_path: Path) -> None:
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "0.9",
                "sample_id": "row-1",
                "task_id": "task2",
                "model_id": "model",
                "videos": {"candidate": "videos/row-1.mp4"},
                "unexpected": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_submission(manifest, tmp_path)
    assert not result["passes"]
    rendered = json.dumps(result)
    assert "schema_version" in rendered
    assert "unexpected" in rendered


def test_submission_rejects_participant_video_timing(tmp_path: Path) -> None:
    (tmp_path / "candidate.mp4").touch()
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "sample_id": "row-1",
                "task_id": "task2",
                "model_id": "model",
                "videos": {"candidate": "candidate.mp4"},
                "video_timing": {
                    "candidate": {"fps": 999.0, "frame_timestamps_sec": [0.0, 999.0]}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_submission(manifest, tmp_path)
    assert not result["passes"]
    assert "video_timing" in json.dumps(result)


def test_submission_rejects_obsolete_generations_field(tmp_path: Path) -> None:
    for seed in (11, 22, 33):
        (tmp_path / f"candidate-{seed}.mp4").touch()
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "sample_id": "row-1",
                "task_id": "task2",
                "model_id": "model",
                "generations": [
                    {
                        "seed": seed,
                        "videos": {"candidate": f"candidate-{seed}.mp4"},
                    }
                    for seed in (11, 22, 33)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_submission(manifest, tmp_path)
    assert not result["passes"]
    assert "generations" in json.dumps(result)
    assert "videos" in json.dumps(result)


def test_package_submission_includes_prediction_video(tmp_path: Path) -> None:
    (tmp_path / "candidate.mp4").touch()
    manifest = tmp_path / "submission.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "sample_id": "row-1",
                "task_id": "task2",
                "model_id": "model",
                "videos": {"candidate": "candidate.mp4"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "submission.zip"
    result = package_submission(manifest, tmp_path, output, decode=False)

    assert result["files"] == 2
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "submission.jsonl",
            "candidate.mp4",
        }
