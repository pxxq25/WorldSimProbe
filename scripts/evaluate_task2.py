#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from worldsimprobe.common.join import join_references_with_submission
from worldsimprobe.common.manifest import write_jsonl
from worldsimprobe.evaluation.task2_action_source.robotseg_flow import (
    evaluate_task2_manifest_robotseg_flow,
)
from worldsimprobe.submission.preflight import validate_joined_submission
from worldsimprobe.submission.validator import require_valid_submission
from worldsimprobe.submission.video_config import load_video_timing_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WorldSimProbe Task 2.")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robotseg-root", required=True)
    parser.add_argument("--robotseg-checkpoint", required=True)
    parser.add_argument("--flow-model", choices=["farneback", "dpflow"], default="dpflow")
    parser.add_argument("--dpflow-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video-config", type=Path)
    args = parser.parse_args()
    video_config = load_video_timing_config(args.video_config)

    require_valid_submission(
        args.submission,
        args.submission_root,
        video_config=video_config,
    )
    rows = join_references_with_submission(
        args.references,
        args.submission,
        args.submission_root,
        task_id="task2",
        video_config=video_config,
    )
    validate_joined_submission(rows, video_config=video_config)
    with tempfile.TemporaryDirectory(prefix="worldsimprobe_task2_") as temporary:
        manifest = Path(temporary) / "joined.jsonl"
        write_jsonl(manifest, rows)
        result = evaluate_task2_manifest_robotseg_flow(
            manifest,
            args.reference_root,
            candidate_mode="output",
            camera=args.camera,
            limit=args.limit,
            robotseg_root=args.robotseg_root,
            robotseg_checkpoint=args.robotseg_checkpoint,
            flow_model=args.flow_model,
            dpflow_ckpt=args.dpflow_checkpoint or "checkpoints/dpflow-things-2012b5d6.ckpt",
            device=args.device,
        )
    result["task_id"] = "task2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["rows_scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
