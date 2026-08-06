#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldsimprobe.common.join import join_references_with_submission
from worldsimprobe.evaluation.task1_action_calibration.evaluator import evaluate_task1_rows
from worldsimprobe.submission.preflight import validate_joined_submission
from worldsimprobe.submission.validator import require_valid_submission
from worldsimprobe.submission.video_config import load_video_timing_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WorldSimProbe Task 1.")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        task_id="task1",
        video_config=video_config,
    )
    validate_joined_submission(rows, video_config=video_config)
    result = evaluate_task1_rows(rows, args.submission_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["rows_scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
