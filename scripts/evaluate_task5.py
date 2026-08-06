#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from worldsimprobe.common.join import join_references_with_submission
from worldsimprobe.common.manifest import write_jsonl
from worldsimprobe.evaluation.task5_interaction_dynamics.evaluator import (
    evaluate_task5_manifest_qwen3_vl_vqa,
)
from worldsimprobe.submission.preflight import validate_joined_submission
from worldsimprobe.submission.validator import require_valid_submission
from worldsimprobe.submission.video_config import load_video_timing_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WorldSimProbe Task 5.")
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/evaluation/task5.yaml", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--sample-count", type=int, default=12)
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
        task_id="task5",
        video_config=video_config,
    )
    validate_joined_submission(rows, video_config=video_config)
    with tempfile.TemporaryDirectory(prefix="worldsimprobe_task5_") as temporary:
        manifest = Path(temporary) / "joined.jsonl"
        write_jsonl(manifest, rows)
        result = evaluate_task5_manifest_qwen3_vl_vqa(
            manifest=manifest,
            root=args.reference_root,
            task_config=args.config,
            model_name_or_path=args.model,
            candidate_mode="candidate",
            sample_count=args.sample_count,
            suffix_only=False,
            limit=args.limit,
            device_map=args.device_map,
            dtype_name=args.dtype,
        )
    result["task_id"] = "task5"
    result["oracle_filter_applied"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["rows_scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
