from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from worldsimprobe.submission.package import package_submission
from worldsimprobe.submission.validator import validate_submission
from worldsimprobe.submission.video_config import load_video_timing_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldsimprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-submission")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--root", required=True, type=Path)
    validate.add_argument("--decode", action="store_true")
    validate.add_argument("--video-config", type=Path)

    package = subparsers.add_parser("package-submission")
    package.add_argument("--manifest", required=True, type=Path)
    package.add_argument("--root", required=True, type=Path)
    package.add_argument("--output", required=True, type=Path)
    package.add_argument("--skip-decode", action="store_true")
    package.add_argument("--video-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    video_config = load_video_timing_config(args.video_config)
    if args.command == "validate-submission":
        result = validate_submission(
            args.manifest,
            args.root,
            decode=args.decode,
            video_config=video_config,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passes"] else 1
    if args.command == "package-submission":
        result = package_submission(
            args.manifest,
            args.root,
            args.output,
            decode=not args.skip_decode,
            video_config=video_config,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
