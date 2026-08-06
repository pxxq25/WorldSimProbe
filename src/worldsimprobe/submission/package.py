from __future__ import annotations

import zipfile
from pathlib import Path

from worldsimprobe.common.manifest import read_jsonl
from worldsimprobe.submission.validator import resolve_relative_video, validate_submission
from worldsimprobe.submission.video_config import VideoTimingConfig


def package_submission(
    manifest: Path,
    root: Path,
    output: Path,
    *,
    decode: bool = True,
    video_config: VideoTimingConfig | None = None,
) -> dict:
    report = validate_submission(manifest, root, decode=decode, video_config=video_config)
    if not report["passes"]:
        raise ValueError(f"submission validation failed for {report['error_rows']} rows")

    paths = {manifest.resolve()}
    for row in read_jsonl(manifest):
        for value in row["videos"].values():
            paths.add(resolve_relative_video(root, value))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(manifest, "submission.jsonl")
        for path in sorted(paths - {manifest.resolve()}):
            archive.write(path, path.relative_to(root.resolve()).as_posix())

    return {
        "output": str(output),
        "rows": report["rows"],
        "files": len(paths),
        "bytes": output.stat().st_size,
    }
