#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

DISALLOWED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5"}
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
TOKEN_PATTERN = re.compile(r"\b(?:ms|hf)_[A-Za-z0-9_-]{20,}\b|\bms-[A-Za-z0-9-]{20,}\b")
FORBIDDEN_TEXT = (
    "/" + "mnt" + "/workspace",
    "/" + "data" + "/shared",
    "116" + ".198" + ".70" + ".4",
    "120" + ".220" + ".142",
)
FORBIDDEN_BRANDING = (
    "world" + "simbench",
    "act" + "-bench",
    "act" + "_" + "bench",
)
FORBIDDEN_TERMS = re.compile(
    rf"\b(?:{'|'.join(('h' + '100', 'h' + '200'))})\b|"
    rf"\b{'suite'}\s*[1-5]\b",
    flags=re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a WorldSimProbe public release tree.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-mib", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []
    checked = 0
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        if any(part.endswith(".egg-info") for part in path.parts):
            continue
        if not path.is_file():
            continue
        checked += 1
        relative = path.relative_to(root)
        relative_lower = str(relative).lower()
        for fragment in FORBIDDEN_BRANDING:
            if fragment in relative_lower:
                errors.append(f"legacy branding in path: {relative}")
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            errors.append(f"disallowed artifact: {relative}")
        if path.stat().st_size > args.max_file_mib * 1024 * 1024:
            errors.append(f"file exceeds public size limit: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for fragment in FORBIDDEN_TEXT:
            if fragment in text:
                errors.append(f"internal path or host in {relative}: {fragment}")
        text_lower = text.lower()
        for fragment in FORBIDDEN_BRANDING:
            if fragment in text_lower:
                errors.append(f"legacy branding in {relative}: {fragment}")
        if FORBIDDEN_TERMS.search(text):
            errors.append(f"internal hardware or legacy task term in {relative}")
        if TOKEN_PATTERN.search(text):
            errors.append(f"possible access token in {relative}")

    print(f"checked_files={checked}")
    print(f"errors={len(errors)}")
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
