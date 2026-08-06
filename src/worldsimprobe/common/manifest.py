from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRow:
    raw: dict[str, Any]

    @property
    def task_id(self) -> str | None:
        return self.raw.get("worldsimprobe_task_id") or self.raw.get("task_id")

    @property
    def task(self) -> str | None:
        return self.raw.get("task") or self.raw.get("receiver_task") or self.raw.get("source_task")

    @property
    def episode(self) -> int | None:
        value = self.raw.get("episode")
        return int(value) if value is not None else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
