from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def nested_value(row: dict[str, Any], path: tuple[str, ...] | str) -> Any:
    keys = (path,) if isinstance(path, str) else path
    value: Any = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite_values(rows: Iterable[dict[str, Any]], path: tuple[str, ...] | str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = nested_value(row, path)
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.append(numeric)
    return values
