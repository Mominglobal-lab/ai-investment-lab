from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict, Any

import pandas as pd


@dataclass(frozen=True)
class CacheStatus:
    exists: bool
    is_fresh: bool
    age_days: float
    schema_ok: bool
    missing_columns: tuple[str, ...]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _file_age_days(path: str) -> float:
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds / 86400.0


def validate_schema_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> Tuple[bool, tuple[str, ...]]:
    required = list(required_columns)
    missing = tuple([c for c in required if c not in df.columns])
    return (len(missing) == 0), missing


def read_parquet_safe(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        return pd.read_parquet(path), None
    except Exception as e:
        return None, str(e)


def save_parquet_atomic(df: pd.DataFrame, path: str) -> None:
    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def get_cache_status(
    path: str,
    max_age_days: int = 7,
    required_columns: Optional[Iterable[str]] = None,
) -> CacheStatus:
    if not os.path.exists(path):
        return CacheStatus(
            exists=False,
            is_fresh=False,
            age_days=float("inf"),
            schema_ok=False,
            missing_columns=tuple(required_columns or ()),
        )

    age_days = _file_age_days(path)
    is_fresh = age_days < max_age_days

    schema_ok = True
    missing_cols: tuple[str, ...] = tuple()
    if required_columns:
        df, err = read_parquet_safe(path)
        if df is None:
            schema_ok = False
            missing_cols = tuple(required_columns)
        else:
            schema_ok, missing_cols = validate_schema_columns(df, required_columns)

    return CacheStatus(
        exists=True,
        is_fresh=is_fresh,
        age_days=age_days,
        schema_ok=schema_ok,
        missing_columns=missing_cols,
    )


def write_json_report(report: Dict[str, Any], path: str) -> None:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, tuple):
            return [_sanitize(v) for v in value]
        if hasattr(value, "item"):
            try:
                return _sanitize(value.item())
            except Exception:
                pass
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            out = float(value)
            if not math.isfinite(out):
                return None
            return out
        return value

    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(report), f, indent=2, sort_keys=False, allow_nan=False)
    os.replace(tmp_path, path)
