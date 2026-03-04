from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class DataHealthReport:
    run_id: str
    run_timestamp_utc: str
    universe: str
    tickers_requested: int
    tickers_ok: int
    tickers_partial: int
    tickers_failed: int
    rate_limited: bool
    top_missing_fields: list[dict[str, Any]]
    errors_sample: list[str]
    cache_path: str
    cache_written: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_missing_field_stats(df: pd.DataFrame, core_fields: Iterable[str]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for col in core_fields:
        if col not in df.columns:
            stats.append({"field": col, "missing_count": None, "reason": "column_missing"})
            continue
        missing = int(df[col].isna().sum())
        stats.append({"field": col, "missing_count": missing})
    stats.sort(key=lambda x: (x["missing_count"] is None, -(x["missing_count"] or 0), x["field"]))
    return stats


def summarize_refresh_outcome(
    df: pd.DataFrame,
    requested_count: int,
    success_count: int,
    failure_count: int,
    rate_limited: bool,
    errors_sample: list[str],
    *,
    universe: str,
    cache_path: str,
    cache_written: bool,
    notes: str,
    core_fields: Iterable[str] | None = None,
    run_id: str | None = None,
) -> DataHealthReport:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    core = list(core_fields) if core_fields is not None else [
        "Close",
        "MarketCap",
        "Revenue_Growth_YoY_Pct",
        "Earnings_Growth_Pct",
        "EBITDA_Margin",
        "ROE",
        "PE_Ratio",
        "PEG_Ratio",
        "Rule_of_40",
    ]
    top_missing = build_missing_field_stats(df, core)

    tickers_ok = int(success_count)
    tickers_failed = int(failure_count)
    remaining = max(0, requested_count - tickers_ok - tickers_failed)
    tickers_partial = int(remaining)

    return DataHealthReport(
        run_id=run_id,
        run_timestamp_utc=_utc_now_iso(),
        universe=universe,
        tickers_requested=int(requested_count),
        tickers_ok=tickers_ok,
        tickers_partial=tickers_partial,
        tickers_failed=tickers_failed,
        rate_limited=bool(rate_limited),
        top_missing_fields=top_missing[:25],
        errors_sample=list(errors_sample)[:25],
        cache_path=cache_path,
        cache_written=bool(cache_written),
        notes=str(notes),
    )


def write_health_report_json(report: DataHealthReport, path: str) -> None:
    payload = report.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
