from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data_pipeline.cache_manager import (
    ensure_parent_dir,
    get_cache_status,
    read_parquet_safe,
    save_parquet_atomic,
    validate_schema_columns,
    write_json_report,
)
from data_pipeline.data_fetcher import SCHEMA_COLUMNS, fetch_sp500_tickers, refresh_fundamentals_yfinance
from data_pipeline.data_health_report import summarize_refresh_outcome


@dataclass(frozen=True)
class PipelineRunResult:
    data: pd.DataFrame
    wrote_cache: bool
    cache_path: str
    health_report_path: str
    reason: str


def run_stock_fundamentals_pipeline(
    *,
    cache_path: str = "data/fundamentals_cache.parquet",
    max_age_days: float = 7.0,
    universe: str = "S&P 500",
    tickers: Optional[list[str]] = None,
    min_refresh_success_ratio: float = 0.25,
    health_report_path: str = "data/fundamentals_health_report.json",
) -> PipelineRunResult:
    status = get_cache_status(cache_path, max_age_days, required_columns=SCHEMA_COLUMNS)

    cached_df = None
    if status.exists and status.schema_ok and status.is_fresh:
        cached_df, _err = read_parquet_safe(cache_path)
        if cached_df is not None:
            return PipelineRunResult(
                data=cached_df,
                wrote_cache=False,
                cache_path=cache_path,
                health_report_path=health_report_path,
                reason="used fresh cache",
            )

    if tickers is None:
        tickers = fetch_sp500_tickers()

    refresh = refresh_fundamentals_yfinance(tickers)

    df = refresh.data.copy()
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    eligible = True
    reason = "cache updated"
    schema_ok, missing_cols = validate_schema_columns(df, SCHEMA_COLUMNS)
    if not schema_ok:
        eligible = False
        reason = f"missing required columns: {', '.join(missing_cols)}"
    elif df.empty:
        eligible = False
        reason = "no rows returned from refresh"
    elif refresh.requested_count > 0:
        ratio = refresh.success_count / refresh.requested_count
        if ratio < min_refresh_success_ratio:
            eligible = False
            reason = f"success ratio {ratio:.1%} below threshold {min_refresh_success_ratio:.0%}"

    wrote_cache = False
    if eligible:
        ensure_parent_dir(cache_path)
        save_parquet_atomic(df, cache_path)
        wrote_cache = True

    ensure_parent_dir(health_report_path)
    report = summarize_refresh_outcome(
        df=df,
        requested_count=refresh.requested_count,
        success_count=refresh.success_count,
        failure_count=refresh.failure_count,
        rate_limited=refresh.rate_limited,
        errors_sample=refresh.errors_sample,
        universe=universe,
        cache_path=cache_path,
        cache_written=wrote_cache,
        notes=reason,
    )
    write_json_report(report.to_dict(), health_report_path)

    if not wrote_cache and status.exists:
        stale_df, _err = read_parquet_safe(cache_path)
        if stale_df is not None and not stale_df.empty:
            return PipelineRunResult(
                data=stale_df,
                wrote_cache=False,
                cache_path=cache_path,
                health_report_path=health_report_path,
                reason=f"refresh not eligible: {reason}. used stale cache",
            )

    return PipelineRunResult(
        data=df,
        wrote_cache=wrote_cache,
        cache_path=cache_path,
        health_report_path=health_report_path,
        reason=reason,
    )
