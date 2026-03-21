from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.cache_manager import save_parquet_atomic, write_json_report
from data_pipeline.data_fetcher import FI_SCHEMA_COLUMNS, SCHEMA_COLUMNS
from data_pipeline.run_pipeline import (
    run_decision_models_pipeline,
    run_explainability_pipeline,
    run_fixed_income_pipeline,
    run_monitoring_pipeline,
    run_prices_cache_pipeline,
    run_stock_fundamentals_pipeline,
    run_uncertainty_pipeline,
)


STOCK_TARGETS = {
    "S&P 500": (
        "data/fundamentals_cache_sp500.parquet",
        "data/fundamentals_health_report_sp500.json",
    ),
    "Nasdaq 100": (
        "data/fundamentals_cache_nasdaq100.parquet",
        "data/fundamentals_health_report_nasdaq100.json",
    ),
}

FIXED_INCOME_TARGETS = {
    "US Treasuries": (
        "data/fixed_income_cache_treasury.parquet",
        "data/fixed_income_health_treasury.json",
    ),
    "Bond ETFs": (
        "data/fixed_income_cache_bond_etf.parquet",
        "data/fixed_income_health_bond_etf.json",
    ),
}

TREASURY_SERIES = {
    "DGS10": "10Y",
    "DGS2": "2Y",
    "DGS3MO": "3M",
}

SAVED_PORTFOLIO_ARTIFACTS = (
    (
        "data/portfolio_suggestions_saved.jsonl",
        "data/portfolio_suggestions_health_report.json",
    ),
    (
        "data/portfolio_optimized_saved.jsonl",
        "data/portfolio_optimized_health_report.json",
    ),
)


def _print_result(label: str, result) -> None:
    row_count = int(len(result.data)) if getattr(result, "data", None) is not None else 0
    print(
        f"{label}: rows={row_count} wrote_cache={getattr(result, 'wrote_cache', False)} "
        f"reason={getattr(result, 'reason', 'n/a')}",
        flush=True,
    )


def _combine_frames(frames: list[pd.DataFrame], schema: list[str], key_col: str) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=schema)

    combined = pd.concat(valid, axis=0, ignore_index=True)
    for col in schema:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[schema]
    combined[key_col] = combined[key_col].astype(str).str.upper().str.strip()
    combined = combined.dropna(subset=[key_col])
    combined = combined.drop_duplicates(subset=[key_col], keep="last")
    return combined.sort_values(key_col).reset_index(drop=True)


def _stage_start(name: str) -> float:
    print(f"[START] {name}", flush=True)
    return time.perf_counter()


def _stage_end(name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[DONE] {name} ({elapsed:.1f}s)", flush=True)


def _update_saved_portfolio_artifact_health(
    *,
    artifact_path: str,
    health_path: str,
    max_entries: int,
    stale_after_days: int = 7,
) -> None:
    artifact_file = Path(artifact_path)
    now = pd.Timestamp.utcnow().tz_localize(None)
    payloads: list[dict] = []
    parse_errors = 0

    if artifact_file.exists():
        raw_lines = [line.strip() for line in artifact_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in raw_lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    payloads.append(row)
                else:
                    parse_errors += 1
            except Exception:
                parse_errors += 1

        if max_entries > 0 and len(payloads) > max_entries:
            payloads = payloads[-max_entries:]
            artifact_file.write_text("".join(json.dumps(p) + "\n" for p in payloads), encoding="utf-8")

    saved_times: list[pd.Timestamp] = []
    for row in payloads:
        raw = row.get("saved_at_utc")
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.notna(ts):
            saved_times.append(pd.Timestamp(ts).tz_convert(None))

    latest_saved_at = max(saved_times).isoformat() if saved_times else None
    cutoff = now - pd.Timedelta(days=max(stale_after_days, 1))
    fresh_count = int(sum(1 for ts in saved_times if ts >= cutoff))
    stale_count = int(max(len(payloads) - fresh_count, 0))

    write_json_report(
        {
            "run_timestamp": pd.Timestamp.utcnow().isoformat(),
            "artifact_path": artifact_path,
            "exists": artifact_file.exists(),
            "entry_count": int(len(payloads)),
            "latest_saved_at_utc": latest_saved_at,
            "fresh_count": fresh_count,
            "stale_count": stale_count,
            "parse_error_count": int(parse_errors),
            "file_size_bytes": int(artifact_file.stat().st_size) if artifact_file.exists() else 0,
            "retention_max_entries": int(max_entries),
        },
        health_path,
    )


def refresh_saved_portfolio_artifacts(*, max_entries: int = 500) -> None:
    for artifact_path, health_path in SAVED_PORTFOLIO_ARTIFACTS:
        _update_saved_portfolio_artifact_health(
            artifact_path=artifact_path,
            health_path=health_path,
            max_entries=max_entries,
        )
        print(f"portfolio_artifact: path={artifact_path} max_entries={max_entries}", flush=True)


def refresh_treasury_yields(*, lookback_years: int = 10) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.DateOffset(years=max(int(lookback_years), 1))

    frames: list[pd.DataFrame] = []
    for series_id, column_name in TREASURY_SERIES.items():
        print(f"treasury: fetching {series_id}", flush=True)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        if frame.empty:
            raise ValueError(f"FRED response for {series_id} was empty")

        cols = {str(c).strip().lower(): c for c in frame.columns}
        date_col = cols.get("date") or cols.get("observation_date")
        value_col = cols.get(series_id.lower())
        if date_col is None or value_col is None:
            raise ValueError(
                f"FRED response for {series_id} missing expected columns. "
                f"Found columns: {list(frame.columns)}"
            )

        frame = frame.rename(columns={date_col: "Date", value_col: column_name})
        frame = frame[["Date", column_name]]
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        frame = frame[(frame["Date"] >= start) & (frame["Date"] <= end)]
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Date", how="outer")
    merged = merged.sort_values("Date").reset_index(drop=True)
    save_parquet_atomic(merged, "data/treasury_yields_cache.parquet")
    write_json_report(
        {
            "run_timestamp": pd.Timestamp.utcnow().isoformat(),
            "cache_path": "data/treasury_yields_cache.parquet",
            "row_count": int(len(merged)),
            "columns": list(merged.columns),
        },
        "data/treasury_yields_health_report.json",
    )
    print(f"treasury: rows={len(merged)} path=data/treasury_yields_cache.parquet")
    return merged


def _write_union_cache(
    *,
    frames: list[pd.DataFrame],
    schema: list[str],
    key_col: str,
    cache_path: str,
    health_path: str,
    source_labels: list[str],
) -> pd.DataFrame:
    combined = _combine_frames(frames, schema, key_col)
    save_parquet_atomic(combined, cache_path)
    write_json_report(
        {
            "run_timestamp": pd.Timestamp.utcnow().isoformat(),
            "cache_path": cache_path,
            "source_labels": source_labels,
            "row_count": int(len(combined)),
            "schema_ok": list(combined.columns) == list(schema),
        },
        health_path,
    )
    return combined


def refresh_stock_universes(universes: list[str], max_age_days: float) -> None:
    frames: list[pd.DataFrame] = []
    labels: list[str] = []

    for universe in universes:
        cache_path, health_path = STOCK_TARGETS[universe]
        result = run_stock_fundamentals_pipeline(
            cache_path=cache_path,
            health_report_path=health_path,
            max_age_days=max_age_days,
            universe=universe,
        )
        _print_result(f"stock:{universe}", result)
        if result.data is not None and not result.data.empty:
            frames.append(result.data)
            labels.append(universe)

    combined = _write_union_cache(
        frames=frames,
        schema=SCHEMA_COLUMNS,
        key_col="Ticker",
        cache_path="data/fundamentals_cache.parquet",
        health_path="data/fundamentals_health_report.json",
        source_labels=labels,
    )
    print(f"stock:combined rows={len(combined)} path=data/fundamentals_cache.parquet")


def refresh_fixed_income_universes(universes: list[str], max_age_days: float) -> None:
    frames: list[pd.DataFrame] = []
    labels: list[str] = []

    for universe in universes:
        cache_path, health_path = FIXED_INCOME_TARGETS[universe]
        result = run_fixed_income_pipeline(
            cache_path=cache_path,
            health_report_path=health_path,
            max_age_days=max_age_days,
            universe=universe,
        )
        _print_result(f"fixed_income:{universe}", result)
        if result.data is not None and not result.data.empty:
            frames.append(result.data)
            labels.append(universe)

    combined = _write_union_cache(
        frames=frames,
        schema=FI_SCHEMA_COLUMNS,
        key_col="Symbol",
        cache_path="data/fixed_income_cache.parquet",
        health_path="data/fixed_income_health_report.json",
        source_labels=labels,
    )
    print(f"fixed_income:combined rows={len(combined)} path=data/fixed_income_cache.parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the scheduled cache and artifact refresh workflow.")
    parser.add_argument(
        "--stock-universe",
        action="append",
        choices=sorted(STOCK_TARGETS.keys()),
        dest="stock_universes",
        help="Stock universes to refresh. Defaults to all supported stock universes.",
    )
    parser.add_argument(
        "--fi-universe",
        action="append",
        choices=sorted(FIXED_INCOME_TARGETS.keys()),
        dest="fi_universes",
        help="Fixed-income universes to refresh. Defaults to all supported fixed-income universes.",
    )
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker for prices and artifact builds.")
    parser.add_argument("--max-age-days", type=float, default=0.0, help="Freshness threshold passed into pipeline functions.")
    parser.add_argument("--skip-stock", action="store_true", help="Skip rebuilding stock fundamentals caches.")
    parser.add_argument("--skip-fixed-income", action="store_true", help="Skip rebuilding fixed-income caches.")
    parser.add_argument("--skip-treasury", action="store_true", help="Skip rebuilding the treasury yields cache.")
    parser.add_argument("--skip-prices", action="store_true", help="Skip rebuilding the shared prices cache.")
    parser.add_argument("--skip-models", action="store_true", help="Skip rebuilding decision model artifacts.")
    parser.add_argument("--skip-explainability", action="store_true", help="Skip rebuilding explainability artifacts.")
    parser.add_argument("--skip-uncertainty", action="store_true", help="Skip rebuilding uncertainty artifacts.")
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip rebuilding monitoring artifacts.")
    parser.add_argument("--skip-portfolio-artifacts", action="store_true", help="Skip updating saved portfolio artifact health/retention.")
    parser.add_argument("--portfolio-max-entries", type=int, default=500, help="Maximum saved rows to retain in each saved portfolio JSONL artifact.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    stock_universes = args.stock_universes or list(STOCK_TARGETS.keys())
    fi_universes = args.fi_universes or list(FIXED_INCOME_TARGETS.keys())

    if not args.skip_stock:
        started_at = _stage_start("stock refresh")
        refresh_stock_universes(stock_universes, max_age_days=float(args.max_age_days))
        _stage_end("stock refresh", started_at)
    else:
        print("[SKIP] stock refresh", flush=True)

    if not args.skip_fixed_income:
        started_at = _stage_start("fixed-income refresh")
        refresh_fixed_income_universes(fi_universes, max_age_days=float(args.max_age_days))
        _stage_end("fixed-income refresh", started_at)
    else:
        print("[SKIP] fixed-income refresh", flush=True)

    if not args.skip_treasury:
        started_at = _stage_start("treasury refresh")
        refresh_treasury_yields()
        _stage_end("treasury refresh", started_at)

    if not args.skip_prices:
        started_at = _stage_start("prices refresh")
        prices_result = run_prices_cache_pipeline(
            prices_cache_path="data/prices_cache.parquet",
            health_report_path="data/prices_health_report.json",
            fundamentals_cache_paths=[
                "data/fundamentals_cache.parquet",
                "data/fundamentals_cache_sp500.parquet",
                "data/fundamentals_cache_nasdaq100.parquet",
            ],
            benchmark_ticker=str(args.benchmark).strip().upper(),
            max_age_days=float(args.max_age_days),
        )
        _print_result("prices", prices_result)
        _stage_end("prices refresh", started_at)

    needs_models = not args.skip_models
    needs_explainability = not args.skip_explainability
    needs_uncertainty = not args.skip_uncertainty
    needs_monitoring = not args.skip_monitoring

    if needs_models or needs_explainability or needs_uncertainty or needs_monitoring:
        started_at = _stage_start("model artifacts")
        model_result = run_decision_models_pipeline(benchmark_ticker=str(args.benchmark).strip().upper())
        _print_result("models", model_result)
        _stage_end("model artifacts", started_at)

    if needs_explainability:
        started_at = _stage_start("explainability artifacts")
        explain_result = run_explainability_pipeline(benchmark_ticker=str(args.benchmark).strip().upper())
        _print_result("explainability", explain_result)
        _stage_end("explainability artifacts", started_at)

    if needs_uncertainty:
        started_at = _stage_start("uncertainty artifacts")
        uncertainty_result = run_uncertainty_pipeline(benchmark_ticker=str(args.benchmark).strip().upper())
        _print_result("uncertainty", uncertainty_result)
        _stage_end("uncertainty artifacts", started_at)

    if needs_monitoring:
        started_at = _stage_start("monitoring artifacts")
        monitoring_result = run_monitoring_pipeline(benchmark_ticker=str(args.benchmark).strip().upper())
        _print_result("monitoring", monitoring_result)
        _stage_end("monitoring artifacts", started_at)

    if not args.skip_portfolio_artifacts:
        started_at = _stage_start("portfolio artifacts")
        refresh_saved_portfolio_artifacts(max_entries=max(int(args.portfolio_max_entries), 1))
        _stage_end("portfolio artifacts", started_at)

    print("scheduled refresh complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
