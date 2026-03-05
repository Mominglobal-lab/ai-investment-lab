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
from data_pipeline.data_fetcher import SCHEMA_COLUMNS, fetch_universe_tickers, refresh_fundamentals_yfinance
from data_pipeline.data_fetcher import (
    FI_SCHEMA_COLUMNS,
    PRICE_SCHEMA_COLUMNS,
    fetch_fixed_income_universe_instruments,
    refresh_fixed_income_yfinance,
    refresh_prices_yfinance,
)
from data_pipeline.data_health_report import summarize_refresh_outcome
from ai_models.feature_builder import build_feature_table
from ai_models.quality_score_model import run_quality_score_model
from ai_models.regime_detection_model import run_regime_detection_model
from ai_models.risk_detector import run_systemic_risk_detector
from ai_models.explainability_engine import build_quality_explanations
from ai_models.evidence_builder import build_regime_evidence, build_risk_evidence


@dataclass(frozen=True)
class PipelineRunResult:
    data: pd.DataFrame
    wrote_cache: bool
    cache_path: str
    health_report_path: str
    reason: str


@dataclass(frozen=True)
class FixedIncomePipelineRunResult:
    data: pd.DataFrame
    wrote_cache: bool
    cache_path: str
    health_report_path: str
    reason: str


@dataclass(frozen=True)
class PricesPipelineRunResult:
    data: pd.DataFrame
    wrote_cache: bool
    cache_path: str
    health_report_path: str
    reason: str
    requested_count: int
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class ModelPipelineRunResult:
    quality_scores: pd.DataFrame
    regime_signals: pd.DataFrame
    risk_signals: pd.DataFrame
    wrote_artifacts: bool
    reason: str


@dataclass(frozen=True)
class ExplainabilityPipelineRunResult:
    quality_explanations: pd.DataFrame
    regime_evidence: pd.DataFrame
    risk_evidence: pd.DataFrame
    wrote_artifacts: bool
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
        tickers = fetch_universe_tickers(universe)

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


def run_fixed_income_pipeline(
    *,
    cache_path: str = "data/fixed_income_cache.parquet",
    max_age_days: float = 7.0,
    universe: str = "US Treasuries",
    instruments: Optional[list[dict[str, object]]] = None,
    min_refresh_success_ratio: float = 0.25,
    health_report_path: str = "data/fixed_income_health_report.json",
) -> FixedIncomePipelineRunResult:
    status = get_cache_status(cache_path, max_age_days, required_columns=FI_SCHEMA_COLUMNS)

    cached_df = None
    if status.exists and status.schema_ok and status.is_fresh:
        cached_df, _err = read_parquet_safe(cache_path)
        if cached_df is not None:
            return FixedIncomePipelineRunResult(
                data=cached_df,
                wrote_cache=False,
                cache_path=cache_path,
                health_report_path=health_report_path,
                reason="used fresh cache",
            )

    if instruments is None:
        instruments = fetch_fixed_income_universe_instruments(universe)

    refresh = refresh_fixed_income_yfinance(instruments)

    df = refresh.data.copy()
    for col in FI_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[FI_SCHEMA_COLUMNS]

    eligible = True
    reason = "cache updated"
    schema_ok, missing_cols = validate_schema_columns(df, FI_SCHEMA_COLUMNS)
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
        core_fields=("Price", "Yield_Pct", "Duration_Years", "Expense_Ratio_Pct", "AUM"),
    )
    write_json_report(report.to_dict(), health_report_path)

    if not wrote_cache and status.exists:
        stale_df, _err = read_parquet_safe(cache_path)
        if stale_df is not None and not stale_df.empty:
            return FixedIncomePipelineRunResult(
                data=stale_df,
                wrote_cache=False,
                cache_path=cache_path,
                health_report_path=health_report_path,
                reason=f"refresh not eligible: {reason}. used stale cache",
            )

    return FixedIncomePipelineRunResult(
        data=df,
        wrote_cache=wrote_cache,
        cache_path=cache_path,
        health_report_path=health_report_path,
        reason=reason,
    )


def _load_tickers_from_fundamentals_paths(paths: list[str]) -> list[str]:
    tickers: set[str] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        df, _err = read_parquet_safe(path)
        if df is None or df.empty or "Ticker" not in df.columns:
            continue
        vals = df["Ticker"].astype(str).str.upper().str.strip()
        tickers.update([t for t in vals.tolist() if t and t.lower() != "nan"])
    return sorted(tickers)


def run_prices_cache_pipeline(
    *,
    prices_cache_path: str = "data/prices_cache.parquet",
    health_report_path: str = "data/prices_health_report.json",
    fundamentals_cache_paths: Optional[list[str]] = None,
    benchmark_ticker: str = "SPY",
    always_include_benchmarks: Optional[list[str]] = None,
    max_age_days: float = 7.0,
    lookback_years: int = 5,
    min_refresh_success_ratio: float = 0.25,
) -> PricesPipelineRunResult:
    if fundamentals_cache_paths is None:
        fundamentals_cache_paths = [
            "data/fundamentals_cache.parquet",
            "data/fundamentals_cache_sp500.parquet",
            "data/fundamentals_cache_nasdaq100.parquet",
        ]

    status = get_cache_status(prices_cache_path, max_age_days, required_columns=PRICE_SCHEMA_COLUMNS)
    if status.exists and status.schema_ok and status.is_fresh:
        cached_df, _err = read_parquet_safe(prices_cache_path)
        if cached_df is not None:
            return PricesPipelineRunResult(
                data=cached_df,
                wrote_cache=False,
                cache_path=prices_cache_path,
                health_report_path=health_report_path,
                reason="used fresh cache",
                requested_count=0,
                success_count=0,
                failure_count=0,
            )

    tickers = _load_tickers_from_fundamentals_paths(fundamentals_cache_paths)
    if always_include_benchmarks is None:
        always_include_benchmarks = ["SPY", "QQQ", "IWM", "DIA"]
    fixed_benchmarks = [str(t).strip().upper() for t in always_include_benchmarks if str(t).strip()]
    if fixed_benchmarks:
        tickers = sorted(set(tickers + fixed_benchmarks))

    bench = str(benchmark_ticker or "").strip().upper()
    if bench:
        tickers = sorted(set(tickers + [bench]))
    if not tickers:
        raise ValueError("No tickers found in fundamentals cache; run fundamentals refresh first")

    refresh = refresh_prices_yfinance(tickers=tickers, lookback_years=lookback_years)
    df = refresh.data.copy()
    for col in PRICE_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[PRICE_SCHEMA_COLUMNS]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AdjClose"] = pd.to_numeric(df["AdjClose"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    schema_ok, missing_cols = validate_schema_columns(df, PRICE_SCHEMA_COLUMNS)
    wrote_cache = False
    reason = "cache updated"
    if not schema_ok:
        reason = f"missing required columns: {', '.join(missing_cols)}"
    elif df.empty:
        reason = "no rows returned from refresh"
    elif refresh.requested_count > 0 and (refresh.success_count / refresh.requested_count) < min_refresh_success_ratio:
        reason = f"success ratio {(refresh.success_count / refresh.requested_count):.1%} below threshold {min_refresh_success_ratio:.0%}"
    else:
        ensure_parent_dir(prices_cache_path)
        save_parquet_atomic(df, prices_cache_path)
        wrote_cache = True

    date_start = df["Date"].min()
    date_end = df["Date"].max()
    report = {
        "run_timestamp": pd.Timestamp.utcnow().isoformat(),
        "tickers_requested": int(refresh.requested_count),
        "tickers_successfully_fetched": int(refresh.success_count),
        "tickers_failed": int(refresh.failure_count),
        "period_start": date_start.isoformat() if pd.notna(date_start) else None,
        "period_end": date_end.isoformat() if pd.notna(date_end) else None,
        "cache_timestamp": pd.Timestamp.utcnow().isoformat(),
        "wrote_cache": bool(wrote_cache),
        "reason": str(reason),
        "errors_sample": list(refresh.errors_sample or []),
    }
    write_json_report(report, health_report_path)

    if not wrote_cache and status.exists:
        stale_df, _err = read_parquet_safe(prices_cache_path)
        if stale_df is not None and not stale_df.empty:
            return PricesPipelineRunResult(
                data=stale_df,
                wrote_cache=False,
                cache_path=prices_cache_path,
                health_report_path=health_report_path,
                reason=f"refresh not eligible: {reason}. used stale cache",
                requested_count=refresh.requested_count,
                success_count=refresh.success_count,
                failure_count=refresh.failure_count,
            )

    return PricesPipelineRunResult(
        data=df,
        wrote_cache=wrote_cache,
        cache_path=prices_cache_path,
        health_report_path=health_report_path,
        reason=reason,
        requested_count=refresh.requested_count,
        success_count=refresh.success_count,
        failure_count=refresh.failure_count,
    )


def run_pipeline(
    *,
    build_prices_cache: bool = False,
    run_models: bool = False,
    run_explanations: bool = False,
    max_age_days: float = 7.0,
    stock_universe: str = "S&P 500",
    fi_universe: str = "US Treasuries",
    benchmark_ticker: str = "SPY",
) -> dict[str, object]:
    stock_result = run_stock_fundamentals_pipeline(max_age_days=max_age_days, universe=stock_universe)
    fi_result = run_fixed_income_pipeline(max_age_days=max_age_days, universe=fi_universe)
    prices_result = None
    if build_prices_cache:
        prices_result = run_prices_cache_pipeline(
            max_age_days=max_age_days,
            benchmark_ticker=benchmark_ticker,
        )
    model_result = None
    if run_models:
        model_result = run_decision_models_pipeline(benchmark_ticker=benchmark_ticker)
    explain_result = None
    if run_explanations:
        explain_result = run_explainability_pipeline(benchmark_ticker=benchmark_ticker)
    return {
        "stock": stock_result,
        "fixed_income": fi_result,
        "prices": prices_result,
        "models": model_result,
        "explanations": explain_result,
    }


def run_decision_models_pipeline(
    *,
    fundamentals_path: str = "data/fundamentals_cache.parquet",
    prices_path: str = "data/prices_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    quality_out_path: str = "data/quality_scores_cache.parquet",
    regime_out_path: str = "data/regime_cache.parquet",
    risk_out_path: str = "data/risk_signals_cache.parquet",
    model_registry_path: str = "data/model_registry.json",
    model_health_path: str = "data/model_health_report.json",
    benchmark_ticker: str = "SPY",
) -> ModelPipelineRunResult:
    feature_result = build_feature_table(
        fundamentals_path=fundamentals_path,
        prices_path=prices_path,
        treasury_path=treasury_path,
    )
    features = feature_result.features.reset_index()

    quality = run_quality_score_model(features)
    regime = run_regime_detection_model(
        prices_path=prices_path,
        treasury_path=treasury_path,
        benchmark_ticker=benchmark_ticker,
    )
    risk = run_systemic_risk_detector(
        prices_path=prices_path,
        treasury_path=treasury_path,
        benchmark_ticker=benchmark_ticker,
    )

    ensure_parent_dir(quality_out_path)
    ensure_parent_dir(regime_out_path)
    ensure_parent_dir(risk_out_path)
    save_parquet_atomic(quality, quality_out_path)
    save_parquet_atomic(regime, regime_out_path)
    save_parquet_atomic(risk, risk_out_path)

    now = pd.Timestamp.utcnow().isoformat()
    registry = {
        "generated_at": now,
        "models": [
            {
                "model_name": "quality_score_model",
                "model_version": "1.0.0",
                "feature_set_used": [
                    "Revenue_Growth_YoY_Pct",
                    "EBITDA_Margin",
                    "ROE",
                    "FreeCashFlow_Margin",
                    "Volatility_63D",
                    "Drawdown_252D",
                ],
                "training_or_evaluation_window": "cross-sectional latest snapshot",
                "timestamp": now,
                "evaluation_summary": {
                    "rows_scored": int(len(quality)),
                    "score_min": float(quality["QualityScore"].min()) if not quality.empty else None,
                    "score_max": float(quality["QualityScore"].max()) if not quality.empty else None,
                },
            },
            {
                "model_name": "regime_detection_model",
                "model_version": "1.0.0",
                "feature_set_used": [
                    "YieldSlope",
                    "YieldInverted",
                    "YieldVolatility",
                    "Benchmark_Volatility",
                    "Benchmark_Trend",
                ],
                "training_or_evaluation_window": "daily rule-based regime classification",
                "timestamp": now,
                "evaluation_summary": {
                    "rows_scored": int(len(regime)),
                    "latest_regime": str(regime["RegimeLabel"].iloc[-1]) if not regime.empty else None,
                },
            },
            {
                "model_name": "risk_detector",
                "model_version": "1.0.0",
                "feature_set_used": [
                    "volatility_expansion",
                    "drawdown_acceleration",
                    "correlation_spike",
                    "yield_inversion",
                    "rate_shock",
                ],
                "training_or_evaluation_window": "daily rule-based systemic risk scoring",
                "timestamp": now,
                "evaluation_summary": {
                    "rows_scored": int(len(risk)),
                    "latest_risk_level": str(risk["RiskLevel"].iloc[-1]) if not risk.empty else None,
                },
            },
        ],
    }
    write_json_report(registry, model_registry_path)

    fund_status = get_cache_status(fundamentals_path, 365, required_columns=["Ticker"])
    prices_status = get_cache_status(prices_path, 365, required_columns=["Ticker", "Date", "AdjClose"])
    treasury_status = get_cache_status(treasury_path, 365, required_columns=[])
    health = {
        "generated_at": now,
        "model_freshness": {
            "quality_scores_cache": "fresh" if os.path.exists(quality_out_path) else "missing",
            "regime_cache": "fresh" if os.path.exists(regime_out_path) else "missing",
            "risk_signals_cache": "fresh" if os.path.exists(risk_out_path) else "missing",
        },
        "input_cache_coverage": {
            "fundamentals_cache": {"exists": fund_status.exists, "schema_ok": fund_status.schema_ok},
            "prices_cache": {"exists": prices_status.exists, "schema_ok": prices_status.schema_ok},
            "treasury_yields_cache": {"exists": treasury_status.exists},
        },
        "missing_data_warnings": feature_result.warnings,
        "feature_availability_indicators": feature_result.input_coverage,
    }
    write_json_report(health, model_health_path)

    return ModelPipelineRunResult(
        quality_scores=quality,
        regime_signals=regime,
        risk_signals=risk,
        wrote_artifacts=True,
        reason="model artifacts updated",
    )


def run_explainability_pipeline(
    *,
    benchmark_ticker: str = "SPY",
    quality_scores_path: str = "data/quality_scores_cache.parquet",
    regime_cache_path: str = "data/regime_cache.parquet",
    risk_cache_path: str = "data/risk_signals_cache.parquet",
    prices_path: str = "data/prices_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    quality_explain_path: str = "data/quality_explanations_cache.parquet",
    regime_evidence_path: str = "data/regime_evidence_cache.parquet",
    risk_evidence_path: str = "data/risk_evidence_cache.parquet",
    model_registry_path: str = "data/model_registry.json",
    model_health_path: str = "data/model_health_report.json",
) -> ExplainabilityPipelineRunResult:
    feature_result = build_feature_table(
        fundamentals_path="data/fundamentals_cache.parquet",
        prices_path=prices_path,
        treasury_path=treasury_path,
    )
    feature_df = feature_result.features.reset_index()

    quality_df, _qerr = read_parquet_safe(quality_scores_path)
    regime_df, _rerr = read_parquet_safe(regime_cache_path)
    risk_df, _kerr = read_parquet_safe(risk_cache_path)
    prices_df, _perr = read_parquet_safe(prices_path)
    treasury_df, _terr = read_parquet_safe(treasury_path)
    if quality_df is None:
        quality_df = pd.DataFrame(columns=["Ticker", "QualityScore", "QualityTier"])
    if regime_df is None:
        regime_df = pd.DataFrame(columns=["Date", "RegimeLabel", "ConfidenceScore"])
    if risk_df is None:
        risk_df = pd.DataFrame(columns=["Date", "RiskScore", "RiskLevel"])
    if prices_df is None:
        raise FileNotFoundError(f"Price cache missing or empty: {prices_path}")

    quality_explain = build_quality_explanations(feature_df=feature_df, quality_df=quality_df)
    regime_evidence = build_regime_evidence(prices_df=prices_df, treasury_df=treasury_df, regime_df=regime_df)
    risk_evidence = build_risk_evidence(prices_df=prices_df, treasury_df=treasury_df, risk_df=risk_df)

    save_parquet_atomic(quality_explain, quality_explain_path)
    save_parquet_atomic(regime_evidence, regime_evidence_path)
    save_parquet_atomic(risk_evidence, risk_evidence_path)

    # Update registry and health with explanation artifact entries.
    now = pd.Timestamp.utcnow().isoformat()
    registry: dict = {}
    try:
        import json

        with open(model_registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        registry = {"generated_at": now, "models": []}
    registry.setdefault("models", [])
    registry["generated_at"] = now
    registry["models"].append(
        {
            "model_name": "explainability_layer",
            "model_version": "1.0.0",
            "feature_set_used": [
                "quality component percentiles",
                "yield curve and benchmark regime indicators",
                "systemic risk indicator decomposition",
            ],
            "training_or_evaluation_window": "latest cached artifacts",
            "timestamp": now,
            "evaluation_summary": {
                "quality_explanations_rows": int(len(quality_explain)),
                "regime_evidence_rows": int(len(regime_evidence)),
                "risk_evidence_rows": int(len(risk_evidence)),
            },
        }
    )
    write_json_report(registry, model_registry_path)

    health: dict = {}
    try:
        import json

        with open(model_health_path, "r", encoding="utf-8") as f:
            health = json.load(f)
    except Exception:
        health = {"generated_at": now}
    health["generated_at"] = now
    health["explanation_coverage"] = {
        "quality_explanations_cache": {"rows": int(len(quality_explain)), "path": quality_explain_path},
        "regime_evidence_cache": {"rows": int(len(regime_evidence)), "path": regime_evidence_path},
        "risk_evidence_cache": {"rows": int(len(risk_evidence)), "path": risk_evidence_path},
    }
    health.setdefault("missing_data_warnings", [])
    health["missing_data_warnings"] = list(dict.fromkeys(list(health["missing_data_warnings"]) + feature_result.warnings))
    write_json_report(health, model_health_path)

    return ExplainabilityPipelineRunResult(
        quality_explanations=quality_explain,
        regime_evidence=regime_evidence,
        risk_evidence=risk_evidence,
        wrote_artifacts=True,
        reason="explainability artifacts updated",
    )
