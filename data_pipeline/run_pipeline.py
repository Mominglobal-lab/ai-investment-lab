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
from ai_models.uncertainty_engine import build_quality_uncertainty, build_risk_uncertainty
from ai_models.probability_calibrator import build_regime_probabilities
from ai_models.drift_engine import compute_feature_drift, compute_signal_instability
from ai_models.monitoring_engine import build_drift_report, build_monitoring_health_report
from ai_models.alert_engine import generate_alerts


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


@dataclass(frozen=True)
class UncertaintyPipelineRunResult:
    quality_uncertainty: pd.DataFrame
    regime_probabilities: pd.DataFrame
    risk_uncertainty: pd.DataFrame
    wrote_artifacts: bool
    reason: str


@dataclass(frozen=True)
class MonitoringPipelineRunResult:
    drift_signals: pd.DataFrame
    alerts: pd.DataFrame
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
    run_uncertainty: bool = False,
    run_monitoring: bool = False,
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
    uncertainty_result = None
    if run_uncertainty:
        uncertainty_result = run_uncertainty_pipeline(benchmark_ticker=benchmark_ticker)
    monitoring_result = None
    if run_monitoring:
        monitoring_result = run_monitoring_pipeline(benchmark_ticker=benchmark_ticker)
    return {
        "stock": stock_result,
        "fixed_income": fi_result,
        "prices": prices_result,
        "models": model_result,
        "explanations": explain_result,
        "uncertainty": uncertainty_result,
        "monitoring": monitoring_result,
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
    treasury_status = get_cache_status(treasury_path, 365, required_columns=["Date", "10Y", "2Y", "3M"])
    treasury_rows = 0
    treasury_df, _terr = read_parquet_safe(treasury_path)
    if treasury_df is not None:
        treasury_rows = int(len(treasury_df))
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
            "treasury_yields_cache": {
                "exists": treasury_status.exists,
                "schema_ok": treasury_status.schema_ok,
                "row_count": treasury_rows,
            },
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


def run_uncertainty_pipeline(
    *,
    benchmark_ticker: str = "SPY",
    prices_path: str = "data/prices_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    quality_scores_path: str = "data/quality_scores_cache.parquet",
    regime_cache_path: str = "data/regime_cache.parquet",
    risk_cache_path: str = "data/risk_signals_cache.parquet",
    quality_uncertainty_path: str = "data/quality_uncertainty_cache.parquet",
    regime_prob_path: str = "data/regime_probabilities_cache.parquet",
    risk_uncertainty_path: str = "data/risk_uncertainty_cache.parquet",
    model_registry_path: str = "data/model_registry.json",
    model_health_path: str = "data/model_health_report.json",
) -> UncertaintyPipelineRunResult:
    feature_result = build_feature_table(
        fundamentals_path="data/fundamentals_cache.parquet",
        prices_path=prices_path,
        treasury_path=treasury_path,
    )
    feature_df = feature_result.features.reset_index()

    quality_df, _qerr = read_parquet_safe(quality_scores_path)
    regime_df, _rerr = read_parquet_safe(regime_cache_path)
    risk_df, _kerr = read_parquet_safe(risk_cache_path)
    if quality_df is None:
        quality_df = pd.DataFrame(columns=["Ticker", "QualityScore", "QualityTier"])
    if regime_df is None:
        regime_df = pd.DataFrame(columns=["Date", "RegimeLabel", "ConfidenceScore"])
    if risk_df is None:
        risk_df = pd.DataFrame(columns=["Date", "RiskScore", "RiskLevel"])

    q_unc = build_quality_uncertainty(feature_df=feature_df, quality_df=quality_df, n_boot=300, seed=42)
    r_prob = build_regime_probabilities(regime_df=regime_df, window=20)
    k_unc = build_risk_uncertainty(risk_df=risk_df, window=252)

    save_parquet_atomic(q_unc, quality_uncertainty_path)
    save_parquet_atomic(r_prob, regime_prob_path)
    save_parquet_atomic(k_unc, risk_uncertainty_path)

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
            "model_name": "uncertainty_layer",
            "model_version": "1.0.0",
            "feature_set_used": [
                "quality bootstrap score bands",
                "regime confidence-to-probability mapping",
                "risk rolling movement bands",
            ],
            "training_or_evaluation_window": "latest cached artifacts",
            "timestamp": now,
            "evaluation_summary": {
                "quality_uncertainty_rows": int(len(q_unc)),
                "regime_probabilities_rows": int(len(r_prob)),
                "risk_uncertainty_rows": int(len(k_unc)),
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
    health["uncertainty_coverage"] = {
        "quality_uncertainty_cache": {"rows": int(len(q_unc)), "path": quality_uncertainty_path},
        "regime_probabilities_cache": {"rows": int(len(r_prob)), "path": regime_prob_path},
        "risk_uncertainty_cache": {"rows": int(len(k_unc)), "path": risk_uncertainty_path},
    }
    health.setdefault("runtime_notes", [])
    health["runtime_notes"] = list(
        dict.fromkeys(list(health["runtime_notes"]) + ["Quality uncertainty uses deterministic bootstrap with seed=42."])
    )
    write_json_report(health, model_health_path)

    return UncertaintyPipelineRunResult(
        quality_uncertainty=q_unc,
        regime_probabilities=r_prob,
        risk_uncertainty=k_unc,
        wrote_artifacts=True,
        reason="uncertainty artifacts updated",
    )


def _build_feature_history_for_monitoring(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None,
    treasury_df: pd.DataFrame | None,
    benchmark_ticker: str = "SPY",
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    px = prices_df.copy()
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px["AdjClose"] = pd.to_numeric(px["AdjClose"], errors="coerce")
    px = px.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"])

    rows: list[dict[str, object]] = []
    for t, g in px.groupby("Ticker"):
        s = g.set_index("Date")["AdjClose"].sort_index()
        r = s.pct_change()
        mom252 = s.pct_change(252)
        vol63 = r.rolling(63, min_periods=10).std() * (252 ** 0.5)
        roll_max = s.rolling(252, min_periods=30).max()
        dd252 = (s / roll_max) - 1.0
        tmp = pd.DataFrame(
            {
                "Date": s.index,
                "Ticker": t,
                "Momentum_252d": mom252.values,
                "Volatility_63d": vol63.values,
                "MaxDrawdown_252d": dd252.values,
            }
        )
        rows.extend(tmp.to_dict(orient="records"))
    fh = pd.DataFrame(rows)

    # Benchmark volatility series
    b = px[px["Ticker"] == benchmark_ticker.upper()].sort_values("Date")
    if b.empty:
        warnings.append(f"Benchmark {benchmark_ticker} missing for benchmark volatility feature.")
        fh["BenchmarkVolatility_63d"] = pd.NA
    else:
        bs = b.set_index("Date")["AdjClose"].sort_index()
        bv = bs.pct_change().rolling(63, min_periods=10).std() * (252 ** 0.5)
        bdf = pd.DataFrame({"Date": bs.index, "BenchmarkVolatility_63d": bv.values})
        fh = fh.merge(bdf, on="Date", how="left")

    # Fundamentals static features
    if fundamentals_df is not None and not fundamentals_df.empty and "Ticker" in fundamentals_df.columns:
        f = fundamentals_df.copy()
        f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
        fmap = f.set_index("Ticker")
        for tgt, src in [
            ("RevenueGrowth", "Revenue_Growth_YoY_Pct"),
            ("EBITDAMargin", "EBITDA_Margin"),
            ("ROE", "ROE"),
            ("FCFMargin", "FreeCashFlow_Margin"),
        ]:
            if src in fmap.columns:
                fh[tgt] = fh["Ticker"].map(pd.to_numeric(fmap[src], errors="coerce"))
            else:
                fh[tgt] = pd.NA
                warnings.append(f"Fundamentals missing {src}; {tgt} unavailable.")
    else:
        for tgt in ["RevenueGrowth", "EBITDAMargin", "ROE", "FCFMargin"]:
            fh[tgt] = pd.NA
        warnings.append("Fundamentals cache missing for monitoring features.")

    # Yield slope optional
    fh["YC_Slope_10Y_2Y"] = pd.NA
    if treasury_df is not None and not treasury_df.empty:
        t = treasury_df.copy()
        cols = list(t.columns)
        canon = {c.lower().replace("_", "").replace(" ", ""): c for c in cols}
        dcol = canon.get("date") or canon.get("timestamp") or canon.get("asofdate")
        y10 = canon.get("10y") or canon.get("dgs10") or canon.get("yield10y")
        y2 = canon.get("2y") or canon.get("dgs2") or canon.get("yield2y")
        y3 = canon.get("3m") or canon.get("dgs3mo") or canon.get("yield3m")
        if dcol and y10 and (y2 or y3):
            t["Date"] = pd.to_datetime(t[dcol], errors="coerce")
            s10 = pd.to_numeric(t[y10], errors="coerce")
            ss = pd.to_numeric(t[y2], errors="coerce") if y2 else pd.to_numeric(t[y3], errors="coerce")
            ydf = pd.DataFrame({"Date": t["Date"], "YC_Slope_10Y_2Y": s10 - ss}).dropna(subset=["Date"])
            fh = fh.merge(ydf, on="Date", how="left")
        else:
            warnings.append("Treasury cache missing yield columns for YC slope.")
    else:
        warnings.append("Treasury cache missing for yield-curve monitoring feature.")

    fh["Date"] = pd.to_datetime(fh["Date"], errors="coerce")
    fh = fh.dropna(subset=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return fh, warnings


def run_monitoring_pipeline(
    *,
    benchmark_ticker: str = "SPY",
    prices_path: str = "data/prices_cache.parquet",
    fundamentals_path: str = "data/fundamentals_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    regime_path: str = "data/regime_cache.parquet",
    risk_path: str = "data/risk_signals_cache.parquet",
    drift_signals_path: str = "data/drift_signals_cache.parquet",
    drift_signals_history_path: str = "data/drift_signals_history.parquet",
    drift_report_path: str = "data/drift_report.json",
    alert_log_path: str = "data/alert_log.parquet",
    monitoring_health_path: str = "data/monitoring_health_report.json",
    model_registry_path: str = "data/model_registry.json",
    model_health_path: str = "data/model_health_report.json",
) -> MonitoringPipelineRunResult:
    prices_df, _p = read_parquet_safe(prices_path)
    if prices_df is None or prices_df.empty:
        raise FileNotFoundError(f"Price cache missing or empty: {prices_path}")
    fundamentals_df, _f = read_parquet_safe(fundamentals_path)
    treasury_df, _t = read_parquet_safe(treasury_path)
    regime_df, _r = read_parquet_safe(regime_path)
    risk_df, _k = read_parquet_safe(risk_path)

    fh, warnings = _build_feature_history_for_monitoring(
        prices_df=prices_df,
        fundamentals_df=fundamentals_df,
        treasury_df=treasury_df,
        benchmark_ticker=benchmark_ticker,
    )
    if fh.empty:
        raise ValueError("Feature history empty; cannot compute drift.")
    latest = fh["Date"].max()
    current_end = latest
    current_start = latest - pd.tseries.offsets.BDay(59)
    baseline_end = latest - pd.tseries.offsets.BDay(60)
    baseline_start = baseline_end - pd.tseries.offsets.BDay(251)
    if baseline_start < fh["Date"].min():
        baseline_start = fh["Date"].min()
        warnings.append("Insufficient history for default baseline window; used shorter baseline window.")
    if current_start < fh["Date"].min():
        current_start = fh["Date"].min()
        warnings.append("Insufficient history for default current window; used shorter current window.")

    feat_drift = compute_feature_drift(
        feature_history_df=fh,
        baseline_window=(pd.Timestamp(baseline_start), pd.Timestamp(baseline_end)),
        current_window=(pd.Timestamp(current_start), pd.Timestamp(current_end)),
    )
    sig_drift = compute_signal_instability(
        regime_df=regime_df,
        risk_df=risk_df,
        quality_history_df=None,
        feature_history_df=fh,
    )
    drift_df = pd.concat([feat_drift, sig_drift], axis=0, ignore_index=True)
    save_parquet_atomic(drift_df, drift_signals_path)
    # Keep a longitudinal history for trend charts while preserving snapshot behavior.
    hist_prev, _he = read_parquet_safe(drift_signals_history_path)

    def _bootstrap_feature_history_rows() -> pd.DataFrame:
        hist_rows: list[pd.DataFrame] = []
        try:
            all_dates = sorted(pd.to_datetime(fh["Date"], errors="coerce").dropna().unique().tolist())
            eval_dates = all_dates[-90:] if len(all_dates) > 90 else all_dates
            fh_min = pd.to_datetime(fh["Date"], errors="coerce").dropna().min()
            for d in eval_dates:
                d = pd.Timestamp(d)
                current_end = d
                current_start = d - pd.tseries.offsets.BDay(59)
                baseline_end = d - pd.tseries.offsets.BDay(60)
                baseline_start = baseline_end - pd.tseries.offsets.BDay(251)
                if pd.notna(fh_min):
                    if baseline_start < fh_min:
                        baseline_start = fh_min
                    if current_start < fh_min:
                        current_start = fh_min
                if baseline_end < baseline_start:
                    continue
                tmp = compute_feature_drift(
                    feature_history_df=fh,
                    baseline_window=(pd.Timestamp(baseline_start), pd.Timestamp(baseline_end)),
                    current_window=(pd.Timestamp(current_start), pd.Timestamp(current_end)),
                )
                if tmp is not None and not tmp.empty:
                    hist_rows.append(tmp)
        except Exception:
            hist_rows = []
        return pd.concat(hist_rows, axis=0, ignore_index=True) if hist_rows else pd.DataFrame()

    need_backfill = False
    if hist_prev is None or hist_prev.empty:
        need_backfill = True
    else:
        hp = hist_prev.copy()
        hp["Date"] = pd.to_datetime(hp.get("Date"), errors="coerce")
        feat_metrics = set(feat_drift["MetricName"].astype(str).tolist()) if not feat_drift.empty else set()
        if feat_metrics and {"MetricName", "Date"}.issubset(set(hp.columns)):
            c = hp[hp["MetricName"].astype(str).isin(feat_metrics)].groupby("MetricName")["Date"].nunique()
            # Backfill if any feature metric has very sparse history.
            need_backfill = (c.reindex(sorted(feat_metrics)).fillna(0) < 10).any()

    base_hist = hist_prev.copy() if (hist_prev is not None and not hist_prev.empty) else pd.DataFrame()
    if need_backfill:
        boot = _bootstrap_feature_history_rows()
        if not boot.empty:
            base_hist = pd.concat([base_hist, boot], axis=0, ignore_index=True)

    hist = pd.concat([base_hist, drift_df], axis=0, ignore_index=True)
    if "Date" in hist.columns:
        hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["Date"])
    dedup_keys = [c for c in ["Date", "MetricName", "MetricType"] if c in hist.columns]
    if dedup_keys:
        hist = hist.drop_duplicates(subset=dedup_keys, keep="last")
    hist = hist.sort_values("Date")
    save_parquet_atomic(hist, drift_signals_history_path)

    coverage_stats = {
        "prices_rows": int(len(prices_df)),
        "feature_history_rows": int(len(fh)),
        "treasury_exists": bool(treasury_df is not None and not treasury_df.empty),
        "expected_min_price_rows": 50000,
        "missing_counts": {c: int(fh[c].isna().sum()) for c in fh.columns if c not in {"Date", "Ticker"}},
    }
    alerts_df = generate_alerts(
        drift_df=drift_df,
        regime_df=regime_df if regime_df is not None else pd.DataFrame(),
        risk_df=risk_df if risk_df is not None else pd.DataFrame(),
        coverage_stats=coverage_stats,
    )
    save_parquet_atomic(alerts_df, alert_log_path)

    drift_report = build_drift_report(
        drift_df=drift_df,
        baseline_window=(pd.Timestamp(baseline_start), pd.Timestamp(baseline_end)),
        current_window=(pd.Timestamp(current_start), pd.Timestamp(current_end)),
        coverage_stats=coverage_stats,
        warnings=warnings,
    )
    write_json_report(drift_report, drift_report_path)
    mon_health = build_monitoring_health_report(
        drift_df=drift_df,
        alerts_df=alerts_df,
        coverage_stats=coverage_stats,
        runtime_notes=["Monitoring uses PSI baseline/current windows with fallback shortening when history is limited."],
    )
    write_json_report(mon_health, monitoring_health_path)

    # update registry / model health
    now = pd.Timestamp.utcnow().isoformat()
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
            "model_name": "monitoring_layer",
            "model_version": "1.0.0",
            "feature_set_used": ["PSI feature drift", "regime/risk instability metrics", "rule-based alerts"],
            "training_or_evaluation_window": "rolling baseline/current windows",
            "timestamp": now,
            "evaluation_summary": {
                "drift_rows": int(len(drift_df)),
                "alerts_rows": int(len(alerts_df)),
            },
        }
    )
    write_json_report(registry, model_registry_path)

    try:
        import json

        with open(model_health_path, "r", encoding="utf-8") as f:
            health = json.load(f)
    except Exception:
        health = {"generated_at": now}
    health["generated_at"] = now
    health["monitoring_summary"] = {
        "drift_signals_rows": int(len(drift_df)),
        "alerts_rows": int(len(alerts_df)),
        "worst_drift_level": "Severe" if (drift_df["DriftLevel"] == "Severe").any() else ("Drift" if (drift_df["DriftLevel"] == "Drift").any() else "Stable"),
    }
    write_json_report(health, model_health_path)

    return MonitoringPipelineRunResult(
        drift_signals=drift_df,
        alerts=alerts_df,
        wrote_artifacts=True,
        reason="monitoring artifacts updated",
    )
