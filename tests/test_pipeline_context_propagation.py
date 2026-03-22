from __future__ import annotations

import pandas as pd

import data_pipeline.run_pipeline as rp
from ai_models.feature_builder import FeatureBuildResult


def test_explainability_pipeline_passes_custom_context(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_build_feature_table(**kwargs):
        captured.update(kwargs)
        return FeatureBuildResult(
            features=pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]).set_index("Ticker"),
            warnings=[],
            input_coverage={},
        )

    def _fake_read_parquet_safe(path: str):
        if "prices" in path:
            return pd.DataFrame([{"Ticker": "QQQ", "Date": "2025-01-02", "AdjClose": 100.0}]), None
        if "treasury" in path:
            return pd.DataFrame([{"Date": "2025-01-02", "10Y": 4.0, "2Y": 3.8}]), None
        if "quality" in path:
            return pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0, "QualityTier": "Strong"}]), None
        if "regime" in path:
            return pd.DataFrame([{"Date": "2025-01-02", "RegimeLabel": "Risk On", "ConfidenceScore": 0.8}]), None
        if "risk" in path:
            return pd.DataFrame([{"Date": "2025-01-02", "RiskScore": 20.0, "RiskLevel": "Low"}]), None
        return pd.DataFrame(), None

    monkeypatch.setattr(rp, "build_feature_table", _fake_build_feature_table)
    monkeypatch.setattr(rp, "read_parquet_safe", _fake_read_parquet_safe)
    monkeypatch.setattr(rp, "build_quality_explanations", lambda feature_df, quality_df: pd.DataFrame([{"Ticker": "AAPL"}]))
    monkeypatch.setattr(rp, "build_regime_evidence", lambda prices_df, treasury_df, regime_df, benchmark_ticker=None: pd.DataFrame([{"Date": "2025-01-02"}]))
    monkeypatch.setattr(rp, "build_risk_evidence", lambda prices_df, treasury_df, risk_df: pd.DataFrame([{"Date": "2025-01-02"}]))
    monkeypatch.setattr(rp, "save_parquet_atomic", lambda df, path: None)
    monkeypatch.setattr(rp, "write_json_report", lambda payload, path: None)

    rp.run_explainability_pipeline(
        fundamentals_path="custom_fundamentals.parquet",
        prices_path="custom_prices.parquet",
        treasury_path="custom_treasury.parquet",
        benchmark_ticker="QQQ",
    )

    assert captured["fundamentals_path"] == "custom_fundamentals.parquet"
    assert captured["prices_path"] == "custom_prices.parquet"
    assert captured["treasury_path"] == "custom_treasury.parquet"
    assert captured["benchmark_ticker"] == "QQQ"


def test_uncertainty_pipeline_passes_custom_context(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_build_feature_table(**kwargs):
        captured.update(kwargs)
        return FeatureBuildResult(
            features=pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]).set_index("Ticker"),
            warnings=[],
            input_coverage={},
        )

    monkeypatch.setattr(rp, "build_feature_table", _fake_build_feature_table)
    monkeypatch.setattr(rp, "read_parquet_safe", lambda path: (pd.DataFrame(), None))
    monkeypatch.setattr(rp, "build_quality_uncertainty", lambda feature_df, quality_df, n_boot, seed: pd.DataFrame([{"Ticker": "AAPL"}]))
    monkeypatch.setattr(rp, "build_regime_probabilities", lambda regime_df, window: pd.DataFrame([{"Date": "2025-01-02"}]))
    monkeypatch.setattr(rp, "build_risk_uncertainty", lambda risk_df, window: pd.DataFrame([{"Date": "2025-01-02"}]))
    monkeypatch.setattr(rp, "save_parquet_atomic", lambda df, path: None)
    monkeypatch.setattr(rp, "write_json_report", lambda payload, path: None)

    rp.run_uncertainty_pipeline(
        fundamentals_path="custom_fundamentals.parquet",
        prices_path="custom_prices.parquet",
        treasury_path="custom_treasury.parquet",
        benchmark_ticker="QQQ",
    )

    assert captured["fundamentals_path"] == "custom_fundamentals.parquet"
    assert captured["prices_path"] == "custom_prices.parquet"
    assert captured["treasury_path"] == "custom_treasury.parquet"
    assert captured["benchmark_ticker"] == "QQQ"


def test_model_pipeline_passes_custom_benchmark_context(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_build_feature_table(**kwargs):
        captured.update(kwargs)
        return FeatureBuildResult(
            features=pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]).set_index("Ticker"),
            warnings=[],
            input_coverage={},
        )

    monkeypatch.setattr(rp, "build_feature_table", _fake_build_feature_table)
    monkeypatch.setattr(rp, "run_quality_score_model", lambda features: pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]))
    monkeypatch.setattr(rp, "run_regime_detection_model", lambda **kwargs: pd.DataFrame([{"Date": "2025-01-02", "RegimeLabel": "Risk On"}]))
    monkeypatch.setattr(rp, "run_systemic_risk_detector", lambda **kwargs: pd.DataFrame([{"Date": "2025-01-02", "RiskLevel": "Low", "RiskScore": 20.0}]))
    monkeypatch.setattr(rp, "save_parquet_atomic", lambda df, path: None)
    monkeypatch.setattr(rp, "write_json_report", lambda payload, path: None)
    monkeypatch.setattr(rp, "get_cache_status", lambda path, max_age_days, required_columns=None: type("S", (), {"exists": False, "schema_ok": False, "is_fresh": False})())
    monkeypatch.setattr(rp, "read_parquet_safe", lambda path: (pd.DataFrame(), None))

    rp.run_decision_models_pipeline(
        fundamentals_path="custom_fundamentals.parquet",
        prices_path="custom_prices.parquet",
        treasury_path="custom_treasury.parquet",
        benchmark_ticker="QQQ",
    )

    assert captured["fundamentals_path"] == "custom_fundamentals.parquet"
    assert captured["prices_path"] == "custom_prices.parquet"
    assert captured["treasury_path"] == "custom_treasury.parquet"
    assert captured["benchmark_ticker"] == "QQQ"
