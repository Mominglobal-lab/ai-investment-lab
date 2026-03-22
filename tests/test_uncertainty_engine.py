from __future__ import annotations

import pandas as pd

from ai_models.uncertainty_engine import build_quality_uncertainty, build_risk_uncertainty


def test_quality_uncertainty_drops_missing_tickers():
    feature_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10, "EBITDA_Margin": 0.2, "ROE": 0.1, "FreeCashFlow_Margin": 0.1, "Volatility_63D": 0.2, "Drawdown_252D": -0.1},
            {"Ticker": "BBB", "Revenue_Growth_YoY_Pct": 5, "EBITDA_Margin": 0.1, "ROE": 0.05, "FreeCashFlow_Margin": 0.05, "Volatility_63D": 0.3, "Drawdown_252D": -0.2},
        ]
    )
    quality_df = pd.DataFrame(
        [
            {"Ticker": None, "QualityScore": 50, "QualityTier": "Neutral"},
            {"Ticker": "AAA", "QualityScore": 75, "QualityTier": "Strong"},
        ]
    )

    out = build_quality_uncertainty(feature_df, quality_df, n_boot=10, seed=1)

    assert out["Ticker"].tolist() == ["AAA"]


def test_quality_uncertainty_deduplicates_tickers():
    feature_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10, "EBITDA_Margin": 0.2, "ROE": 0.1, "FreeCashFlow_Margin": 0.1, "Volatility_63D": 0.2, "Drawdown_252D": -0.1},
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 15, "EBITDA_Margin": 0.25, "ROE": 0.15, "FreeCashFlow_Margin": 0.12, "Volatility_63D": 0.18, "Drawdown_252D": -0.08},
        ]
    )
    quality_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "QualityScore": 70, "QualityTier": "Strong"},
            {"Ticker": "AAA", "QualityScore": 74, "QualityTier": "Strong"},
        ]
    )

    out = build_quality_uncertainty(feature_df, quality_df, n_boot=10, seed=1)

    assert out["Ticker"].tolist() == ["AAA"]


def test_quality_uncertainty_normalizes_missing_quality_tier():
    feature_df = pd.DataFrame()
    quality_df = pd.DataFrame([{"Ticker": "AAA", "QualityScore": 70, "QualityTier": None}])

    out = build_quality_uncertainty(feature_df, quality_df, n_boot=10, seed=1)

    assert out.iloc[0]["TierMostLikely"] == "Unknown"


def test_risk_uncertainty_drops_non_finite_risk_scores():
    risk_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=4, freq="B"),
            "RiskScore": [10.0, float("inf"), 20.0, 30.0],
        }
    )

    out = build_risk_uncertainty(risk_df, window=10)

    assert len(out) == 3
    assert out["RiskScore"].notna().all()


def test_risk_uncertainty_returns_expected_schema_when_all_scores_invalid():
    risk_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=3, freq="B"),
            "RiskScore": [float("inf"), float("nan"), float("-inf")],
        }
    )

    out = build_risk_uncertainty(risk_df, window=10)

    assert list(out.columns) == [
        "Date",
        "RiskScore",
        "RiskP10",
        "RiskP50",
        "RiskP90",
        "RiskLevelMostLikely",
        "RiskLevelStability",
    ]
    assert out.empty
