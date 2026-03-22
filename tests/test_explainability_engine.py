from __future__ import annotations

import json

import pandas as pd
import pytest

from ai_models.explainability_engine import build_quality_explanations


def test_quality_explainability_contributions_and_top_drivers():
    feature_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 30, "EBITDA_Margin": 0.3, "ROE": 0.2, "FreeCashFlow_Margin": 0.15, "Volatility_63D": 0.2, "Drawdown_252D": -0.1},
            {"Ticker": "BBB", "Revenue_Growth_YoY_Pct": 10, "EBITDA_Margin": 0.2, "ROE": 0.12, "FreeCashFlow_Margin": 0.08, "Volatility_63D": 0.3, "Drawdown_252D": -0.2},
            {"Ticker": "CCC", "Revenue_Growth_YoY_Pct": -5, "EBITDA_Margin": 0.1, "ROE": 0.04, "FreeCashFlow_Margin": 0.01, "Volatility_63D": 0.45, "Drawdown_252D": -0.4},
        ]
    )
    quality_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "QualityScore": 90, "QualityTier": "Strong"},
            {"Ticker": "BBB", "QualityScore": 55, "QualityTier": "Neutral"},
            {"Ticker": "CCC", "QualityScore": 20, "QualityTier": "Weak"},
        ]
    )

    out = build_quality_explanations(feature_df, quality_df)
    assert not out.empty
    row = out[out["Ticker"] == "AAA"].iloc[0]
    contrib = json.loads(row["ContributionJSON"])
    assert isinstance(contrib, dict)
    assert len(contrib) > 0
    assert pytest.approx(sum(abs(float(v)) for v in contrib.values()), rel=1e-6) == 1.0
    assert isinstance(row["TopPositiveDrivers"], str)
    assert isinstance(row["TopNegativeDrivers"], str)


def test_quality_explainability_handles_missing_features():
    feature_df = pd.DataFrame([{"Ticker": "AAA"}])
    quality_df = pd.DataFrame([{"Ticker": "AAA", "QualityScore": 50, "QualityTier": "Neutral"}])
    out = build_quality_explanations(feature_df, quality_df)
    assert len(out) == 1
    contrib = json.loads(out.iloc[0]["ContributionJSON"])
    assert isinstance(contrib, dict)


def test_quality_explainability_sanitizes_non_finite_json_values():
    feature_df = pd.DataFrame([{"Ticker": "AAA"}])
    quality_df = pd.DataFrame([{"Ticker": "AAA", "QualityScore": float("inf"), "QualityTier": "Neutral"}])

    out = build_quality_explanations(feature_df, quality_df)

    assert len(out) == 1
    row = out.iloc[0]
    assert pd.isna(row["QualityScore"])
    assert "Infinity" not in row["ContributionJSON"]
    assert "NaN" not in row["ContributionJSON"]
    contrib = json.loads(row["ContributionJSON"])
    assert isinstance(contrib, dict)


def test_quality_explainability_drops_missing_tickers():
    feature_df = pd.DataFrame([{"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10}])
    quality_df = pd.DataFrame(
        [
            {"Ticker": None, "QualityScore": 55, "QualityTier": "Neutral"},
            {"Ticker": "AAA", "QualityScore": 80, "QualityTier": "Strong"},
        ]
    )

    out = build_quality_explanations(feature_df, quality_df)

    assert out["Ticker"].tolist() == ["AAA"]


def test_quality_explainability_deduplicates_tickers():
    feature_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10},
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 20},
        ]
    )
    quality_df = pd.DataFrame(
        [
            {"Ticker": "AAA", "QualityScore": 70, "QualityTier": "Strong"},
            {"Ticker": "AAA", "QualityScore": 75, "QualityTier": "Strong"},
        ]
    )

    out = build_quality_explanations(feature_df, quality_df)

    assert out["Ticker"].tolist() == ["AAA"]


def test_quality_explainability_normalizes_missing_quality_tier():
    feature_df = pd.DataFrame([{"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10}])
    quality_df = pd.DataFrame([{"Ticker": "AAA", "QualityScore": 80, "QualityTier": None}])

    out = build_quality_explanations(feature_df, quality_df)

    assert out.iloc[0]["QualityTier"] == "Unknown"

