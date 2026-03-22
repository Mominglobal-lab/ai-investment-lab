from __future__ import annotations

import pandas as pd

from ai_models.quality_score_model import run_quality_score_model


def test_quality_score_normalization_and_tiers():
    features = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Revenue_Growth_YoY_Pct": 30,
                "EBITDA_Margin": 0.30,
                "ROE": 0.25,
                "FreeCashFlow_Margin": 0.18,
                "Volatility_63D": 0.18,
                "Drawdown_252D": -0.10,
            },
            {
                "Ticker": "BBB",
                "Revenue_Growth_YoY_Pct": 5,
                "EBITDA_Margin": 0.12,
                "ROE": 0.10,
                "FreeCashFlow_Margin": 0.05,
                "Volatility_63D": 0.28,
                "Drawdown_252D": -0.30,
            },
            {
                "Ticker": "CCC",
                "Revenue_Growth_YoY_Pct": -10,
                "EBITDA_Margin": 0.04,
                "ROE": 0.02,
                "FreeCashFlow_Margin": -0.01,
                "Volatility_63D": 0.45,
                "Drawdown_252D": -0.55,
            },
        ]
    )
    out = run_quality_score_model(features)
    assert not out.empty
    assert out["QualityScore"].between(0, 100).all()
    assert set(out["QualityTier"].unique()).issubset({"Strong", "Neutral", "Weak"})
    assert len(out["Explanation"].iloc[0]) > 0


def test_quality_score_handles_missing_feature_columns():
    features = pd.DataFrame(
        [
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 10},
            {"Ticker": "BBB", "Revenue_Growth_YoY_Pct": 5},
        ]
    )

    out = run_quality_score_model(features)

    assert len(out) == 2
    assert out["QualityScore"].between(0, 100).all()
    assert set(out["QualityTier"].unique()).issubset({"Strong", "Neutral", "Weak"})


def test_quality_score_drops_missing_tickers():
    features = pd.DataFrame(
        [
            {"Ticker": None, "Revenue_Growth_YoY_Pct": 10},
            {"Ticker": "AAA", "Revenue_Growth_YoY_Pct": 5},
        ]
    )

    out = run_quality_score_model(features)

    assert out["Ticker"].tolist() == ["AAA"]

