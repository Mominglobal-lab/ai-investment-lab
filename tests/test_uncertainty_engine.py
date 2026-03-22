from __future__ import annotations

import pandas as pd

from ai_models.uncertainty_engine import build_quality_uncertainty


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
