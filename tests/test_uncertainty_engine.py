from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_models.uncertainty_engine import build_quality_uncertainty, build_risk_uncertainty
from ai_models.quality_score_model import run_quality_score_model


def _make_feature_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Revenue_Growth_YoY_Pct": rng.normal(10, 8, n),
            "EBITDA_Margin": rng.uniform(0.05, 0.35, n),
            "ROE": rng.uniform(0.02, 0.30, n),
            "FreeCashFlow_Margin": rng.uniform(-0.02, 0.20, n),
            "Volatility_63D": rng.uniform(0.15, 0.55, n),
            "Drawdown_252D": -rng.uniform(0.05, 0.60, n),
        }
    )


def test_quality_uncertainty_deterministic_and_bands():
    f = _make_feature_df(40)
    q = run_quality_score_model(f)
    u1 = build_quality_uncertainty(f, q, n_boot=120, seed=42)
    u2 = build_quality_uncertainty(f, q, n_boot=120, seed=42)
    pd.testing.assert_frame_equal(u1.sort_values("Ticker").reset_index(drop=True), u2.sort_values("Ticker").reset_index(drop=True))

    assert (u1["ScoreP10"] <= u1["ScoreP50"]).all()
    assert (u1["ScoreP50"] <= u1["ScoreP90"]).all()
    assert u1["TierStability"].between(0, 1).all()
    assert set(u1["TierMostLikely"].unique()).issubset({"Strong", "Neutral", "Weak"})


def test_risk_uncertainty_bands_shape():
    dates = pd.date_range("2024-01-01", periods=320, freq="B")
    rs = np.clip(40 + np.sin(np.linspace(0, 20, len(dates))) * 15, 0, 100)
    risk_df = pd.DataFrame({"Date": dates, "RiskScore": rs, "RiskLevel": ["Moderate"] * len(dates)})
    out = build_risk_uncertainty(risk_df, window=252)
    assert not out.empty
    assert (out["RiskP10"] <= out["RiskP50"]).all()
    assert (out["RiskP50"] <= out["RiskP90"]).all()
    assert out["RiskLevelStability"].between(0, 1).all()
    # Point estimate should generally be in the plausible band for rolling-delta construction.
    frac_in_band = ((out["RiskScore"] >= out["RiskP10"]) & (out["RiskScore"] <= out["RiskP90"])).mean()
    assert frac_in_band >= 0.7

