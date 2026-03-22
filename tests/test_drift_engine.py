from __future__ import annotations

import numpy as np
import pandas as pd

from ai_models.drift_engine import compute_feature_drift, compute_signal_instability


def test_feature_drift_psi_thresholds():
    dates = pd.date_range("2024-01-01", periods=400, freq="B")
    tickers = [f"T{i:03d}" for i in range(40)]
    rows = []
    for d in dates:
        shift = 0.0 if d < dates[-60] else 2.0
        for t in tickers:
            rows.append(
                {
                    "Date": d,
                    "Ticker": t,
                    "Momentum_252d": np.random.normal(0 + shift, 1.0),
                    "Volatility_63d": np.random.normal(0.2 + 0.1 * shift, 0.05),
                }
            )
    df = pd.DataFrame(rows)
    b0 = dates[-320]
    b1 = dates[-61]
    c0 = dates[-60]
    c1 = dates[-1]
    out = compute_feature_drift(df, baseline_window=(b0, b1), current_window=(c0, c1))
    assert not out.empty
    assert {"BaselineWindowStart", "BaselineWindowEnd", "CurrentWindowStart", "CurrentWindowEnd"}.issubset(out.columns)
    mom = out[out["MetricName"] == "Momentum_252d"].iloc[0]
    assert mom["DriftScore"] > 0.10
    assert mom["DriftLevel"] in {"Drift", "Severe"}


def test_feature_drift_detects_shift_when_baseline_is_constant():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    tickers = [f"T{i:03d}" for i in range(25)]
    rows = []
    for d in dates:
        current_val = 5.0 if d >= dates[-20] else 0.0
        for t in tickers:
            rows.append(
                {
                    "Date": d,
                    "Ticker": t,
                    "Momentum_252d": current_val,
                }
            )
    df = pd.DataFrame(rows)
    b0 = dates[0]
    b1 = dates[-21]
    c0 = dates[-20]
    c1 = dates[-1]

    out = compute_feature_drift(df, baseline_window=(b0, b1), current_window=(c0, c1))

    mom = out[out["MetricName"] == "Momentum_252d"].iloc[0]
    assert mom["DriftScore"] > 0.25
    assert mom["DriftLevel"] == "Severe"


def test_quality_proxy_drift_uses_current_quality_feature_names():
    dates = pd.date_range("2025-01-01", periods=320, freq="B")
    feature_history_df = pd.DataFrame(
        {
            "Date": dates,
            "Revenue_Growth_YoY_Pct": [0.05] * 220 + [0.30] * 100,
            "EBITDA_Margin": [0.10] * 220 + [0.35] * 100,
            "ROE": [0.08] * 220 + [0.20] * 100,
            "FreeCashFlow_Margin": [0.04] * 220 + [0.18] * 100,
        }
    )

    out = compute_signal_instability(
        regime_df=None,
        risk_df=None,
        quality_history_df=None,
        feature_history_df=feature_history_df,
    )

    row = out[out["MetricName"] == "QualityProxyDrift"].iloc[0]
    assert row["DriftScore"] > 0


def test_signal_instability_ignores_case_only_label_flips():
    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    regime_df = pd.DataFrame(
        {
            "Date": dates,
            "RegimeLabel": ["Risk Off", " risk off ", "RISK OFF", "Risk Off"] * 5,
            "ConfidenceScore": [0.6] * 20,
        }
    )
    risk_df = pd.DataFrame(
        {
            "Date": dates,
            "RiskScore": [50.0] * 20,
            "RiskLevel": ["Moderate", " moderate ", "MODERATE", "Moderate"] * 5,
        }
    )

    out = compute_signal_instability(
        regime_df=regime_df,
        risk_df=risk_df,
        quality_history_df=None,
        feature_history_df=None,
    )

    regime_row = out[out["MetricName"] == "RegimeFlipRate_60d"].iloc[0]
    risk_row = out[out["MetricName"] == "RiskScoreVol_60d"].iloc[0]
    assert regime_row["DriftScore"] == 0.0
    assert "Level changes=0" in risk_row["Notes"]

