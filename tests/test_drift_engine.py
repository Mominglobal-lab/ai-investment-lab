from __future__ import annotations

import numpy as np
import pandas as pd

from ai_models.drift_engine import compute_feature_drift


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

