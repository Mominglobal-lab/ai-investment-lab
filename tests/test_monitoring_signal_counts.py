from __future__ import annotations

import pandas as pd

from ai_models.alert_engine import generate_alerts
from ai_models.drift_engine import compute_signal_instability


def test_compute_signal_instability_does_not_count_first_row_as_flip():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    regime_df = pd.DataFrame(
        {
            "Date": dates,
            "RegimeLabel": ["Neutral"] * len(dates),
            "ConfidenceScore": [0.6] * len(dates),
        }
    )

    out = compute_signal_instability(regime_df=regime_df, risk_df=None)

    row = out[out["MetricName"] == "RegimeFlipRate_60d"].iloc[0]
    assert float(row["DriftScore"]) == 0.0


def test_compute_signal_instability_does_not_count_first_risk_level_as_change():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    risk_df = pd.DataFrame(
        {
            "Date": dates,
            "RiskScore": [20, 21, 22, 23, 24],
            "RiskLevel": ["Low"] * len(dates),
        }
    )

    out = compute_signal_instability(regime_df=None, risk_df=risk_df)

    row = out[out["MetricName"] == "RiskScoreVol_60d"].iloc[0]
    assert "Level changes=0" in str(row["Notes"])


def test_generate_alerts_does_not_emit_whipsaw_or_flip_alerts_for_constant_series():
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    regime_df = pd.DataFrame({"Date": dates, "RegimeLabel": ["Neutral"] * len(dates)})
    risk_df = pd.DataFrame(
        {
            "Date": dates,
            "RiskScore": [25.0] * len(dates),
            "RiskLevel": ["Low"] * len(dates),
        }
    )

    alerts = generate_alerts(
        drift_df=pd.DataFrame(),
        regime_df=regime_df,
        risk_df=risk_df,
        coverage_stats={"treasury_exists": True, "prices_rows": 50000, "expected_min_price_rows": 50000},
    )

    if not alerts.empty:
        assert "RegimeFlipExcess" not in set(alerts["AlertType"])
        assert "RiskWhipsaw" not in set(alerts["AlertType"])
