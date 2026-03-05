from __future__ import annotations

import json
import pandas as pd

from ai_models.alert_engine import generate_alerts


def test_alert_engine_generates_critical_with_valid_evidence():
    drift_df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-01-10"),
                "MetricName": "Momentum_252d",
                "MetricType": "Feature",
                "DriftScore": 0.30,
                "DriftLevel": "Severe",
            }
        ]
    )
    regime_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-10-01", periods=60, freq="B"),
            "RegimeLabel": ["Risk On", "Risk Off"] * 30,
            "ConfidenceScore": [0.6] * 60,
        }
    )
    risk_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-10-01", periods=60, freq="B"),
            "RiskScore": [50] * 55 + [70, 75, 80, 86, 90],
            "RiskLevel": ["Moderate"] * 55 + ["Elevated"] * 5,
        }
    )
    alerts = generate_alerts(
        drift_df=drift_df,
        regime_df=regime_df,
        risk_df=risk_df,
        coverage_stats={"treasury_exists": False, "prices_rows": 10000, "expected_min_price_rows": 50000},
    )
    assert not alerts.empty
    assert (alerts["Severity"] == "Critical").any()
    assert (alerts["SuggestedAction"].astype(str).str.len() > 0).all()
    for ev in alerts["EvidenceJSON"].head(5).tolist():
        parsed = json.loads(ev)
        assert isinstance(parsed, dict)

