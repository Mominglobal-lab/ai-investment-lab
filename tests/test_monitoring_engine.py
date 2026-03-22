from __future__ import annotations

import pandas as pd

from ai_models.monitoring_engine import build_drift_report, build_monitoring_health_report


def test_build_drift_report_sanitizes_bad_drift_scores():
    drift_df = pd.DataFrame(
        [
            {"MetricName": "Momentum_12M", "MetricType": "Feature", "DriftScore": "bad", "DriftLevel": "Drift"},
            {"MetricName": "RegimeFlipRate_60d", "MetricType": "Signal", "DriftScore": float("inf"), "DriftLevel": "Severe"},
        ]
    )

    report = build_drift_report(
        drift_df,
        baseline_window=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        current_window=(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
        coverage_stats={},
        warnings=[],
    )

    assert report["top_drifting_features"][0]["DriftScore"] == 0.0
    assert report["top_drifting_signals"][0]["DriftScore"] == 0.0


def test_monitoring_reports_normalize_drift_level_strings():
    drift_df = pd.DataFrame(
        [
            {"MetricName": "Momentum_12M", "MetricType": "Feature", "DriftScore": 0.12, "DriftLevel": " drift "},
            {"MetricName": "RiskScoreVol_60d", "MetricType": "Signal", "DriftScore": 0.40, "DriftLevel": "SEVERE"},
        ]
    )

    report = build_drift_report(
        drift_df,
        baseline_window=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        current_window=(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
        coverage_stats={},
        warnings=[],
    )

    health = build_monitoring_health_report(
        drift_df=drift_df,
        alerts_df=pd.DataFrame(),
        coverage_stats={},
        runtime_notes=[],
    )

    assert "overall Severe drift" in report["short_narrative_summary"]
    assert health["worst_drift_level"] == "Severe"


def test_build_drift_report_sorts_with_malformed_drift_scores():
    drift_df = pd.DataFrame(
        [
            {"MetricName": "FeatureA", "MetricType": "Feature", "DriftScore": "bad", "DriftLevel": "Drift"},
            {"MetricName": "FeatureB", "MetricType": "Feature", "DriftScore": 0.4, "DriftLevel": "Severe"},
            {"MetricName": "SignalA", "MetricType": "Signal", "DriftScore": None, "DriftLevel": "Stable"},
            {"MetricName": "SignalB", "MetricType": "Signal", "DriftScore": 0.3, "DriftLevel": "Drift"},
        ]
    )

    report = build_drift_report(
        drift_df,
        baseline_window=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
        current_window=(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
        coverage_stats={},
        warnings=[],
    )

    assert report["top_drifting_features"][0]["MetricName"] == "FeatureB"
    assert report["top_drifting_signals"][0]["MetricName"] == "SignalB"
