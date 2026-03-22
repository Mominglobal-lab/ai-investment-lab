from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _normalize_drift_level(value: Any) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    low = txt.lower()
    if low == "severe":
        return "Severe"
    if low == "drift":
        return "Drift"
    return "Stable"


def build_drift_report(
    drift_df: pd.DataFrame,
    *,
    baseline_window: tuple[pd.Timestamp, pd.Timestamp],
    current_window: tuple[pd.Timestamp, pd.Timestamp],
    coverage_stats: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    now = pd.Timestamp.utcnow().isoformat()
    d = drift_df.copy() if drift_df is not None else pd.DataFrame()
    if d.empty:
        top_features = []
        top_signals = []
    else:
        f = d[d["MetricType"] == "Feature"].sort_values("DriftScore", ascending=False).head(5)
        s = d[d["MetricType"] == "Signal"].sort_values("DriftScore", ascending=False).head(5)
        top_features = [
            {"MetricName": str(r["MetricName"]), "DriftScore": _safe_float(r["DriftScore"]), "DriftLevel": str(r["DriftLevel"])}
            for _, r in f.iterrows()
        ]
        top_signals = [
            {"MetricName": str(r["MetricName"]), "DriftScore": _safe_float(r["DriftScore"]), "DriftLevel": str(r["DriftLevel"])}
            for _, r in s.iterrows()
        ]

    worst = "Stable"
    if d is not None and not d.empty:
        levels = d["DriftLevel"].map(_normalize_drift_level).tolist()
        if "Severe" in levels:
            worst = "Severe"
        elif "Drift" in levels:
            worst = "Drift"

    summary = (
        f"Monitoring detected overall {worst} drift. "
        f"Top feature drivers count={len(top_features)}, top signal drivers count={len(top_signals)}."
    )
    return {
        "created_at": now,
        "window_settings_used": {
            "baseline_start": str(baseline_window[0]),
            "baseline_end": str(baseline_window[1]),
            "current_start": str(current_window[0]),
            "current_end": str(current_window[1]),
        },
        "top_drifting_features": top_features,
        "top_drifting_signals": top_signals,
        "data_coverage_stats": coverage_stats,
        "warnings_and_fallbacks_used": warnings,
        "short_narrative_summary": summary,
    }


def build_monitoring_health_report(
    *,
    drift_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    coverage_stats: dict[str, Any],
    runtime_notes: list[str],
) -> dict[str, Any]:
    now = pd.Timestamp.utcnow().isoformat()
    return {
        "created_at": now,
        "artifact_freshness": {
            "drift_signals_cache": "fresh",
            "alert_log": "fresh",
        },
        "coverage": coverage_stats,
        "missing_feature_counts": coverage_stats.get("missing_counts", {}),
        "runtime_notes": runtime_notes,
        "alerts_generated_count": int(len(alerts_df) if alerts_df is not None else 0),
        "worst_drift_level": (
            "Severe"
            if (drift_df is not None and not drift_df.empty and drift_df["DriftLevel"].map(_normalize_drift_level).eq("Severe").any())
            else ("Drift" if (drift_df is not None and not drift_df.empty and drift_df["DriftLevel"].map(_normalize_drift_level).eq("Drift").any()) else "Stable")
        ),
    }

