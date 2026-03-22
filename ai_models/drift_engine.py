from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    notes: str


def _count_state_changes(series: pd.Series) -> int:
    s = series.copy()
    prev = s.shift(1)
    changed = s.ne(prev)
    if len(changed) > 0:
        changed.iloc[0] = False
    return int(changed.sum())


def _state_flip_rate(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(_count_state_changes(series) / float(len(series) - 1))


def _psi_from_arrays(base: np.ndarray, curr: np.ndarray, n_bins: int = 10) -> float:
    base = base[np.isfinite(base)]
    curr = curr[np.isfinite(curr)]
    if len(base) < 20 or len(curr) < 20:
        return float("nan")
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(base, quantiles))
    if len(bins) < 3:
        return 0.0
    b_hist, _ = np.histogram(base, bins=bins)
    c_hist, _ = np.histogram(curr, bins=bins)
    b_pct = np.clip(b_hist / max(np.sum(b_hist), 1), 1e-6, None)
    c_pct = np.clip(c_hist / max(np.sum(c_hist), 1), 1e-6, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def _drift_level(psi: float) -> str:
    if not np.isfinite(psi):
        return "Stable"
    if psi < 0.10:
        return "Stable"
    if psi < 0.25:
        return "Drift"
    return "Severe"


def compute_feature_drift(
    feature_history_df: pd.DataFrame,
    baseline_window: tuple[pd.Timestamp, pd.Timestamp],
    current_window: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    if feature_history_df is None or feature_history_df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "MetricName",
                "MetricType",
                "DriftScore",
                "DriftLevel",
                "BaselineWindowStart",
                "BaselineWindowEnd",
                "CurrentWindowStart",
                "CurrentWindowEnd",
                "Notes",
            ]
        )

    df = feature_history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    b0, b1 = baseline_window
    c0, c1 = current_window

    feature_cols = [c for c in df.columns if c not in {"Date", "Ticker"}]
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        base_vals = pd.to_numeric(df[(df["Date"] >= b0) & (df["Date"] <= b1)][col], errors="coerce").values
        curr_vals = pd.to_numeric(df[(df["Date"] >= c0) & (df["Date"] <= c1)][col], errors="coerce").values
        psi = _psi_from_arrays(base_vals, curr_vals)
        rows.append(
            {
                "Date": pd.Timestamp(c1),
                "MetricName": col,
                "MetricType": "Feature",
                "DriftScore": float(psi) if np.isfinite(psi) else 0.0,
                "DriftLevel": _drift_level(float(psi) if np.isfinite(psi) else 0.0),
                "BaselineWindowStart": pd.Timestamp(b0),
                "BaselineWindowEnd": pd.Timestamp(b1),
                "CurrentWindowStart": pd.Timestamp(c0),
                "CurrentWindowEnd": pd.Timestamp(c1),
                "Notes": "PSI computed on pooled values.",
            }
        )
    return pd.DataFrame(rows)


def compute_signal_instability(
    regime_df: pd.DataFrame | None,
    risk_df: pd.DataFrame | None,
    quality_history_df: pd.DataFrame | None = None,
    feature_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.utcnow()

    if regime_df is not None and not regime_df.empty and {"Date", "RegimeLabel"}.issubset(set(regime_df.columns)):
        r = regime_df.copy()
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        r = r.dropna(subset=["Date"]).sort_values("Date").tail(60)
        flip_rate = _state_flip_rate(r["RegimeLabel"]) if len(r) > 1 else 0.0
        avg_conf = float(pd.to_numeric(r.get("ConfidenceScore"), errors="coerce").mean()) if "ConfidenceScore" in r.columns else np.nan
        score = flip_rate
        rows.append(
            {
                "Date": now,
                "MetricName": "RegimeFlipRate_60d",
                "MetricType": "Signal",
                "DriftScore": score,
                "DriftLevel": "Severe" if score >= 0.20 else ("Drift" if score >= 0.10 else "Stable"),
                "BaselineWindowStart": r["Date"].min() if not r.empty else pd.NaT,
                "BaselineWindowEnd": r["Date"].max() if not r.empty else pd.NaT,
                "CurrentWindowStart": r["Date"].min() if not r.empty else pd.NaT,
                "CurrentWindowEnd": r["Date"].max() if not r.empty else pd.NaT,
                "Notes": f"Average confidence={avg_conf:.3f}" if np.isfinite(avg_conf) else "Average confidence unavailable.",
            }
        )

    if risk_df is not None and not risk_df.empty and {"Date", "RiskScore"}.issubset(set(risk_df.columns)):
        k = risk_df.copy()
        k["Date"] = pd.to_datetime(k["Date"], errors="coerce")
        k["RiskScore"] = pd.to_numeric(k["RiskScore"], errors="coerce")
        k = k.dropna(subset=["Date", "RiskScore"]).sort_values("Date").tail(60)
        score_vol = float(k["RiskScore"].std(ddof=1)) if len(k) > 1 else 0.0
        level_changes = 0
        if "RiskLevel" in k.columns:
            level_changes = _count_state_changes(k["RiskLevel"])
        whipsaw = level_changes > 4
        rows.append(
            {
                "Date": now,
                "MetricName": "RiskScoreVol_60d",
                "MetricType": "Signal",
                "DriftScore": score_vol,
                "DriftLevel": "Severe" if score_vol >= 12 else ("Drift" if score_vol >= 6 else "Stable"),
                "BaselineWindowStart": k["Date"].min() if not k.empty else pd.NaT,
                "BaselineWindowEnd": k["Date"].max() if not k.empty else pd.NaT,
                "CurrentWindowStart": k["Date"].min() if not k.empty else pd.NaT,
                "CurrentWindowEnd": k["Date"].max() if not k.empty else pd.NaT,
                "Notes": f"Level changes={level_changes}; whipsaw={whipsaw}",
            }
        )

    # Quality instability fallback via feature drift proxy if no history is present.
    if quality_history_df is None and feature_history_df is not None and not feature_history_df.empty:
        df = feature_history_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        if not df.empty:
            recent = df[df["Date"] >= (df["Date"].max() - pd.Timedelta(days=90))]
            old = df[(df["Date"] < recent["Date"].min()) & (df["Date"] >= (recent["Date"].min() - pd.Timedelta(days=252)))]
            # proxy on selected quality-related features
            feat_cols = [c for c in ["RevenueGrowth", "EBITDAMargin", "ROE", "FCFMargin"] if c in df.columns]
            psis: list[float] = []
            for c in feat_cols:
                ps = _psi_from_arrays(pd.to_numeric(old[c], errors="coerce").values, pd.to_numeric(recent[c], errors="coerce").values)
                if np.isfinite(ps):
                    psis.append(float(ps))
            proxy = float(np.nanmean(psis)) if psis else 0.0
            rows.append(
                {
                    "Date": now,
                    "MetricName": "QualityProxyDrift",
                    "MetricType": "Signal",
                    "DriftScore": proxy,
                    "DriftLevel": _drift_level(proxy),
                    "BaselineWindowStart": old["Date"].min() if not old.empty else pd.NaT,
                    "BaselineWindowEnd": old["Date"].max() if not old.empty else pd.NaT,
                    "CurrentWindowStart": recent["Date"].min() if not recent.empty else pd.NaT,
                    "CurrentWindowEnd": recent["Date"].max() if not recent.empty else pd.NaT,
                    "Notes": "Proxy drift from quality-related feature distributions.",
                }
            )

    return pd.DataFrame(rows)

