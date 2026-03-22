from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _normalize_alert_date(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts


def _mk_alert(
    *,
    date,
    severity: str,
    alert_type: str,
    title: str,
    description: str,
    evidence: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    return {
        "Date": _normalize_alert_date(date),
        "Severity": severity,
        "AlertType": alert_type,
        "Title": title,
        "Description": description,
        "EvidenceJSON": json.dumps(_json_safe(evidence), sort_keys=True, allow_nan=False),
        "SuggestedAction": action,
    }


def _count_state_changes(series: pd.Series) -> int:
    s = series.copy()
    prev = s.shift(1)
    changed = s.ne(prev)
    if len(changed) > 0:
        changed.iloc[0] = False
    return int(changed.sum())


def _normalize_state_label(value: Any) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    low = txt.lower()
    if low in {"risk on", "risk off", "neutral", "low", "moderate", "elevated"}:
        return " ".join([part.capitalize() for part in low.split(" ")])
    if low in {"", "nan", "none"}:
        return ""
    return txt


def _normalize_drift_level(value: Any) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    low = txt.lower()
    if low == "severe":
        return "Severe"
    if low == "drift":
        return "Drift"
    return "Stable"


def generate_alerts(
    drift_df: pd.DataFrame,
    regime_df: pd.DataFrame | None,
    risk_df: pd.DataFrame | None,
    coverage_stats: dict[str, Any],
) -> pd.DataFrame:
    alerts: list[dict[str, Any]] = []
    now = _normalize_alert_date(pd.Timestamp.utcnow())

    if drift_df is not None and not drift_df.empty:
        for _, r in drift_df.iterrows():
            score = _safe_float(r.get("DriftScore", 0.0))
            name = str(r.get("MetricName", "UnknownMetric"))
            level = _normalize_drift_level(r.get("DriftLevel", "Stable"))
            if level == "Severe":
                alerts.append(
                    _mk_alert(
                        date=r.get("Date", now),
                        severity="Critical",
                        alert_type="SevereFeatureDrift" if str(r.get("MetricType")) == "Feature" else "SignalInstability",
                        title=f"Severe drift detected: {name}",
                        description=f"{name} drift score is {score:.3f}, exceeding severe threshold.",
                        evidence={"MetricName": name, "DriftScore": score, "DriftLevel": level},
                        action="Review feature coverage and retrain/recalibrate affected models if persistent.",
                    )
                )
            elif level == "Drift":
                alerts.append(
                    _mk_alert(
                        date=r.get("Date", now),
                        severity="Warning",
                        alert_type="SignalInstability" if str(r.get("MetricType")) == "Signal" else "FeatureDrift",
                        title=f"Drift warning: {name}",
                        description=f"{name} drift score is {score:.3f}, above drift threshold.",
                        evidence={"MetricName": name, "DriftScore": score, "DriftLevel": level},
                        action="Monitor trend and validate model robustness over next refresh cycles.",
                    )
                )

    if regime_df is not None and not regime_df.empty and "RegimeLabel" in regime_df.columns:
        recent = regime_df.copy()
        recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
        recent["RegimeLabel"] = recent["RegimeLabel"].map(_normalize_state_label)
        recent = recent.dropna(subset=["Date"]).sort_values("Date").tail(60)
        if len(recent) > 2:
            flips = _count_state_changes(recent["RegimeLabel"])
            if flips > 8:
                alerts.append(
                    _mk_alert(
                        date=recent["Date"].iloc[-1],
                        severity="Critical",
                        alert_type="RegimeFlipExcess",
                        title="Regime is flipping too frequently",
                        description=f"Regime label changed {flips} times in the last 60 business days.",
                        evidence={"FlipCount60d": flips},
                        action="Treat regime signal as unstable; reduce sensitivity in tactical allocation decisions.",
                    )
                )

    if risk_df is not None and not risk_df.empty and {"Date", "RiskScore"}.issubset(set(risk_df.columns)):
        recent = risk_df.copy()
        recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
        recent["RiskScore"] = pd.to_numeric(recent["RiskScore"], errors="coerce")
        recent.loc[~pd.Series(recent["RiskScore"]).apply(math.isfinite), "RiskScore"] = pd.NA
        if "RiskLevel" in recent.columns:
            recent["RiskLevel"] = recent["RiskLevel"].map(_normalize_state_label)
        recent = recent.dropna(subset=["Date", "RiskScore"]).sort_values("Date").tail(60)
        if not recent.empty:
            latest = _safe_float(recent["RiskScore"].iloc[-1])
            slope = _safe_float(recent["RiskScore"].diff(5).iloc[-1]) if len(recent) > 6 else 0.0
            if latest >= 65 and slope > 5:
                alerts.append(
                    _mk_alert(
                        date=recent["Date"].iloc[-1],
                        severity="Critical",
                        alert_type="SignalInstability",
                        title="Elevated systemic risk rising rapidly",
                        description=f"RiskScore is {latest:.1f} with sharp 5-day increase of {slope:.1f}.",
                        evidence={"RiskScore": latest, "RiskScoreDelta5d": slope},
                        action="Review hedging posture and tighten risk limits while elevated trend persists.",
                    )
                )
            if "RiskLevel" in recent.columns:
                changes = _count_state_changes(recent["RiskLevel"])
                if changes > 4:
                    alerts.append(
                        _mk_alert(
                            date=recent["Date"].iloc[-1],
                            severity="Warning",
                            alert_type="RiskWhipsaw",
                            title="Risk level whipsaw detected",
                            description=f"Risk level changed {changes} times in 60 business days.",
                            evidence={"RiskLevelChanges60d": changes},
                            action="Use smoothed risk indicators for decision gating until whipsaw subsides.",
                        )
                    )

    # Coverage / missing-data informational alerts.
    treasury_exists = bool(coverage_stats.get("treasury_exists", True))
    if not treasury_exists:
        alerts.append(
            _mk_alert(
                date=now,
                severity="Info",
                alert_type="TreasuryDataMissing",
                title="Treasury data unavailable",
                description="Treasury cache is missing; yield-curve based monitoring is degraded.",
                evidence={"treasury_exists": treasury_exists},
                action="Refresh treasury cache to restore full regime/risk evidence coverage.",
            )
        )
    price_rows = int(coverage_stats.get("prices_rows", 0))
    if price_rows < int(coverage_stats.get("expected_min_price_rows", 50000)):
        alerts.append(
            _mk_alert(
                date=now,
                severity="Warning",
                alert_type="PriceCoverageDrop",
                title="Price coverage appears low",
                description=f"Observed price rows={price_rows}, below expected threshold.",
                evidence={"prices_rows": price_rows},
                action="Run price refresh and verify ticker coverage in price cache.",
            )
        )

    out = pd.DataFrame(alerts)
    if out.empty:
        out = pd.DataFrame(columns=["Date", "Severity", "AlertType", "Title", "Description", "EvidenceJSON", "SuggestedAction"])
    else:
        out = out.sort_values("Date", ascending=False).reset_index(drop=True)
    return out
