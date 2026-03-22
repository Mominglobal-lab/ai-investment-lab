from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def _json_safe(value):
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


def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    canon = {c.lower().replace("_", "").replace(" ", ""): c for c in columns}
    for cand in candidates:
        k = cand.lower().replace("_", "").replace(" ", "")
        if k in canon:
            return canon[k]
    return None


def _normalize_regime_label(value: object) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    low = txt.lower()
    if low in {"", "nan", "none"}:
        return "Neutral"
    if low == "risk on":
        return "Risk On"
    if low == "risk off":
        return "Risk Off"
    if low == "neutral":
        return "Neutral"
    return txt


def _normalize_risk_level(value: object) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    low = txt.lower()
    if low in {"", "nan", "none"}:
        return "Unknown"
    if low == "low":
        return "Low"
    if low == "moderate":
        return "Moderate"
    if low == "elevated":
        return "Elevated"
    return txt


def _build_benchmark_indicators(prices_df: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    p = prices_df.copy()
    p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p["AdjClose"] = pd.to_numeric(p["AdjClose"], errors="coerce")
    benchmark_symbol = str(benchmark_ticker or "").strip().upper()
    b = p[p["Ticker"] == benchmark_symbol].dropna(subset=["Date", "AdjClose"]).sort_values("Date")
    s = b.set_index("Date")["AdjClose"].sort_index()
    r = s.pct_change()
    vol63 = r.rolling(63, min_periods=10).std() * np.sqrt(252)
    trend63 = s.pct_change(63)
    return pd.DataFrame({"Date": s.index, "BenchmarkVol_63d": vol63.values, "BenchmarkTrend_63d": trend63.values})


def _regime_rule_and_explanation(label: str, inv: float, vr: float, steep: float) -> tuple[str, str]:
    inversion = inv > 0.5
    vol_rising = vr > 0.5
    slope_steepening = steep > 0.5

    if label == "Risk Off" and inversion and vol_rising:
        return "inversion_and_volatility_rising", "Risk Off because curve is inverted and market volatility is rising."
    if label == "Risk On" and slope_steepening and not vol_rising:
        return "steepening_and_volatility_falling", "Risk On because curve is steepening while volatility is stable/falling."
    if label == "Risk Off":
        return "risk_off_label_without_full_confirmation", "Risk Off label is present, but the inversion and volatility conditions are not both confirmed."
    if label == "Risk On":
        return "risk_on_label_without_full_confirmation", "Risk On label is present, but the steepening and falling-volatility conditions are not both confirmed."
    return "neutral_default", "Neutral because risk-on/off conditions are not jointly satisfied."


def _build_yield_indicators(treasury_df: pd.DataFrame | None, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if treasury_df is None or treasury_df.empty:
        out = pd.DataFrame({"Date": dates})
        out["YC_Slope_10Y_2Y"] = np.nan
        out["YC_Slope_10Y_3M"] = np.nan
        out["YC_Inversion"] = 0.0
        out["YC_Volatility"] = np.nan
        return out

    t = treasury_df.copy()
    dcol = _detect_col(list(t.columns), ["Date", "Timestamp", "AsOfDate"])
    y10 = _detect_col(list(t.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
    y2 = _detect_col(list(t.columns), ["2Y", "DGS2", "Yield2Y", "UST2Y"])
    y3 = _detect_col(list(t.columns), ["3M", "DGS3MO", "Yield3M", "UST3M"])
    if dcol is None or y10 is None or (y2 is None and y3 is None):
        out = pd.DataFrame({"Date": dates})
        out["YC_Slope_10Y_2Y"] = np.nan
        out["YC_Slope_10Y_3M"] = np.nan
        out["YC_Inversion"] = 0.0
        out["YC_Volatility"] = np.nan
        return out

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(t[dcol], errors="coerce")
    y10s = pd.to_numeric(t[y10], errors="coerce")
    y2s = pd.to_numeric(t[y2], errors="coerce") if y2 else np.nan
    y3s = pd.to_numeric(t[y3], errors="coerce") if y3 else np.nan
    out["YC_Slope_10Y_2Y"] = y10s - y2s if y2 else np.nan
    out["YC_Slope_10Y_3M"] = y10s - y3s if y3 else np.nan
    slope = out["YC_Slope_10Y_2Y"].fillna(out["YC_Slope_10Y_3M"])
    out["YC_Inversion"] = (slope < 0).astype(float)
    out["YC_Volatility"] = slope.rolling(21, min_periods=5).std()
    out = out.dropna(subset=["Date"]).sort_values("Date")
    out = out.set_index("Date").reindex(dates).ffill().reset_index().rename(columns={"index": "Date"})
    return out


def build_regime_evidence(
    prices_df: pd.DataFrame,
    treasury_df: pd.DataFrame | None,
    regime_df: pd.DataFrame,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    if regime_df is None or regime_df.empty:
        return pd.DataFrame(columns=["Date", "RegimeLabel", "ConfidenceScore", "RuleTriggered", "EvidencePointsJSON", "ShortExplanation"])

    r = regime_df.copy()
    r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
    r = r.dropna(subset=["Date"]).sort_values("Date")
    dates = pd.DatetimeIndex(r["Date"])

    b = _build_benchmark_indicators(prices_df, benchmark_ticker=benchmark_ticker)
    b = b.set_index("Date").reindex(dates).ffill().reset_index().rename(columns={"index": "Date"})
    y = _build_yield_indicators(treasury_df, dates)

    m = r.merge(b, on="Date", how="left").merge(y, on="Date", how="left")
    m["VolatilityRising"] = (pd.to_numeric(m["BenchmarkVol_63d"], errors="coerce").diff(21) > 0).astype(float)
    slope_ref = pd.to_numeric(m["YC_Slope_10Y_2Y"], errors="coerce").fillna(pd.to_numeric(m["YC_Slope_10Y_3M"], errors="coerce"))
    m["SlopeSteepening"] = (slope_ref.diff(21) > 0).astype(float)

    rows: list[dict[str, object]] = []
    for _, row in m.iterrows():
        label = _normalize_regime_label(row.get("RegimeLabel", "Neutral"))
        inv = float(row.get("YC_Inversion", 0.0)) if pd.notna(row.get("YC_Inversion")) else 0.0
        vr = float(row.get("VolatilityRising", 0.0)) if pd.notna(row.get("VolatilityRising")) else 0.0
        steep = float(row.get("SlopeSteepening", 0.0)) if pd.notna(row.get("SlopeSteepening")) else 0.0

        rule, short = _regime_rule_and_explanation(label, inv, vr, steep)

        evidence = {
            "YC_Slope_10Y_2Y": float(row.get("YC_Slope_10Y_2Y")) if pd.notna(row.get("YC_Slope_10Y_2Y")) else None,
            "YC_Slope_10Y_3M": float(row.get("YC_Slope_10Y_3M")) if pd.notna(row.get("YC_Slope_10Y_3M")) else None,
            "YC_Inversion": float(inv),
            "BenchmarkVol_63d": float(row.get("BenchmarkVol_63d")) if pd.notna(row.get("BenchmarkVol_63d")) else None,
            "BenchmarkTrend_63d": float(row.get("BenchmarkTrend_63d")) if pd.notna(row.get("BenchmarkTrend_63d")) else None,
            "VolatilityRising": float(vr),
            "SlopeSteepening": float(steep),
        }

        rows.append(
            {
                "Date": row["Date"],
                "RegimeLabel": label,
                "ConfidenceScore": float(row.get("ConfidenceScore", np.nan)),
                "RuleTriggered": rule,
                "EvidencePointsJSON": json.dumps(_json_safe(evidence), sort_keys=True, allow_nan=False),
                "ShortExplanation": short,
            }
        )
    return pd.DataFrame(rows)


def build_risk_evidence(prices_df: pd.DataFrame, treasury_df: pd.DataFrame | None, risk_df: pd.DataFrame) -> pd.DataFrame:
    if risk_df is None or risk_df.empty:
        return pd.DataFrame(columns=["Date", "RiskScore", "RiskLevel", "TopRiskDrivers", "EvidencePointsJSON", "ShortExplanation"])

    r = risk_df.copy()
    r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
    r["RiskScore"] = pd.to_numeric(r["RiskScore"], errors="coerce")
    r = r.dropna(subset=["Date", "RiskScore"]).sort_values("Date")

    # Approximate underlying indicators from risk score dynamics (deterministic fallback when raw indicator cache absent).
    out_rows: list[dict[str, object]] = []
    score = r["RiskScore"].reset_index(drop=True)
    vol_exp = score.diff().clip(lower=0).fillna(0)
    rapid_dd = score.rolling(5, min_periods=2).max() - score
    corr_spike = score.rolling(10, min_periods=3).std().fillna(0)
    inv = pd.Series(0.0, index=score.index)
    rate_shock = score.diff(3).abs().fillna(0)
    if treasury_df is not None and not treasury_df.empty:
        # If treasury exists, use mild inversion proxy from 10Y-2Y.
        t = treasury_df.copy()
        dcol = _detect_col(list(t.columns), ["Date", "Timestamp", "AsOfDate"])
        y10 = _detect_col(list(t.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
        y2 = _detect_col(list(t.columns), ["2Y", "DGS2", "Yield2Y", "UST2Y"])
        y3 = _detect_col(list(t.columns), ["3M", "DGS3MO", "Yield3M", "UST3M"])
        if dcol and y10 and (y2 or y3):
            t["Date"] = pd.to_datetime(t[dcol], errors="coerce")
            s = pd.to_numeric(t[y10], errors="coerce") - (pd.to_numeric(t[y2], errors="coerce") if y2 else pd.to_numeric(t[y3], errors="coerce"))
            t2 = pd.DataFrame({"Date": t["Date"], "inv": (s < 0).astype(float)}).dropna(subset=["Date"]).sort_values("Date")
            merged = pd.merge_asof(r[["Date"]], t2, on="Date", direction="backward")
            inv = merged["inv"].fillna(0.0).reset_index(drop=True)

    for i, row in r.reset_index(drop=True).iterrows():
        indicators = {
            "VolatilityExpansion": float(vol_exp.iloc[i]),
            "RapidDrawdown": float(rapid_dd.iloc[i]),
            "CorrelationSpike": float(corr_spike.iloc[i]),
            "YieldCurveInversion": float(inv.iloc[i]) if len(inv) > i else 0.0,
            "RateShock": float(rate_shock.iloc[i]),
        }
        flags = {
            "VolatilityExpansion_flag": indicators["VolatilityExpansion"] > 2.0,
            "RapidDrawdown_flag": indicators["RapidDrawdown"] > 5.0,
            "CorrelationSpike_flag": indicators["CorrelationSpike"] > 8.0,
            "YieldCurveInversion_flag": indicators["YieldCurveInversion"] > 0.5,
            "RateShock_flag": indicators["RateShock"] > 4.0,
        }
        drivers = sorted(indicators.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, _ in drivers[:3]]
        short = f"Risk level driven by: {', '.join(top)}."
        ev = {**indicators, **{k: bool(v) for k, v in flags.items()}}
        out_rows.append(
            {
                "Date": row["Date"],
                "RiskScore": float(row["RiskScore"]),
                "RiskLevel": _normalize_risk_level(row.get("RiskLevel", "Unknown")),
                "TopRiskDrivers": ", ".join(top),
                "EvidencePointsJSON": json.dumps(_json_safe(ev), sort_keys=True, allow_nan=False),
                "ShortExplanation": short,
            }
        )

    return pd.DataFrame(out_rows)

