from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ai_models.path_utils import resolve_project_path


def _safe_read(path: str) -> pd.DataFrame | None:
    resolved = resolve_project_path(path)
    try:
        return pd.read_parquet(resolved)
    except Exception:
        return None


def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    canon = {c.lower().replace("_", "").replace(" ", ""): c for c in columns}
    for cand in candidates:
        k = cand.lower().replace("_", "").replace(" ", "")
        if k in canon:
            return canon[k]
    return None


def _build_yield_signals(treasury: pd.DataFrame | None, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if treasury is None or treasury.empty:
        return pd.DataFrame({"Date": dates, "YieldSlope": np.nan, "YieldInverted": 0.0, "YieldVolatility": np.nan})

    t = treasury.copy()
    date_col = _detect_col(list(t.columns), ["Date", "Timestamp", "AsOfDate"])
    y10_col = _detect_col(list(t.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
    y2_col = _detect_col(list(t.columns), ["2Y", "DGS2", "Yield2Y", "UST2Y"])
    y3m_col = _detect_col(list(t.columns), ["3M", "DGS3MO", "Yield3M", "UST3M"])
    if date_col is None or y10_col is None or (y2_col is None and y3m_col is None):
        return pd.DataFrame({"Date": dates, "YieldSlope": np.nan, "YieldInverted": 0.0, "YieldVolatility": np.nan})

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(t[date_col], errors="coerce")
    y10 = pd.to_numeric(t[y10_col], errors="coerce")
    y2 = pd.to_numeric(t[y2_col], errors="coerce") if y2_col else np.nan
    y3m = pd.to_numeric(t[y3m_col], errors="coerce") if y3m_col else np.nan
    slope = (y10 - y2) if y2_col else (y10 - y3m)
    out["YieldSlope"] = slope
    out["YieldInverted"] = (slope < 0).astype(float)
    out["YieldVolatility"] = slope.rolling(21, min_periods=5).std()
    out = out.dropna(subset=["Date"]).sort_values("Date")
    out = out.set_index("Date").reindex(dates).ffill().reset_index().rename(columns={"index": "Date"})
    return out


def _apply_persistence(raw_labels: list[str], persistence_days: int) -> list[str]:
    if not raw_labels:
        return []
    out = [raw_labels[0]]
    candidate = raw_labels[0]
    streak = 1
    current = raw_labels[0]
    for lbl in raw_labels[1:]:
        if lbl == candidate:
            streak += 1
        else:
            candidate = lbl
            streak = 1
        if candidate != current and streak >= persistence_days:
            current = candidate
        out.append(current)
    return out


def run_regime_detection_model(
    *,
    prices_path: str = "data/prices_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    benchmark_ticker: str = "SPY",
    persistence_days: int = 3,
) -> pd.DataFrame:
    prices = _safe_read(prices_path)
    if prices is None or prices.empty:
        raise FileNotFoundError(f"Price cache missing or empty: {prices_path}")
    if not {"Ticker", "Date", "AdjClose"}.issubset(prices.columns):
        raise ValueError("Price cache missing required columns: Ticker, Date, AdjClose")

    px = prices.copy()
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px["AdjClose"] = pd.to_numeric(px["AdjClose"], errors="coerce")
    px = px.dropna(subset=["Ticker", "Date", "AdjClose"])

    benchmark_symbol = str(benchmark_ticker or "").strip().upper()
    b = px[px["Ticker"] == benchmark_symbol].sort_values("Date")
    if b.empty:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} not found in prices cache")

    b = b.set_index("Date")["AdjClose"].sort_index()
    ret = b.pct_change()
    vol = ret.rolling(21, min_periods=5).std() * np.sqrt(252)
    vol_chg = vol.diff(5)
    trend = b.pct_change(63)

    dates = b.index
    treasury = _safe_read(treasury_path)
    y = _build_yield_signals(treasury, dates)
    y = y.set_index("Date")

    raw_labels: list[str] = []
    conf: list[float] = []
    for dt in dates:
        inv = float(y.loc[dt, "YieldInverted"]) if dt in y.index else 0.0
        slope = float(y.loc[dt, "YieldSlope"]) if dt in y.index and pd.notna(y.loc[dt, "YieldSlope"]) else np.nan
        vchg = float(vol_chg.loc[dt]) if pd.notna(vol_chg.loc[dt]) else 0.0
        tr = float(trend.loc[dt]) if pd.notna(trend.loc[dt]) else 0.0

        if inv > 0 and vchg > 0:
            label = "Risk Off"
            score = 0.60 + min(0.35, abs(vchg) * 4.0)
        elif pd.notna(slope) and slope > 0 and vchg <= 0 and tr > 0:
            label = "Risk On"
            score = 0.60 + min(0.30, abs(tr) * 0.8)
        else:
            label = "Neutral"
            score = 0.55
        raw_labels.append(label)
        conf.append(float(max(0.0, min(score, 0.95))))

    smoothed = _apply_persistence(raw_labels, max(1, int(persistence_days)))
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates, errors="coerce"),
            "RegimeLabel": smoothed,
            "ConfidenceScore": [float(x) for x in conf],
        }
    )
    return out.dropna(subset=["Date"]).reset_index(drop=True)

