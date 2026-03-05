from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_read(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    canon = {c.lower().replace("_", "").replace(" ", ""): c for c in columns}
    for cand in candidates:
        k = cand.lower().replace("_", "").replace(" ", "")
        if k in canon:
            return canon[k]
    return None


def _yield_inversion_series(treasury: pd.DataFrame | None, index_dates: pd.DatetimeIndex) -> pd.Series:
    if treasury is None or treasury.empty:
        return pd.Series(0.0, index=index_dates)
    t = treasury.copy()
    dcol = _detect_col(list(t.columns), ["Date", "Timestamp", "AsOfDate"])
    y10_col = _detect_col(list(t.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
    y2_col = _detect_col(list(t.columns), ["2Y", "DGS2", "Yield2Y", "UST2Y"])
    y3m_col = _detect_col(list(t.columns), ["3M", "DGS3MO", "Yield3M", "UST3M"])
    if dcol is None or y10_col is None or (y2_col is None and y3m_col is None):
        return pd.Series(0.0, index=index_dates)

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(t[dcol], errors="coerce")
    y10 = pd.to_numeric(t[y10_col], errors="coerce")
    y2 = pd.to_numeric(t[y2_col], errors="coerce") if y2_col else np.nan
    y3 = pd.to_numeric(t[y3m_col], errors="coerce") if y3m_col else np.nan
    slope = (y10 - y2) if y2_col else (y10 - y3)
    out["inv"] = (slope < 0).astype(float)
    out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
    return out["inv"].reindex(index_dates).ffill().fillna(0.0)


def run_systemic_risk_detector(
    *,
    prices_path: str = "data/prices_cache.parquet",
    treasury_path: str = "data/treasury_yields_cache.parquet",
    benchmark_ticker: str = "SPY",
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
    px = px.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"])

    pivot = px.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last").sort_index()
    if benchmark_ticker.upper() not in pivot.columns:
        raise ValueError(f"Benchmark ticker {benchmark_ticker} not found in prices cache")
    b = pivot[benchmark_ticker.upper()].dropna()
    r = b.pct_change().dropna()
    dates = r.index

    vol21 = r.rolling(21, min_periods=5).std() * np.sqrt(252)
    vol252 = r.rolling(252, min_periods=30).std() * np.sqrt(252)
    vol_exp = ((vol21 / vol252) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    roll_peak = b.rolling(63, min_periods=10).max()
    dd = (b / roll_peak) - 1.0
    dd_accel = (-dd.diff(5)).clip(lower=0.0).reindex(dates).fillna(0.0)

    corr_tickers = [t for t in ["SPY", "QQQ", "IWM", "DIA"] if t in pivot.columns]
    corr_spike = pd.Series(0.0, index=dates)
    if len(corr_tickers) >= 2:
        corr_returns = pivot[corr_tickers].pct_change().dropna()
        vals: list[float] = []
        idx: list[pd.Timestamp] = []
        for i in range(20, len(corr_returns)):
            window = corr_returns.iloc[i - 20 : i + 1]
            cmat = window.corr().values
            if cmat.size <= 1:
                mean_corr = 0.0
            else:
                tri = cmat[np.triu_indices_from(cmat, k=1)]
                mean_corr = float(np.nanmean(tri))
            vals.append(mean_corr)
            idx.append(corr_returns.index[i])
        cser = pd.Series(vals, index=idx).clip(lower=0.0).fillna(0.0)
        corr_spike = cser.reindex(dates).ffill().fillna(0.0)

    treasury = _safe_read(treasury_path)
    inversion = _yield_inversion_series(treasury, dates)

    rate_shock = pd.Series(0.0, index=dates)
    if treasury is not None and not treasury.empty:
        dcol = _detect_col(list(treasury.columns), ["Date", "Timestamp", "AsOfDate"])
        y10 = _detect_col(list(treasury.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
        if dcol and y10:
            t = treasury.copy()
            t["Date"] = pd.to_datetime(t[dcol], errors="coerce")
            t["y10"] = pd.to_numeric(t[y10], errors="coerce")
            t = t.dropna(subset=["Date", "y10"]).set_index("Date").sort_index()
            shock = t["y10"].diff(5).abs()
            rate_shock = shock.reindex(dates).ffill().fillna(0.0)

    # Normalize indicators to [0,1] scale using robust percentile clipping.
    def _norm(s: pd.Series) -> pd.Series:
        x = s.fillna(0.0)
        p95 = float(np.nanpercentile(x.values, 95)) if len(x) else 1.0
        if p95 <= 0:
            p95 = 1.0
        return (x / p95).clip(lower=0.0, upper=1.0)

    n_vol = _norm(vol_exp)
    n_dd = _norm(dd_accel)
    n_corr = _norm(corr_spike)
    n_inv = inversion.clip(lower=0.0, upper=1.0)
    n_rate = _norm(rate_shock)

    score = (
        (0.30 * n_vol)
        + (0.25 * n_dd)
        + (0.20 * n_corr)
        + (0.15 * n_inv)
        + (0.10 * n_rate)
    ) * 100.0
    score = score.clip(lower=0.0, upper=100.0)

    risk_level = pd.cut(
        score,
        bins=[-1e-9, 35, 65, 100],
        labels=["Low", "Moderate", "Elevated"],
    ).astype(str)

    flags_list: list[str] = []
    explanations: list[str] = []
    for dt in dates:
        flags: list[str] = []
        if n_vol.loc[dt] > 0.7:
            flags.append("Volatility spike")
        if n_inv.loc[dt] > 0.5:
            flags.append("Yield inversion")
        if n_dd.loc[dt] > 0.7:
            flags.append("Drawdown acceleration")
        if n_corr.loc[dt] > 0.7:
            flags.append("Correlation spike")
        if n_rate.loc[dt] > 0.7:
            flags.append("Rate shock")
        flags_list.append("; ".join(flags) if flags else "None")

        drivers = sorted(
            [
                ("volatility expansion", float(n_vol.loc[dt])),
                ("drawdown acceleration", float(n_dd.loc[dt])),
                ("cross-asset correlation", float(n_corr.loc[dt])),
                ("yield inversion", float(n_inv.loc[dt])),
                ("rate shock", float(n_rate.loc[dt])),
            ],
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        explanations.append("Primary drivers: " + ", ".join([d[0] for d in drivers]))

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates, errors="coerce"),
            "RiskScore": [float(x) for x in score.tolist()],
            "RiskLevel": risk_level.tolist(),
            "RiskFlags": flags_list,
            "Explanation": explanations,
        }
    )
    return out.reset_index(drop=True)

