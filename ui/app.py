from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.cache_manager import get_cache_status, read_parquet_safe, save_parquet_atomic, write_json_report
from data_pipeline.data_fetcher import (
    FI_SCHEMA_COLUMNS,
    SCHEMA_COLUMNS,
    fetch_ticker_details,
    fetch_universe_tickers,
    refresh_fundamentals_yfinance,
)
from data_pipeline.run_pipeline import run_fixed_income_pipeline

st.set_page_config(page_title="Investment Lab", layout="wide")

STOCK_UNIVERSE_OPTIONS = ["S&P 500", "Nasdaq 100"]
FI_UNIVERSE_OPTIONS = ["US Treasuries", "Bond ETFs"]
MAX_AGE_DAYS = 7
MIN_REFRESH_SUCCESS_RATIO = 0.25


@dataclass(frozen=True)
class StockUpdateResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    rate_limited: bool
    errors_sample: list[str]
    wrote_cache: bool
    reason: str


def _stock_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace("&", "and").replace(" ", "").replace("-", "")
    if key in {"sandp500", "sp500"}:
        return "data/fundamentals_cache_sp500.parquet", "data/fundamentals_health_report_sp500.json"
    if key in {"nasdaq100", "ndx"}:
        return "data/fundamentals_cache_nasdaq100.parquet", "data/fundamentals_health_report_nasdaq100.json"
    raise ValueError(f"Unsupported stock universe: {universe}")


def _fi_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace(" ", "").replace("-", "")
    if key in {"ustreasuries", "treasury", "treasuries"}:
        return "data/fixed_income_cache_treasury.parquet", "data/fixed_income_health_treasury.json"
    if key in {"bondetfs", "bondetf", "etf"}:
        return "data/fixed_income_cache_bond_etf.parquet", "data/fixed_income_health_bond_etf.json"
    raise ValueError(f"Unsupported fixed-income universe: {universe}")


def _ensure_stock_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[SCHEMA_COLUMNS]
    out["PEG_Ratio"] = (pd.to_numeric(out["PE_Ratio"], errors="coerce") / pd.to_numeric(out["Earnings_Growth_Pct"], errors="coerce")).where(
        pd.to_numeric(out["Earnings_Growth_Pct"], errors="coerce") > 0
    )
    out["Rule_of_40"] = pd.to_numeric(out["Revenue_Growth_YoY_Pct"], errors="coerce") + (
        pd.to_numeric(out["EBITDA_Margin"], errors="coerce") * 100.0
    )
    return out


def _update_stock_cache(cache_path: str, health_report_path: str, universe: str, include_metadata: bool) -> StockUpdateResult:
    tickers = fetch_universe_tickers(universe)
    refresh = refresh_fundamentals_yfinance(tickers, include_metadata=include_metadata)
    df = _ensure_stock_schema(refresh.data)

    reason = "cache updated"
    wrote_cache = False
    if df.empty:
        reason = "no rows returned from refresh"
    elif refresh.requested_count > 0 and (refresh.success_count / refresh.requested_count) < MIN_REFRESH_SUCCESS_RATIO:
        reason = f"success ratio {(refresh.success_count / refresh.requested_count):.1%} below threshold {MIN_REFRESH_SUCCESS_RATIO:.0%}"
    else:
        save_parquet_atomic(df, cache_path)
        wrote_cache = True

    report = {
        "run_timestamp": pd.Timestamp.utcnow().isoformat(),
        "universe": universe,
        "requested_count": int(refresh.requested_count),
        "success_count": int(refresh.success_count),
        "failure_count": int(refresh.failure_count),
        "rate_limited": bool(refresh.rate_limited),
        "wrote_cache": bool(wrote_cache),
        "reason": str(reason),
        "errors_sample": list(refresh.errors_sample or []),
    }
    write_json_report(report, health_report_path)

    return StockUpdateResult(
        data=df,
        requested_count=refresh.requested_count,
        success_count=refresh.success_count,
        failure_count=refresh.failure_count,
        rate_limited=refresh.rate_limited,
        errors_sample=refresh.errors_sample,
        wrote_cache=wrote_cache,
        reason=reason,
    )


def _show_stock_tab() -> None:
    st.markdown("## Stock Screener")
    status_text = ""
    telemetry_text = ""
    left, mid, right = st.columns([2.3, 1.0, 1.4], vertical_alignment="bottom")
    with left:
        universe = st.selectbox("Universe", STOCK_UNIVERSE_OPTIONS, index=0, key="stock_universe")
    with mid:
        include_metadata = st.checkbox("Enrich metadata (slower)", value=False, key="stock_enrich")
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        force_refresh = st.button("Refresh Stock Data", use_container_width=True, key="stock_refresh")

    cache_path, health_report_path = _stock_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    if stale_df is not None:
        stale_df = _ensure_stock_schema(stale_df)

    needs_refresh = force_refresh or stale_df is None or (not status.is_fresh) or (not status.schema_ok)
    if needs_refresh:
        with st.spinner("Refreshing stock data..."):
            try:
                update = _update_stock_cache(cache_path, health_report_path, universe, include_metadata)
                if update.wrote_cache:
                    df_raw = update.data
                elif stale_df is not None and not stale_df.empty:
                    df_raw = stale_df
                    st.warning(f"Refresh incomplete ({update.reason}). Using stale cache.")
                else:
                    df_raw = update.data
                    st.warning(f"Refresh incomplete ({update.reason}).")
                telemetry_text = f"Updated {update.success_count}/{update.requested_count} tickers"
            except Exception as e:
                if stale_df is not None and not stale_df.empty:
                    df_raw = stale_df
                    st.warning(f"Refresh failed ({e}). Using stale cache.")
                else:
                    st.error(f"Refresh failed and no usable cache is available: {e}")
                    return
    else:
        df_raw = stale_df
        telemetry_text = "Using fresh stock cache"

    if df_raw is None or df_raw.empty:
        st.warning("No stock data available.")
        return

    df = df_raw.copy()
    f1, f2, f3, f4 = st.columns([2.0, 2.0, 1.6, 1.6], vertical_alignment="bottom")
    with f1:
        query = st.text_input("Search ticker/company", value="", key="stock_query").strip().lower()
    with f2:
        sector_options = sorted([s for s in df["Sector"].dropna().unique().tolist()])
        sectors = st.multiselect("Sectors", options=sector_options, key="stock_sectors")
    with f3:
        min_rule40 = st.number_input(
            "Rule of 40 min",
            min_value=-50.0,
            max_value=80.0,
            value=-50.0,
            step=1.0,
            key="stock_rule40",
        )
    def _reset_stock_filters() -> None:
        defaults = {
            "stock_query": "",
            "stock_sectors": [],
            "stock_rule40": -50.0,
            "stock_ebitda_min": 0.0,
            "stock_roe_min": 0.0,
            "stock_rev_growth_min": -50.0,
            "stock_earnings_growth_min": -50.0,
            "stock_pe_min": 0.0,
            "stock_peg_min": 0.0,
            "stock_mcap_min_b": 0.0,
            "stock_mcap_max_b": 10000.0,
            "stock_require_complete": False,
        }
        for k, v in defaults.items():
            st.session_state[k] = v

    with f4:
        st.button("Reset Filters", use_container_width=True, key="stock_reset", on_click=_reset_stock_filters)

    with st.expander("Advanced Screening Filters", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            ebitda_min = st.number_input("EBITDA margin min", min_value=0.0, max_value=0.8, value=0.0, step=0.01, key="stock_ebitda_min")
        with s2:
            roe_min = st.number_input("ROE min", min_value=0.0, max_value=0.8, value=0.0, step=0.01, key="stock_roe_min")
        with s3:
            revenue_growth_min = st.number_input(
                "Revenue growth YoY min (Pct)",
                min_value=-50.0,
                max_value=200.0,
                value=-50.0,
                step=1.0,
                key="stock_rev_growth_min",
            )
        with s4:
            earnings_growth_min = st.number_input(
                "Earnings growth min (Pct)",
                min_value=-50.0,
                max_value=200.0,
                value=-50.0,
                step=1.0,
                key="stock_earnings_growth_min",
            )

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            pe_min = st.number_input("P/E ratio min", min_value=0.0, max_value=200.0, value=0.0, step=1.0, key="stock_pe_min")
        with v2:
            peg_min = st.number_input("PEG ratio min", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="stock_peg_min")
        with v3:
            mcap_min_b = st.number_input("Market cap min (B)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0, key="stock_mcap_min_b")
        with v4:
            mcap_max_b = st.number_input(
                "Market cap max (B)",
                min_value=0.0,
                max_value=10000.0,
                value=10000.0,
                step=10.0,
                key="stock_mcap_max_b",
            )

        require_complete = st.checkbox("Require complete data", value=False, key="stock_require_complete")

    if query:
        df = df[
            df["Ticker"].astype(str).str.lower().str.contains(query, na=False)
            | df["Company"].astype(str).str.lower().str.contains(query, na=False)
        ]
    if sectors:
        df = df[df["Sector"].isin(sectors)]
    allow_missing = not require_complete

    if ebitda_min > 0:
        mask = df["EBITDA_Margin"] >= ebitda_min
        df = df[mask | df["EBITDA_Margin"].isna()] if allow_missing else df[mask]
    if roe_min > 0:
        mask = df["ROE"] >= roe_min
        df = df[mask | df["ROE"].isna()] if allow_missing else df[mask]
    if revenue_growth_min > -50:
        mask = df["Revenue_Growth_YoY_Pct"] >= revenue_growth_min
        df = df[mask | df["Revenue_Growth_YoY_Pct"].isna()] if allow_missing else df[mask]
    if earnings_growth_min > -50:
        mask = df["Earnings_Growth_Pct"] >= earnings_growth_min
        df = df[mask | df["Earnings_Growth_Pct"].isna()] if allow_missing else df[mask]
    if min_rule40 > -50:
        mask = df["Rule_of_40"] >= min_rule40
        df = df[mask | df["Rule_of_40"].isna()] if allow_missing else df[mask]
    if pe_min > 0:
        mask = df["PE_Ratio"] >= pe_min
        df = df[mask | df["PE_Ratio"].isna()] if allow_missing else df[mask]
    if peg_min > 0:
        mask = df["PEG_Ratio"] >= peg_min
        df = df[mask | df["PEG_Ratio"].isna()] if allow_missing else df[mask]
    if mcap_min_b > 0:
        mask = df["MarketCap"] >= (mcap_min_b * 1e9)
        df = df[mask | df["MarketCap"].isna()] if allow_missing else df[mask]
    if mcap_max_b < 10000:
        mask = df["MarketCap"] <= (mcap_max_b * 1e9)
        df = df[mask | df["MarketCap"].isna()] if allow_missing else df[mask]

    if require_complete:
        df = df.dropna(
            subset=[
                "EBITDA_Margin",
                "ROE",
                "Revenue_Growth_YoY_Pct",
                "Earnings_Growth_Pct",
                "PE_Ratio",
                "PEG_Ratio",
                "Rule_of_40",
                "MarketCap",
            ]
        )
    df = df.sort_values(by="MarketCap", ascending=False)

    df_show = df.copy()
    df_show["MarketCap (B)"] = (pd.to_numeric(df_show["MarketCap"], errors="coerce") / 1e9).round(2)
    df_show["EBITDA Margin (Pct)"] = (pd.to_numeric(df_show["EBITDA_Margin"], errors="coerce") * 100).round(2)
    df_show["ROE (Pct)"] = (pd.to_numeric(df_show["ROE"], errors="coerce") * 100).round(2)
    df_show = df_show[
        [
            "Ticker",
            "Company",
            "Sector",
            "Close",
            "MarketCap (B)",
            "Revenue_Growth_YoY_Pct",
            "Earnings_Growth_Pct",
            "PE_Ratio",
            "PEG_Ratio",
            "Rule_of_40",
            "EBITDA Margin (Pct)",
            "ROE (Pct)",
        ]
    ]
    st.write(f"Matches: {len(df_show):,}")
    st.dataframe(df_show, use_container_width=True, height=500, hide_index=True)

    ticker = st.selectbox("Ticker details", options=[""] + df["Ticker"].astype(str).tolist(), index=0, key="stock_detail_ticker")
    if ticker:
        try:
            d = fetch_ticker_details(ticker)
            st.markdown(f"**{d.get('Company') or ticker} ({ticker})**")
            st.write(f"Sector: {d.get('Sector')}")
            st.write(f"Close: {d.get('Close')}")
            st.write(f"P/E: {d.get('PE_Ratio')}")
            st.write(f"PEG: {d.get('PEG_Ratio')}")
            st.write(f"Rule of 40: {d.get('Rule_of_40')}")
        except Exception as e:
            st.caption(f"Live detail unavailable: {e}")

    st.divider()
    if status_text:
        st.caption(status_text)
    if telemetry_text:
        st.caption(telemetry_text)


def _show_fixed_income_tab() -> None:
    st.markdown("## Bond & Treasury Screener")
    status_text = ""
    telemetry_text = ""
    left, right = st.columns([2.3, 1.4], vertical_alignment="bottom")
    with left:
        universe = st.selectbox("Universe", FI_UNIVERSE_OPTIONS, index=0, key="fi_universe")
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        force_refresh = st.button("Refresh Fixed-Income Data", use_container_width=True, key="fi_refresh")

    cache_path, health_report_path = _fi_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=FI_SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    needs_refresh = force_refresh or stale_df is None or (not status.is_fresh) or (not status.schema_ok)
    if needs_refresh:
        with st.spinner("Refreshing fixed-income data..."):
            result = run_fixed_income_pipeline(
                cache_path=cache_path,
                health_report_path=health_report_path,
                universe=universe,
                max_age_days=MAX_AGE_DAYS,
            )
            df_raw = result.data
            if (not result.wrote_cache) and stale_df is not None and not stale_df.empty:
                df_raw = stale_df
                st.warning(f"Refresh incomplete ({result.reason}). Using stale cache.")
            telemetry_text = result.reason
    else:
        df_raw = stale_df
        telemetry_text = "Using fresh fixed-income cache"

    if df_raw is None or df_raw.empty:
        st.warning("No fixed-income data available.")
        return

    df = df_raw.copy()
    for col in ["Price", "Yield_Pct", "Duration_Years", "Expense_Ratio_Pct", "AUM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        q = st.text_input("Search symbol/name", value="", key="fi_query").strip().lower()
    with c2:
        buckets = sorted([b for b in df["Maturity_Bucket"].dropna().astype(str).unique().tolist()])
        selected_buckets = st.multiselect("Maturity bucket", options=buckets, key="fi_buckets")
    with c3:
        min_yield = st.slider("Yield min (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1, key="fi_yield_min")
    with c4:
        max_duration = st.slider("Duration max (yrs)", min_value=0.0, max_value=30.0, value=30.0, step=0.1, key="fi_dur_max")

    if q:
        df = df[
            df["Symbol"].astype(str).str.lower().str.contains(q, na=False)
            | df["Name"].astype(str).str.lower().str.contains(q, na=False)
        ]
    if selected_buckets:
        df = df[df["Maturity_Bucket"].astype(str).isin(selected_buckets)]
    df = df[df["Yield_Pct"].fillna(-1) >= min_yield]
    df = df[df["Duration_Years"].fillna(999) <= max_duration]
    df = df.sort_values(by="Yield_Pct", ascending=False)

    show = df.copy()
    show["AUM (B)"] = (show["AUM"] / 1e9).round(2)
    show = show[
        [
            "Symbol",
            "Name",
            "Universe",
            "Type",
            "Price",
            "Yield_Pct",
            "Duration_Years",
            "Maturity_Bucket",
            "Expense_Ratio_Pct",
            "AUM (B)",
        ]
    ]
    st.write(f"Matches: {len(show):,}")
    st.dataframe(show, use_container_width=True, height=500, hide_index=True)

    detail_symbol = st.selectbox("Instrument details", options=[""] + df["Symbol"].astype(str).tolist(), index=0, key="fi_detail")
    if detail_symbol:
        row = df[df["Symbol"] == detail_symbol].head(1).iloc[0]
        st.markdown(f"**{row.get('Name')} ({detail_symbol})**")
        st.write(f"Yield (%): {row.get('Yield_Pct')}")
        st.write(f"Duration (yrs): {row.get('Duration_Years')}")
        st.write(f"Expense ratio (%): {row.get('Expense_Ratio_Pct')}")
        st.write(f"AUM: {row.get('AUM')}")
        if pd.notna(row.get("Duration_Years")):
            shock = -float(row["Duration_Years"])
            st.write(f"Estimated price impact for +100 bps move: {shock:.2f}%")

    st.divider()
    if status_text:
        st.caption(status_text)
    if telemetry_text:
        st.caption(telemetry_text)


stock_tab, fi_tab = st.tabs(["Stock Screener", "Bond & Treasury Screener"])
with stock_tab:
    _show_stock_tab()
with fi_tab:
    _show_fixed_income_tab()
