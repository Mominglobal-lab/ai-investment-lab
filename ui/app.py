# app.py
# Bloomberg-style Stock Screener UI
# Uses local cache if < 7 days old, otherwise refreshes via data_fetcher.
# Zero API calls during filtering and sorting.

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# Ensure repo root is importable even when Streamlit starts from a nested cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.cache_manager import (
    get_cache_status,
    save_parquet_atomic,
    write_json_report,
    read_parquet_safe,
)
from data_pipeline.data_fetcher import (
    SCHEMA_COLUMNS,
    fetch_universe_tickers,
    fetch_ticker_details,
    refresh_fundamentals_yfinance,
)

UNIVERSE_OPTIONS = ["S&P 500", "Nasdaq 100"]
MAX_AGE_DAYS = 7
MIN_REFRESH_SUCCESS_RATIO = 0.25
FILTER_DEFAULTS = {
    "selected_sectors": [],
    "ebitda_min": 0.00,
    "roe_min": 0.00,
    "revenue_growth_min": -50.0,
    "earnings_growth_min": -50.0,
    "pe_min": 0.0,
    "peg_min": 0.0,
    "rule40_min": -50.0,
    "mcap_min_b": 0.0,
    "mcap_max_b": 10000.0,
    "require_complete": False,
    "sort_by": "MarketCap",
    "ascending": False,
    "query": "",
}


def _universe_key(universe: str) -> str:
    return (universe or "").strip().lower().replace("&", "and").replace(" ", "").replace("-", "")


def _paths_for_universe(universe: str) -> tuple[str, str]:
    key = _universe_key(universe)
    if key in {"sandp500", "sp500"}:
        return "data/fundamentals_cache_sp500.parquet", "data/fundamentals_health_report_sp500.json"
    if key in {"nasdaq100", "ndx"}:
        return "data/fundamentals_cache_nasdaq100.parquet", "data/fundamentals_health_report_nasdaq100.json"
    raise ValueError(f"Unsupported universe: {universe}")


@dataclass(frozen=True)
class UpdateCacheResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    rate_limited: bool
    errors_sample: list[str]
    wrote_cache: bool
    reason: str


st.set_page_config(page_title="Stock Screener", layout="wide")


@st.cache_data(show_spinner=False)
def load_cache(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=1800)
def load_ticker_details(ticker: str) -> dict:
    return fetch_ticker_details(ticker)


def _read_cache_safe(path: str) -> tuple[pd.DataFrame | None, str | None]:
    # Avoid streamlit cache masking schema problems; use a direct read here.
    return read_parquet_safe(path)


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[SCHEMA_COLUMNS]


def _ensure_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pe = pd.to_numeric(out["PE_Ratio"], errors="coerce")
    eg = pd.to_numeric(out["Earnings_Growth_Pct"], errors="coerce")
    out["PEG_Ratio"] = (pe / eg).where(eg > 0)

    out["Rule_of_40"] = (
        pd.to_numeric(out["Revenue_Growth_YoY_Pct"], errors="coerce")
        + (pd.to_numeric(out["EBITDA_Margin"], errors="coerce") * 100.0)
    )
    return out


def _is_write_eligible(df: pd.DataFrame, requested_count: int, success_count: int) -> tuple[bool, str]:
    missing_cols = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"missing required columns: {', '.join(missing_cols)}"
    if df.empty:
        return False, "no rows returned from refresh"
    if requested_count > 0:
        ratio = success_count / requested_count
        if ratio < MIN_REFRESH_SUCCESS_RATIO:
            return False, f"success ratio {ratio:.1%} below threshold {MIN_REFRESH_SUCCESS_RATIO:.0%}"
    return True, "cache updated"


def update_cache(path: str, include_metadata: bool, universe: str, health_report_path: str) -> UpdateCacheResult:
    tickers = fetch_universe_tickers(universe)
    refresh = refresh_fundamentals_yfinance(tickers, include_metadata=include_metadata)
    df = _ensure_derived_metrics(_ensure_schema(refresh.data))
    eligible, reason = _is_write_eligible(df, refresh.requested_count, refresh.success_count)
    wrote_cache = False

    if eligible:
        save_parquet_atomic(df, path)
        wrote_cache = True

    # Write a lightweight health report for diagnostics and repo credibility.
    try:
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
    except Exception:
        # Never allow reporting failures to break the UI.
        pass

    return UpdateCacheResult(
        data=df,
        requested_count=refresh.requested_count,
        success_count=refresh.success_count,
        failure_count=refresh.failure_count,
        rate_limited=refresh.rate_limited,
        errors_sample=refresh.errors_sample,
        wrote_cache=wrote_cache,
        reason=reason,
    )


def _fmt_num(v, digits: int = 2) -> str:
    return "N/A" if pd.isna(v) else f"{float(v):,.{digits}f}"


def _apply_min_filter(df: pd.DataFrame, column: str, value: float, allow_missing: bool) -> pd.DataFrame:
    mask = df[column] >= value
    if allow_missing:
        mask = mask | df[column].isna()
    return df[mask]


def _apply_max_filter(df: pd.DataFrame, column: str, value: float, allow_missing: bool) -> pd.DataFrame:
    mask = df[column] <= value
    if allow_missing:
        mask = mask | df[column].isna()
    return df[mask]


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close"] = out["Close"].round(2)
    out["MarketCap (B)"] = (out["MarketCap"] / 1e9).round(2)
    out["EBITDA Margin (Pct)"] = (out["EBITDA_Margin"] * 100).round(2)
    out["ROE (Pct)"] = (out["ROE"] * 100).round(2)
    out["Revenue Growth YoY (Pct)"] = out["Revenue_Growth_YoY_Pct"].round(2)
    out["Earnings Growth (Pct)"] = out["Earnings_Growth_Pct"].round(2)
    out["P/E"] = out["PE_Ratio"].round(2)
    out["PEG"] = out["PEG_Ratio"].round(2)
    out["Rule of 40"] = out["Rule_of_40"].round(2)

    core = [
        "Close",
        "MarketCap",
        "EBITDA_Margin",
        "ROE",
        "Revenue_Growth_YoY_Pct",
        "Earnings_Growth_Pct",
        "PE_Ratio",
        "PEG_Ratio",
        "Rule_of_40",
    ]
    out["Data Coverage"] = out[core].notna().sum(axis=1).astype(int).astype(str) + f"/{len(core)}"

    cols = [
        "Ticker",
        "Company",
        "Sector",
        "Close",
        "MarketCap (B)",
        "Revenue Growth YoY (Pct)",
        "Earnings Growth (Pct)",
        "P/E",
        "PEG",
        "Rule of 40",
        "EBITDA Margin (Pct)",
        "ROE (Pct)",
        "Data Coverage",
    ]
    return out[cols]


def safe_str(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


left, mid, right = st.columns([2, 2, 2], vertical_alignment="center")
with left:
    st.markdown("## Stock Screener")
with mid:
    universe = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0)
with right:
    cache_path, health_report_path = _paths_for_universe(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=SCHEMA_COLUMNS)
    cache_text = "No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days"
    fresh_text = "Fresh" if status.is_fresh else "Stale"
    schema_text = "Schema OK" if status.schema_ok else "Schema mismatch"
    st.write(f"{cache_text}  |  {fresh_text}  |  {schema_text}")
    enrich_metadata = st.checkbox("Enrich metadata (slower)", value=False)
    force_refresh = st.button("Refresh Data", use_container_width=True)

st.divider()

df_raw = None
telemetry = None
banner_warning = None

stale_cache_df, _stale_cache_error = _read_cache_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
if stale_cache_df is not None:
    stale_cache_df = _ensure_derived_metrics(_ensure_schema(stale_cache_df))

needs_refresh = force_refresh or (not status.exists) or (not status.is_fresh) or (not status.schema_ok) or stale_cache_df is None

if needs_refresh:
    with st.spinner("Refreshing data (API calls happening now)..."):
        try:
            update = update_cache(
                cache_path,
                include_metadata=enrich_metadata,
                universe=universe,
                health_report_path=health_report_path,
            )
            telemetry = (
                f"Updated {update.success_count}/{update.requested_count} tickers | "
                f"Rate-limited: {'yes' if update.rate_limited else 'no'}"
            )
            if update.wrote_cache:
                load_cache.clear()
                df_raw = update.data
            else:
                if stale_cache_df is not None and not stale_cache_df.empty:
                    df_raw = stale_cache_df
                    banner_warning = (
                        f"Refresh returned partial/invalid data ({update.reason}). "
                        "Using stale cache."
                    )
                else:
                    df_raw = update.data
                    banner_warning = f"Refresh incomplete ({update.reason}). No stale cache available."
        except Exception as e:
            if stale_cache_df is not None and not stale_cache_df.empty:
                df_raw = stale_cache_df
                banner_warning = f"Refresh failed ({e}). Using stale cache."
                telemetry = "Updated 0/0 tickers | Rate-limited: unknown | Using stale cache"
            else:
                st.error(f"Refresh failed and no usable cache is available: {e}")
                st.stop()
else:
    df_raw = stale_cache_df
    telemetry = f"Using fresh cache (no API refresh) | Universe: {universe}"

if telemetry:
    st.caption(telemetry)
if banner_warning:
    st.warning(banner_warning)

if df_raw is None or df_raw.empty:
    st.warning("No data available. Click Refresh Data to build the local cache.")
    st.stop()

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = None
for k, v in FILTER_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

filters_col, results_col, details_col = st.columns([1.1, 2.4, 1.2], gap="large")

with filters_col:
    st.markdown("### Filters")
    if st.button("Reset Filters", use_container_width=True):
        for k, v in FILTER_DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    sector_options = sorted([s for s in df_raw["Sector"].dropna().unique().tolist()])
    selected_sectors = st.multiselect("Sectors", options=sector_options, key="selected_sectors")

    st.markdown("#### Core Rules")
    ebitda_min = st.slider("EBITDA margin min", min_value=0.0, max_value=0.8, step=0.01, key="ebitda_min")
    roe_min = st.slider("ROE min", min_value=0.0, max_value=0.8, step=0.01, key="roe_min")
    revenue_growth_min = st.slider(
        "Revenue growth YoY min (Pct)", min_value=-50.0, max_value=200.0, step=1.0, key="revenue_growth_min"
    )

    st.markdown("#### Valuation + Growth")
    earnings_growth_min = st.slider(
        "Earnings growth min (Pct)", min_value=-50.0, max_value=200.0, step=1.0, key="earnings_growth_min"
    )
    pe_min = st.slider("P/E ratio min", min_value=0.0, max_value=200.0, step=1.0, key="pe_min")
    peg_min = st.slider("PEG ratio min", min_value=0.0, max_value=10.0, step=0.1, key="peg_min")
    rule40_min = st.slider("Rule of 40 min", min_value=-50.0, max_value=80.0, step=1.0, key="rule40_min")

    st.markdown("#### Size")
    mcap_min_b = st.slider("Market cap min (B)", min_value=0.0, max_value=10000.0, step=1.0, key="mcap_min_b")
    mcap_max_b = st.slider("Market cap max (B)", min_value=0.0, max_value=10000.0, step=10.0, key="mcap_max_b")

    st.markdown("#### Data Quality")
    require_complete = st.checkbox("Require complete data", key="require_complete")

    st.markdown("#### Ranking")
    sort_by = st.selectbox(
        "Sort by",
        [
            "MarketCap",
            "Close",
            "Revenue_Growth_YoY_Pct",
            "Earnings_Growth_Pct",
            "PE_Ratio",
            "PEG_Ratio",
            "Rule_of_40",
            "EBITDA_Margin",
            "ROE",
        ],
        key="sort_by",
    )
    ascending = st.checkbox("Ascending", key="ascending")

    st.markdown("#### Search")
    query = st.text_input("Ticker or company contains", key="query").strip()

df = df_raw.copy()
allow_missing = not require_complete

if selected_sectors:
    df = df[df["Sector"].isin(selected_sectors)]

if ebitda_min > 0:
    df = _apply_min_filter(df, "EBITDA_Margin", ebitda_min, allow_missing)
if roe_min > 0:
    df = _apply_min_filter(df, "ROE", roe_min, allow_missing)
if revenue_growth_min > -50:
    df = _apply_min_filter(df, "Revenue_Growth_YoY_Pct", revenue_growth_min, allow_missing)
if earnings_growth_min > -50:
    df = _apply_min_filter(df, "Earnings_Growth_Pct", earnings_growth_min, allow_missing)
if rule40_min > -50:
    df = _apply_min_filter(df, "Rule_of_40", rule40_min, allow_missing)
if mcap_min_b > 0:
    df = _apply_min_filter(df, "MarketCap", mcap_min_b * 1e9, allow_missing)
if mcap_max_b < 10000:
    df = _apply_max_filter(df, "MarketCap", mcap_max_b * 1e9, allow_missing)
if pe_min > 0:
    df = _apply_min_filter(df, "PE_Ratio", pe_min, allow_missing)
if peg_min > 0:
    df = _apply_min_filter(df, "PEG_Ratio", peg_min, allow_missing)

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

if query:
    q = query.lower()
    df = df[
        df["Ticker"].astype(str).str.lower().str.contains(q, na=False)
        | df["Company"].astype(str).str.lower().str.contains(q, na=False)
    ]

if sort_by in df.columns:
    df = df.sort_values(by=sort_by, ascending=ascending)
df_display = format_for_display(df)

with results_col:
    st.markdown("### Results")
    st.write(f"Matches: {len(df_display):,}")
    if df_display.empty:
        st.info("No matches with current filters. Widen growth/valuation sliders or disable 'Require complete data'.")

    tickers_in_view = df_display["Ticker"].astype(str).tolist()
    selected = st.selectbox(
        "Select a ticker to view details",
        options=[""] + tickers_in_view,
        index=0,
    )
    if selected:
        st.session_state["selected_ticker"] = selected

    if AGGRID_AVAILABLE:
        grid_df = df_display.reset_index(drop=True)
        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        gb.configure_column("Ticker", pinned="left")
        gb.configure_column("Company", pinned="left")
        gb.configure_grid_options(domLayout="normal")
        AgGrid(
            grid_df,
            gridOptions=gb.build(),
            height=560,
            fit_columns_on_grid_load=False,
            theme="streamlit",
        )
    else:
        st.dataframe(df_display, use_container_width=True, height=560, hide_index=True)
        st.caption("Tip: install streamlit-aggrid to pin Ticker/Company while scrolling.")

    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="stock_screener_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with details_col:
    st.markdown("### Details")
    lookup = st.text_input("Lookup ticker", value=safe_str(st.session_state.get("selected_ticker"))).strip().upper()
    if st.button("Get Ticker Detail", use_container_width=True) and lookup:
        st.session_state["selected_ticker"] = lookup

    t = st.session_state.get("selected_ticker")
    if not t:
        st.caption("Select a ticker or use lookup to open details.")
    else:
        row = df_raw[df_raw["Ticker"] == t].head(1)
        if row.empty:
            st.caption("Ticker not found in local cache.")
        else:
            r = row.iloc[0]
            st.markdown(f"**{safe_str(r.get('Company'))} ({t})**")
            st.write(f"Sector: {safe_str(r.get('Sector'))}")
            st.write(f"Close: {_fmt_num(r.get('Close'))}")
            st.write(f"Market cap (B): {_fmt_num((r.get('MarketCap') or 0) / 1e9) if pd.notna(r.get('MarketCap')) else 'N/A'}")
            st.write(f"Revenue growth YoY (Pct): {_fmt_num(r.get('Revenue_Growth_YoY_Pct'))}")
            st.write(f"Earnings growth (Pct): {_fmt_num(r.get('Earnings_Growth_Pct'))}")
            st.write(f"P/E ratio: {_fmt_num(r.get('PE_Ratio'))}")
            st.write(f"PEG ratio: {_fmt_num(r.get('PEG_Ratio'))}")
            st.write(f"Rule of 40: {_fmt_num(r.get('Rule_of_40'))}")
            st.write(
                f"EBITDA margin (Pct): {_fmt_num((r.get('EBITDA_Margin') or 0) * 100.0) if pd.notna(r.get('EBITDA_Margin')) else 'N/A'}"
            )
            st.write(f"ROE (Pct): {_fmt_num((r.get('ROE') or 0) * 100.0) if pd.notna(r.get('ROE')) else 'N/A'}")

        st.markdown("#### Live Detail")
        try:
            d = load_ticker_details(t)
            st.write(f"Company: {safe_str(d.get('Company'))}")
            st.write(f"Industry: {safe_str(d.get('Industry'))}")
            st.write(f"Website: {safe_str(d.get('Website'))}")
            st.write(f"Close: {_fmt_num(d.get('Close'))}")
            st.write(f"Earnings growth (Pct): {_fmt_num(d.get('Earnings_Growth_Pct'))}")
            st.write(f"P/E ratio: {_fmt_num(d.get('PE_Ratio'))}")
            st.write(f"PEG ratio: {_fmt_num(d.get('PEG_Ratio'))}")
            st.write(f"Rule of 40: {_fmt_num(d.get('Rule_of_40'))}")
            summary = safe_str(d.get("Summary"))
            if summary:
                st.caption(summary[:800] + ("..." if len(summary) > 800 else ""))
        except Exception as e:
            st.caption(f"Live detail unavailable: {e}")
