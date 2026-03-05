# data_fetcher.py
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import logging
import random
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = [
    "Ticker",
    "Company",
    "Sector",
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

NUMERIC_COLUMNS = [
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

MAX_ERROR_SAMPLES = 20
CHUNK_SIZE = 30
MAX_RETRIES = 3
BASE_RETRY_SECONDS = 0.8
INTER_CHUNK_SECONDS = 0.3
COOLDOWN_AFTER_429_STREAK = 8
COOLDOWN_SECONDS = 8.0
PROGRESS_EVERY = 50


@dataclass(frozen=True)
class RefreshResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    rate_limited: bool
    errors_sample: list[str]


@dataclass(frozen=True)
class FixedIncomeRefreshResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    rate_limited: bool
    errors_sample: list[str]


@dataclass(frozen=True)
class PriceRefreshResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    errors_sample: list[str]


FI_SCHEMA_COLUMNS = [
    "Symbol",
    "Name",
    "Universe",
    "Type",
    "Price",
    "Yield_Pct",
    "Duration_Years",
    "Maturity_Bucket",
    "Expense_Ratio_Pct",
    "AUM",
]

FI_NUMERIC_COLUMNS = [
    "Price",
    "Yield_Pct",
    "Duration_Years",
    "Expense_Ratio_Pct",
    "AUM",
]

PRICE_SCHEMA_COLUMNS = [
    "Ticker",
    "Date",
    "AdjClose",
    "Close",
    "Volume",
]

PRICE_NUMERIC_COLUMNS = ["AdjClose", "Close", "Volume"]

TREASURY_ETF_INSTRUMENTS = [
    {"Symbol": "BIL", "Name": "SPDR Bloomberg 1-3 Month T-Bill ETF", "Duration_Years": 0.10, "Maturity_Bucket": "0-1Y"},
    {"Symbol": "SGOV", "Name": "iShares 0-3 Month Treasury Bond ETF", "Duration_Years": 0.12, "Maturity_Bucket": "0-1Y"},
    {"Symbol": "SHY", "Name": "iShares 1-3 Year Treasury Bond ETF", "Duration_Years": 1.90, "Maturity_Bucket": "1-3Y"},
    {"Symbol": "IEI", "Name": "iShares 3-7 Year Treasury Bond ETF", "Duration_Years": 4.40, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "IEF", "Name": "iShares 7-10 Year Treasury Bond ETF", "Duration_Years": 7.30, "Maturity_Bucket": "7-10Y"},
    {"Symbol": "TLT", "Name": "iShares 20+ Year Treasury Bond ETF", "Duration_Years": 16.40, "Maturity_Bucket": "20Y+"},
    {"Symbol": "GOVT", "Name": "iShares U.S. Treasury Bond ETF", "Duration_Years": 6.00, "Maturity_Bucket": "3-7Y"},
]

BOND_ETF_INSTRUMENTS = [
    {"Symbol": "AGG", "Name": "iShares Core U.S. Aggregate Bond ETF", "Duration_Years": 6.20, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "BND", "Name": "Vanguard Total Bond Market ETF", "Duration_Years": 6.10, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "LQD", "Name": "iShares iBoxx Investment Grade Corporate Bond ETF", "Duration_Years": 8.40, "Maturity_Bucket": "7-10Y"},
    {"Symbol": "HYG", "Name": "iShares iBoxx High Yield Corporate Bond ETF", "Duration_Years": 3.20, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "JNK", "Name": "SPDR Bloomberg High Yield Bond ETF", "Duration_Years": 3.40, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "VCIT", "Name": "Vanguard Intermediate-Term Corporate Bond ETF", "Duration_Years": 6.20, "Maturity_Bucket": "3-7Y"},
    {"Symbol": "BNDX", "Name": "Vanguard Total International Bond ETF", "Duration_Years": 7.00, "Maturity_Bucket": "7-10Y"},
]


def _empty_schema_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


def _empty_fixed_income_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FI_SCHEMA_COLUMNS)


def _empty_prices_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_SCHEMA_COLUMNS)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        _is_rate_limited_error(exc)
        or "timeout" in text
        or "timed out" in text
        or "temporar" in text
        or "connection aborted" in text
        or "connection reset" in text
        or "service unavailable" in text
        or "bad gateway" in text
    )


def _retry_delay(attempt: int) -> float:
    return BASE_RETRY_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.4)


def _maybe_parse_growth(stock) -> float | None:
    try:
        fin = stock.financials
        if fin is not None and "Total Revenue" in fin.index and fin.shape[1] >= 2:
            rev = fin.loc["Total Revenue"]
            base = rev.iloc[1]
            if pd.notna(base) and base != 0:
                return float((rev.iloc[0] - base) / base * 100.0)
    except Exception:
        return None
    return None


def _safe_rule_of_40(revenue_growth_pct: float | None, ebitda_margin: float | None) -> float | None:
    if revenue_growth_pct is None or ebitda_margin is None:
        return None
    return float(revenue_growth_pct + (ebitda_margin * 100.0))


def _safe_peg_ratio(pe_ratio: float | None, earnings_growth_pct: float | None) -> float | None:
    if pe_ratio is None or earnings_growth_pct is None:
        return None
    if pd.isna(pe_ratio) or pd.isna(earnings_growth_pct) or earnings_growth_pct <= 0:
        return None
    return float(pe_ratio / earnings_growth_pct)


def _fetch_fast_stage(batch: list[str], tickers_obj, include_metadata: bool) -> tuple[dict[str, dict], list[str], list[str], bool]:
    rows_map: dict[str, dict] = {}
    fallback_needed: list[str] = []
    errors: list[str] = []
    saw_rate_limit = False

    for t in batch:
        try:
            stock = tickers_obj.tickers.get(t)
            if stock is None:
                fallback_needed.append(t)
                continue
            fi = stock.fast_info
            mcap = fi.get("market_cap") if fi is not None else None
            close = fi.get("last_price") if fi is not None else None
            rows_map[t] = {
                "Ticker": t,
                "Company": None,
                "Sector": None,
                "Close": close,
                "MarketCap": mcap,
                "EBITDA_Margin": None,
                "ROE": None,
                "Revenue_Growth_YoY_Pct": None,
                "Earnings_Growth_Pct": None,
                "PE_Ratio": None,
                "PEG_Ratio": None,
                "Rule_of_40": None,
            }
            # Fall back to full fetch when metadata is requested OR
            # when fast fields are missing for this symbol.
            if include_metadata or close is None or mcap is None:
                fallback_needed.append(t)
        except Exception as e:
            fallback_needed.append(t)
            if _is_rate_limited_error(e):
                saw_rate_limit = True
            if len(errors) < MAX_ERROR_SAMPLES:
                errors.append(f"{t}: {e}")

    return rows_map, fallback_needed, errors, saw_rate_limit


def _enrich_symbol(stock, ticker: str) -> dict:
    info = stock.get_info() or {}
    revenue_growth = _maybe_parse_growth(stock)
    close = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    earnings_growth_raw = info.get("earningsGrowth")
    earnings_growth_pct = (earnings_growth_raw * 100.0) if earnings_growth_raw is not None else None
    ebitda_margin = info.get("ebitdaMargins")
    pe_ratio = info.get("trailingPE") if info.get("trailingPE") is not None else info.get("forwardPE")
    peg_ratio = _safe_peg_ratio(pe_ratio=pe_ratio, earnings_growth_pct=earnings_growth_pct)

    return {
        "Ticker": ticker,
        "Company": info.get("longName"),
        "Sector": info.get("sector"),
        "Close": close,
        "MarketCap": info.get("marketCap"),
        "EBITDA_Margin": ebitda_margin,
        "ROE": info.get("returnOnEquity"),
        "Revenue_Growth_YoY_Pct": revenue_growth,
        "Earnings_Growth_Pct": earnings_growth_pct,
        "PE_Ratio": pe_ratio,
        "PEG_Ratio": peg_ratio,
        "Rule_of_40": _safe_rule_of_40(revenue_growth, ebitda_margin),
    }


def fetch_sp500_tickers() -> list[str]:
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    fallback_csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(wiki_url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            raise ValueError("No tables found on Wikipedia page")
        df = tables[0]
        if "Symbol" not in df.columns:
            raise ValueError("Wikipedia table missing Symbol column")
        return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.warning("Primary S&P source failed: %s", e)

    try:
        df = pd.read_csv(fallback_csv_url)
        if "Symbol" not in df.columns:
            raise ValueError("Fallback CSV missing Symbol column")
        return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.error("Fallback S&P source failed: %s", e)
        raise RuntimeError("Unable to fetch S&P 500 ticker list from all sources")


def fetch_nasdaq100_tickers() -> list[str]:
    wiki_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    fallback_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    def _extract_symbols(tables: list[pd.DataFrame]) -> list[str]:
        for df in tables:
            cols = {str(c).strip().lower(): c for c in df.columns}
            symbol_col = None
            for candidate in ("ticker", "symbol"):
                if candidate in cols:
                    symbol_col = cols[candidate]
                    break
            if symbol_col is None:
                continue
            symbols = (
                df[symbol_col]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .tolist()
            )
            symbols = [s for s in symbols if s and s.lower() != "nan"]
            if symbols:
                return symbols
        return []

    for url in (wiki_url, fallback_url):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            symbols = _extract_symbols(tables)
            if symbols:
                return symbols
        except Exception as e:
            logger.warning("Nasdaq-100 source failed (%s): %s", url, e)

    raise RuntimeError("Unable to fetch Nasdaq-100 ticker list from available sources")


def fetch_universe_tickers(universe: str) -> list[str]:
    u = (universe or "").strip().lower()
    if u in {"s&p 500", "sp500", "s and p 500"}:
        return fetch_sp500_tickers()
    if u in {"nasdaq 100", "nasdaq-100", "ndx"}:
        return fetch_nasdaq100_tickers()
    raise ValueError(f"Unsupported universe: {universe}")


def fetch_fixed_income_universe_instruments(universe: str) -> list[dict[str, object]]:
    u = (universe or "").strip().lower()
    if u in {"us treasuries", "treasury", "treasuries"}:
        return [{**dict(x), "Universe": "US Treasuries", "Type": "Treasury ETF"} for x in TREASURY_ETF_INSTRUMENTS]
    if u in {"bond etfs", "bond etf", "etf"}:
        return [{**dict(x), "Universe": "Bond ETFs", "Type": "Bond ETF"} for x in BOND_ETF_INSTRUMENTS]
    raise ValueError(f"Unsupported fixed-income universe: {universe}")


def _safe_pct(v: object) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if pd.isna(fv):
        return None
    if 0.0 <= fv <= 1.0:
        return fv * 100.0
    return fv


def refresh_fixed_income_yfinance(instruments: list[dict[str, object]]) -> FixedIncomeRefreshResult:
    import yfinance as yf

    requested = len(instruments)
    if requested == 0:
        return FixedIncomeRefreshResult(
            data=_empty_fixed_income_df(),
            requested_count=0,
            success_count=0,
            failure_count=0,
            rate_limited=False,
            errors_sample=[],
        )

    rows: list[dict[str, object]] = []
    errors_sample: list[str] = []
    saw_rate_limit = False

    for inst in instruments:
        symbol = str(inst.get("Symbol") or "").strip().upper()
        if not symbol:
            continue

        try:
            stock = yf.Ticker(symbol)
            fi = None
            info = None
            try:
                fi = stock.fast_info
            except Exception:
                fi = None
            try:
                info = stock.get_info() or {}
            except Exception:
                info = {}

            price = None
            if fi is not None:
                price = fi.get("last_price")
            if price is None:
                price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            if price is None:
                try:
                    hist = stock.history(period="5d")
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        price = float(hist["Close"].iloc[-1])
                except Exception:
                    price = None

            yld = _safe_pct(
                info.get("yield")
                or info.get("trailingAnnualDividendYield")
                or info.get("distributionYield")
                or info.get("secYield")
            )
            expense = _safe_pct(info.get("annualReportExpenseRatio") or info.get("expenseRatio"))
            duration = info.get("effectiveDuration") or inst.get("Duration_Years")
            aum = info.get("totalAssets") or info.get("fundAssets")

            rows.append(
                {
                    "Symbol": symbol,
                    "Name": info.get("longName") or inst.get("Name"),
                    "Universe": str(inst.get("Universe") or ""),
                    "Type": str(inst.get("Type") or "ETF"),
                    "Price": price,
                    "Yield_Pct": yld,
                    "Duration_Years": duration,
                    "Maturity_Bucket": inst.get("Maturity_Bucket"),
                    "Expense_Ratio_Pct": expense,
                    "AUM": aum,
                }
            )
        except Exception as e:
            if _is_rate_limited_error(e):
                saw_rate_limit = True
            if len(errors_sample) < MAX_ERROR_SAMPLES:
                errors_sample.append(f"{symbol}: {e}")
            rows.append(
                {
                    "Symbol": symbol,
                    "Name": inst.get("Name"),
                    "Universe": str(inst.get("Universe") or ""),
                    "Type": str(inst.get("Type") or "ETF"),
                    "Price": None,
                    "Yield_Pct": None,
                    "Duration_Years": inst.get("Duration_Years"),
                    "Maturity_Bucket": inst.get("Maturity_Bucket"),
                    "Expense_Ratio_Pct": None,
                    "AUM": None,
                }
            )

    data = pd.DataFrame(rows, columns=FI_SCHEMA_COLUMNS)
    for col in FI_NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    success_count = 0
    if not data.empty:
        success_mask = data[["Price", "Yield_Pct", "Duration_Years"]].notna().any(axis=1)
        success_count = int(success_mask.sum())
    failure_count = max(requested - success_count, 0)

    return FixedIncomeRefreshResult(
        data=data,
        requested_count=requested,
        success_count=success_count,
        failure_count=failure_count,
        rate_limited=saw_rate_limit,
        errors_sample=errors_sample,
    )


def refresh_fundamentals_yfinance(tickers: list[str], include_metadata: bool = False) -> RefreshResult:
    import yfinance as yf

    clean_tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    requested = len(clean_tickers)
    if requested == 0:
        return RefreshResult(
            data=_empty_schema_df(),
            requested_count=0,
            success_count=0,
            failure_count=0,
            rate_limited=False,
            errors_sample=[],
        )

    rows_map: dict[str, dict] = {}
    errors_sample: list[str] = []
    full_success = 0
    partial_success = 0
    consecutive_429 = 0
    saw_rate_limit = False
    processed = 0

    for batch in _chunks(clean_tickers, CHUNK_SIZE):
        tickers_obj = yf.Tickers(" ".join(batch))
        fast_rows, fallback_symbols, fast_errors, fast_rate_limited = _fetch_fast_stage(batch, tickers_obj, include_metadata=include_metadata)
        rows_map.update(fast_rows)
        for err in fast_errors:
            if len(errors_sample) < MAX_ERROR_SAMPLES:
                errors_sample.append(err)
        if fast_rate_limited:
            saw_rate_limit = True

        for t in fallback_symbols:
            stock = tickers_obj.tickers.get(t) or yf.Ticker(t)
            enriched = None
            last_exc: Exception | None = None

            for attempt in range(MAX_RETRIES):
                try:
                    enriched = _enrich_symbol(stock, t)
                    break
                except Exception as e:
                    last_exc = e
                    if _is_rate_limited_error(e):
                        saw_rate_limit = True
                        consecutive_429 += 1
                    else:
                        consecutive_429 = 0

                    should_retry = attempt < (MAX_RETRIES - 1) and _is_transient_error(e)
                    if should_retry:
                        time.sleep(_retry_delay(attempt))
                        continue
                    break

            if enriched is not None:
                rows_map[t] = enriched
                full_success += 1
                consecutive_429 = 0
            else:
                if t in rows_map and pd.notna(rows_map[t].get("MarketCap")):
                    partial_success += 1
                if last_exc is not None and len(errors_sample) < MAX_ERROR_SAMPLES:
                    errors_sample.append(f"{t}: {last_exc}")

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                logger.info(
                    "Refresh progress: %d/%d processed (full=%d, partial=%d)",
                    processed,
                    requested,
                    full_success,
                    partial_success,
                )

            if consecutive_429 >= COOLDOWN_AFTER_429_STREAK:
                logger.info("Rate-limit cooldown triggered for %.1f seconds", COOLDOWN_SECONDS)
                time.sleep(COOLDOWN_SECONDS)
                consecutive_429 = 0

        time.sleep(INTER_CHUNK_SECONDS)

    data = pd.DataFrame(list(rows_map.values()), columns=SCHEMA_COLUMNS)
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # Recompute derived metrics to keep them consistent even on partial rows.
    data["PEG_Ratio"] = data.apply(
        lambda r: _safe_peg_ratio(
            pe_ratio=r.get("PE_Ratio"),
            earnings_growth_pct=r.get("Earnings_Growth_Pct"),
        ),
        axis=1,
    )
    data["Rule_of_40"] = data.apply(
        lambda r: _safe_rule_of_40(
            revenue_growth_pct=r.get("Revenue_Growth_YoY_Pct"),
            ebitda_margin=r.get("EBITDA_Margin"),
        ),
        axis=1,
    )

    success_count = 0
    if not data.empty:
        success_mask = data[NUMERIC_COLUMNS].notna().any(axis=1)
        success_count = int(success_mask.sum())
    failure_count = max(requested - success_count, 0)

    logger.info(
        "Refresh completed: requested=%d success=%d failure=%d rate_limited=%s",
        requested,
        success_count,
        failure_count,
        saw_rate_limit,
    )
    if errors_sample:
        logger.info("Refresh error sample (%d): %s", len(errors_sample), " | ".join(errors_sample[:5]))

    return RefreshResult(
        data=data,
        requested_count=requested,
        success_count=success_count,
        failure_count=failure_count,
        rate_limited=saw_rate_limit,
        errors_sample=errors_sample,
    )


def fetch_ticker_details(ticker: str) -> dict:
    import yfinance as yf

    t = (ticker or "").strip().upper()
    if not t:
        raise ValueError("Ticker is required")

    stock = yf.Ticker(t)
    info = stock.get_info() or {}

    close = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if close is None:
        try:
            hist = stock.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = float(hist["Close"].iloc[-1])
        except Exception:
            close = None

    earnings_growth_raw = info.get("earningsGrowth")
    earnings_growth_pct = (earnings_growth_raw * 100.0) if earnings_growth_raw is not None else None
    ebitda_margin = info.get("ebitdaMargins")
    revenue_growth = _maybe_parse_growth(stock)
    rule_of_40 = _safe_rule_of_40(revenue_growth, ebitda_margin)
    pe_ratio = info.get("trailingPE") if info.get("trailingPE") is not None else info.get("forwardPE")
    peg_ratio = _safe_peg_ratio(pe_ratio, earnings_growth_pct)

    return {
        "Ticker": t,
        "Company": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Website": info.get("website"),
        "Summary": info.get("longBusinessSummary"),
        "Close": close,
        "MarketCap": info.get("marketCap"),
        "PE_Ratio": pe_ratio,
        "PEG_Ratio": peg_ratio,
        "Earnings_Growth_Pct": earnings_growth_pct,
        "Revenue_Growth_YoY_Pct": revenue_growth,
        "EBITDA_Margin_Pct": (ebitda_margin * 100.0) if ebitda_margin is not None else None,
        "ROE_Pct": (info.get("returnOnEquity") * 100.0) if info.get("returnOnEquity") is not None else None,
        "Rule_of_40": rule_of_40,
    }


def refresh_prices_yfinance(tickers: list[str], lookback_years: int = 5) -> PriceRefreshResult:
    import yfinance as yf

    clean_tickers = sorted(set([str(t).strip().upper() for t in tickers if str(t).strip()]))
    requested = len(clean_tickers)
    if requested == 0:
        return PriceRefreshResult(
            data=_empty_prices_df(),
            requested_count=0,
            success_count=0,
            failure_count=0,
            errors_sample=[],
        )

    period_years = max(int(lookback_years), 5)
    rows: list[dict[str, object]] = []
    errors_sample: list[str] = []
    success_count = 0

    for ticker in clean_tickers:
        try:
            hist = yf.Ticker(ticker).history(period=f"{period_years}y", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                raise ValueError("empty history")
            frame = hist.reset_index()
            date_col = "Date" if "Date" in frame.columns else frame.columns[0]
            frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.tz_localize(None)
            frame["AdjClose"] = pd.to_numeric(frame.get("Adj Close"), errors="coerce")
            frame["Close"] = pd.to_numeric(frame.get("Close"), errors="coerce")
            frame["Volume"] = pd.to_numeric(frame.get("Volume"), errors="coerce")
            frame = frame.dropna(subset=["Date", "AdjClose"])
            if frame.empty:
                raise ValueError("no valid AdjClose rows")
            rows.extend(
                {
                    "Ticker": ticker,
                    "Date": r["Date"],
                    "AdjClose": r["AdjClose"],
                    "Close": r["Close"],
                    "Volume": r["Volume"],
                }
                for _, r in frame.iterrows()
            )
            success_count += 1
        except Exception as e:
            if len(errors_sample) < MAX_ERROR_SAMPLES:
                errors_sample.append(f"{ticker}: {e}")

    out = pd.DataFrame(rows, columns=PRICE_SCHEMA_COLUMNS)
    if not out.empty:
        out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        for col in PRICE_NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    failure_count = max(requested - success_count, 0)
    return PriceRefreshResult(
        data=out,
        requested_count=requested,
        success_count=success_count,
        failure_count=failure_count,
        errors_sample=errors_sample,
    )
