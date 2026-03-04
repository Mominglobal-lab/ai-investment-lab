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


def _empty_schema_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


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
            if include_metadata:
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
