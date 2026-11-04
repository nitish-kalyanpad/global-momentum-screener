#!/usr/bin/env python3
"""
Independent verification of screener.py output using yfinance.

For a handful of tickers taken from results.csv, pull daily history from Yahoo
Finance (a completely separate data source) and re-check all five screen
conditions from scratch. Prints a side-by-side comparison of TradingView's
values (as captured in results.csv) vs. yfinance-derived values.

Small numeric differences are expected: the two providers may have a different
as-of session, and TradingView vs. Yahoo can differ on split/dividend
adjustment. The point is directional agreement on each pass/fail condition.
"""

from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf

# TradingView ticker  ->  Yahoo Finance symbol
TICKER_MAP = {
    "NASDAQ:CRNX": "CRNX",
    "NYSE:UTZ": "UTZ",
    "HKEX:1168": "1168.HK",
}

# thresholds mirrored from screener.py
MAX_PCT_BELOW_52W_HIGH = 0.10
MIN_1M_PERF_PCT = 10.0
MIN_VOLUME = 200_000


def yf_metrics(symbol: str) -> dict:
    # 14 months of daily bars -> enough for SMA200 and a 52-week window.
    # auto_adjust=False keeps raw Close, closest to TradingView's default price.
    df = yf.download(symbol, period="14mo", interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"no data for {symbol}")
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    last_close = float(close.iloc[-1])
    last_vol = float(volume.iloc[-1])
    high_52w = float(close.iloc[-252:].max())     # ~52 weeks of sessions
    sma20 = float(close.iloc[-20:].mean())
    sma200 = float(close.iloc[-200:].mean())

    # 1-month performance: close vs ~1 calendar month ago (nearest prior session)
    cutoff = close.index[-1] - pd.Timedelta(days=30)
    prior = close[close.index <= cutoff]
    perf_1m = (last_close / float(prior.iloc[-1]) - 1.0) * 100.0 if len(prior) else float("nan")

    return {
        "asof": close.index[-1].date().isoformat(),
        "close": last_close,
        "volume": last_vol,
        "high_52w": high_52w,
        "pct_below_52w_high": (high_52w / last_close - 1.0) * 100.0,
        "perf_1m": perf_1m,
        "sma20": sma20,
        "sma200": sma200,
    }


def check(m: dict) -> dict:
    pct = m["high_52w"] / m["close"] - 1.0
    return {
        "1_near_high": 0.0 <= pct <= MAX_PCT_BELOW_52W_HIGH,
        "2_perf_1m>10": m["perf_1m"] > MIN_1M_PERF_PCT,
        "3_vol>200k": m["volume"] > MIN_VOLUME,
        "4_close>sma20": m["close"] > m["sma20"],
        "5_close>sma200": m["close"] > m["sma200"],
    }


def main() -> int:
    tv = pd.read_csv("results.csv").set_index("ticker")

    for tv_ticker, yahoo in TICKER_MAP.items():
        if tv_ticker not in tv.index:
            print(f"\n### {tv_ticker}: NOT in results.csv (skipping)")
            continue
        row = tv.loc[tv_ticker]
        m = yf_metrics(yahoo)
        conds = check(m)

        print(f"\n{'='*70}\n{tv_ticker}   (yfinance: {yahoo}, as-of {m['asof']})\n{'='*70}")
        cmp = pd.DataFrame(
            {
                "TradingView": {
                    "close": row["close"],
                    "volume": row["volume"],
                    "Perf.1M %": row["Perf.1M"],
                    "% high above close": row["pct_below_52w_high"],
                    "SMA20": row["SMA20"],
                    "SMA200": row["SMA200"],
                },
                "yfinance": {
                    "close": m["close"],
                    "volume": m["volume"],
                    "Perf.1M %": m["perf_1m"],
                    "% high above close": m["pct_below_52w_high"],
                    "SMA20": m["sma20"],
                    "SMA200": m["sma200"],
                },
            }
        )
        with pd.option_context("display.float_format", lambda v: f"{v:,.4g}"):
            print(cmp.to_string())
        print("  conditions (yfinance):")
        for k, v in conds.items():
            print(f"    {'PASS' if v else 'FAIL'}  {k}")
        print(f"  ==> ALL FIVE PASS (yfinance): {all(conds.values())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
