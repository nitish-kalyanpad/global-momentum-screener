#!/usr/bin/env python3
"""
Reproduce a TradingView stock screen over the *global* equity universe using the
free, public scanner endpoint (via the `tradingview-screener` package). No paid
TradingView plan required.

Filters (a row must pass ALL five):
  1. Price within 10% of its 52-week high: (price_52_week_high / close) - 1
     is between MIN_PCT_BELOW_52W_HIGH and MAX_PCT_BELOW_52W_HIGH.
  2. 1-month performance  > MIN_1M_PERF_PCT   (Perf.1M is in PERCENT units)
  3. Volume               > MIN_VOLUME
  4. close > SMA20
  5. close > SMA200

Why some filters run server-side and two run client-side
--------------------------------------------------------
TradingView exposes a per-market endpoint (e.g. /america/scan) and a combined
/global/scan endpoint. Scanning *all* markets at once requires the global
endpoint. Empirically, the global endpoint rejects any column-to-column
comparison involving the SMA fields with:

    400 Bad Request {"error":"Incompatible types: number and null"}

(the same comparison works fine on a single-market endpoint). So the three
price/perf/volume predicates run server-side, and the two SMA comparisons are
applied client-side on TradingView's *own* SMA20/SMA200 values — numerically
identical to a server-side filter, just evaluated locally. See VERIFY.md notes.

Tune any threshold via the named constants below.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tradingview_screener import Query, col

# --------------------------------------------------------------------------- #
# Tunable thresholds  (edit these)                                            #
# --------------------------------------------------------------------------- #
# Filter 1: how far below the 52-week high the close may sit.
# Bound is on (price_52_week_high / close) - 1, i.e. 0.10 == "within 10% of high".
MIN_PCT_BELOW_52W_HIGH = 0.00
MAX_PCT_BELOW_52W_HIGH = 0.10

MIN_1M_PERF_PCT = 10.0      # Filter 2: Perf.1M threshold, in PERCENT (10 == +10%)
MIN_VOLUME = 200_000        # Filter 3: minimum share volume

REQUIRE_ABOVE_SMA20 = True   # Filter 4: close > SMA20
REQUIRE_ABOVE_SMA200 = True  # Filter 5: close > SMA200

# --------------------------------------------------------------------------- #

# Fields pulled from the scanner (names verified against the installed package
# and confirmed to return real values from the live endpoint).
FIELDS = [
    "name",
    "close",
    "volume",
    "price_52_week_high",
    "Perf.1M",
    "SMA20",
    "SMA200",
    "market_cap_basic",
    "sector",
    "exchange",
    # display-only fields for the HTML view
    "description",
    "industry",
    "country",
    "currency",
]

VIEW_TEMPLATE = Path(__file__).with_name("screener_view.html")

PAGE_SIZE = 500  # rows per request; the endpoint is paginated via range [offset, offset+n]


def build_query() -> Query:
    """Server-side query: global universe + the three globally-safe predicates."""
    return (
        Query()
        .set_markets()  # empty -> /global/scan, the full multi-market equity universe
        .select(*FIELDS)
        .where(
            # (price_52_week_high / close) between (1+min) and (1+max)
            col("price_52_week_high").between_pct(
                "close", 1.0 + MIN_PCT_BELOW_52W_HIGH, 1.0 + MAX_PCT_BELOW_52W_HIGH
            ),
            col("Perf.1M") > MIN_1M_PERF_PCT,
            col("volume") > MIN_VOLUME,
        )
        .order_by("Perf.1M", ascending=False)
    )


def fetch_all() -> tuple[int, pd.DataFrame]:
    """Page through the entire server-side result set."""
    offset = 0
    frames: list[pd.DataFrame] = []
    total = 0
    while True:
        q = build_query().offset(offset).limit(offset + PAGE_SIZE)
        total, df = q.get_scanner_data()
        if df.empty:
            break
        frames.append(df)
        offset += len(df)
        if offset >= total:
            break
    if frames:
        return total, pd.concat(frames, ignore_index=True)
    return total, pd.DataFrame(columns=["ticker", *FIELDS])


def apply_client_side_filters(df: pd.DataFrame, min_market_cap: float | None) -> pd.DataFrame:
    df = df.copy()

    # Filters 4 & 5: close above the moving averages (SMA values come from TV).
    # Rows with a null SMA cannot satisfy "close > SMA", so they are excluded.
    if REQUIRE_ABOVE_SMA20:
        df = df[df["close"] > df["SMA20"]]
    if REQUIRE_ABOVE_SMA200:
        df = df[df["close"] > df["SMA200"]]

    # Optional microcap cut. market_cap_basic is in each instrument's fundamental
    # currency (mixed across markets), so this is an approximate, per-symbol cut.
    # A null market cap cannot be shown to clear the bar, so it is dropped.
    if min_market_cap is not None:
        df = df[df["market_cap_basic"].notna() & (df["market_cap_basic"] >= min_market_cap)]

    return df


def to_output(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "exchange": df["exchange"],
            "name": df["name"],
            "close": df["close"],
            "volume": df["volume"],
            "market_cap": df["market_cap_basic"],
            "sector": df["sector"],
            "Perf.1M": df["Perf.1M"],
            # % the 52-week high sits above the close, matching filter 1's metric:
            # (price_52_week_high / close - 1) * 100
            "pct_below_52w_high": (df["price_52_week_high"] / df["close"] - 1.0) * 100.0,
            "SMA20": df["SMA20"],
            "SMA200": df["SMA200"],
        }
    )
    return out.sort_values("Perf.1M", ascending=False).reset_index(drop=True)


def render_html(df: pd.DataFrame, min_market_cap: float | None) -> str:
    """Self-contained dark/light table view of the screen (50 rows per page, sortable, filterable)."""
    view = pd.DataFrame(
        {
            "t": df["ticker"],
            "ex": df["exchange"],
            "co": df["description"].fillna(df["name"]),
            "se": df["sector"],
            "in": df["industry"],
            "ct": df["country"],
            "mc": df["market_cap_basic"],
            "px": df["close"],
            "cur": df["currency"],
            "pf": df["Perf.1M"],
            "hi": (df["price_52_week_high"] / df["close"] - 1.0) * 100.0,
            "s20": (df["close"] / df["SMA20"] - 1.0) * 100.0,
            "s200": (df["close"] / df["SMA200"] - 1.0) * 100.0,
            "vol": df["volume"],
        }
    ).sort_values("pf", ascending=False)
    meta = {
        "max_below_high": MAX_PCT_BELOW_52W_HIGH * 100,
        "min_perf": MIN_1M_PERF_PCT,
        "min_volume": MIN_VOLUME,
        "min_market_cap": min_market_cap if min_market_cap is not None else 1e9,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    payload = '{"meta": %s, "rows": %s}' % (json.dumps(meta), view.to_json(orient="records"))
    payload = payload.replace("</", "<\\/")  # keep the JSON from closing the <script> tag
    return VIEW_TEMPLATE.read_text().replace("__DATA__", payload)


def wrap_page(fragment: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
        + fragment
        + "\n</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        metavar="CAP",
        help="Drop rows whose market_cap_basic is below CAP (and rows with no market cap). "
        "Value is in USD. Example: --min-market-cap 1e9",
    )
    parser.add_argument("--out", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument(
        "--html",
        default="results.html",
        metavar="PATH",
        help="Also write an interactive table view (default: results.html). The page opens with a "
        "market-cap filter (default $1B, or --min-market-cap) that you can change in the page; "
        "the CSV is never affected by it.",
    )
    args = parser.parse_args()

    print("Querying TradingView global scanner (server-side: 52w-high, Perf.1M, volume)...", file=sys.stderr)
    total, raw = fetch_all()
    print(f"  server-side matches: {total}  (fetched {len(raw)} rows)", file=sys.stderr)

    filtered = apply_client_side_filters(raw, args.min_market_cap)
    out = to_output(filtered)

    out.to_csv(args.out, index=False)
    if args.html:
        # the HTML view always gets the full screen result so its market-cap filter can widen again
        full = apply_client_side_filters(raw, None)
        Path(args.html).write_text(wrap_page(render_html(full, args.min_market_cap)))
        print(f"wrote {args.html}", file=sys.stderr)

    print(f"\n{len(out)} stocks passed all five filters"
          + (f" (min market cap {args.min_market_cap:g})" if args.min_market_cap else "")
          + f" -> wrote {args.out}\n")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(out.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
