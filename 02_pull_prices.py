#!/usr/bin/env python3
"""
02_pull_prices.py  —  Daily price collection for the value-chain universe.

Stage 1 of the data-collection plan: pull daily-close prices (2019-06-01 .. 2026-04-30)
for ~250-350 firms, yfinance primary + Stooq fallback, with split-sanity checks and a
QC report against the >=95%-clean threshold.

WHY 2019-06-01 and not 2020-01-01:
    The affected-company labelling uses a 60-day ROLLING volatility baseline. To have a
    valid baseline for events in early 2020, prices must start ~6 months before the
    2020-01-01 analysis window begins.

NETWORK:
    Must run where Yahoo Finance / Stooq are reachable (i.e. your own machine), NOT inside
    a sandbox with a locked egress allowlist. CPU-only, no GPU. ~250-350 tickers with polite
    pauses ~ a few minutes wall-clock; everything is cached to Parquet so re-runs are cheap.

USAGE:
    pip install yfinance pandas pyarrow requests
    python 02_pull_prices.py                         # uses built-in seed universe
    python 02_pull_prices.py --universe config/universe.csv
    python 02_pull_prices.py --start 2019-06-01 --end 2026-04-30 --out data/raw/prices
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_START = "2019-06-01"   # ~6mo of lookback before the 2020-01-01 analysis window
DEFAULT_END = "2026-04-30"     # labelling cutoff (1-2wk post-event returns must be observable)
SPLIT_JUMP_THRESHOLD = 0.35    # |close-to-close| move above this with no nearby split = suspect
POLITE_PAUSE_SECS = 0.8        # be nice to Yahoo; avoids IP throttling

# Stooq exchange suffixes for non-US listings (extend as your universe grows).
STOOQ_SUFFIX_BY_COUNTRY = {
    "US": ".us", "DE": ".de", "NL": ".nl", "JP": ".jp",
    "UK": ".uk", "FR": ".fr", "TW": ".tw",
}

# Minimal self-contained seed so the script runs with zero config. Replace/extend via
# --universe config/universe.csv (columns: ticker,name,country,has_price[,stooq_symbol]).
SEED_UNIVERSE: list[dict[str, object]] = [
    {"ticker": "NVDA", "name": "NVIDIA", "country": "US", "has_price": True},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "country": "US", "has_price": True},
    {"ticker": "INTC", "name": "Intel", "country": "US", "has_price": True},
    {"ticker": "AVGO", "name": "Broadcom", "country": "US", "has_price": True},
    {"ticker": "QCOM", "name": "Qualcomm", "country": "US", "has_price": True},
    {"ticker": "TSM", "name": "TSMC (ADR)", "country": "US", "has_price": True},
    {"ticker": "ASML", "name": "ASML (ADR)", "country": "US", "has_price": True},
    {"ticker": "AMAT", "name": "Applied Materials", "country": "US", "has_price": True},
    {"ticker": "LRCX", "name": "Lam Research", "country": "US", "has_price": True},
    {"ticker": "KLAC", "name": "KLA", "country": "US", "has_price": True},
    {"ticker": "MU", "name": "Micron", "country": "US", "has_price": True},
    {"ticker": "ARM", "name": "Arm Holdings (ADR)", "country": "US", "has_price": True},
    {"ticker": "IFX.DE", "name": "Infineon", "country": "DE", "has_price": True},
    {"ticker": "NXPI", "name": "NXP Semiconductors", "country": "US", "has_price": True},
    {"ticker": "AAPL", "name": "Apple", "country": "US", "has_price": True},
    {"ticker": "F", "name": "Ford", "country": "US", "has_price": True},
    {"ticker": "GM", "name": "General Motors", "country": "US", "has_price": True},
    {"ticker": "VOW3.DE", "name": "Volkswagen", "country": "DE", "has_price": True},
    {"ticker": "BOSCH", "name": "Robert Bosch (private)", "country": "DE", "has_price": False},
]


@dataclass
class PullResult:
    """Outcome of attempting to pull one ticker."""
    ticker: str
    source: str          # "yfinance" | "stooq" | "skipped" | "failed"
    n_rows: int
    n_split_flags: int
    note: str = ""


# --------------------------------------------------------------------------------------
# Universe loading
# --------------------------------------------------------------------------------------

def load_universe(path: Path | None) -> pd.DataFrame:
    """Load the firm list from CSV, or fall back to the built-in seed list."""
    if path and path.exists():
        df = pd.read_csv(path)
        if "has_price" not in df.columns:
            df["has_price"] = True
        if "country" not in df.columns:
            df["country"] = "US"
        return df
    print(f"[info] no universe CSV at {path}; using built-in {len(SEED_UNIVERSE)}-firm seed.")
    return pd.DataFrame(SEED_UNIVERSE)


# --------------------------------------------------------------------------------------
# Primary source: yfinance
# --------------------------------------------------------------------------------------

def pull_yfinance(ticker: str, start: str, end: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Pull BOTH the adjusted series (auto_adjust=True, for returns/event-study) and the raw
    series (auto_adjust=False, retained to detect mis-applied splits). Returns (adjusted, raw),
    either of which may be None on failure.
    """
    try:
        adj = yf.download(ticker, start=start, end=end, auto_adjust=True,
                          actions=True, progress=False)
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                          actions=True, progress=False)
        if adj is None or adj.empty:
            return None, None
        return adj, raw
    except Exception as exc:  # noqa: BLE001 - log and let caller decide on fallback
        print(f"[warn] yfinance failed for {ticker}: {type(exc).__name__}: {exc}")
        return None, None


# --------------------------------------------------------------------------------------
# Fallback source: Stooq (per-ticker CSV; no API key)
# --------------------------------------------------------------------------------------

def stooq_symbol(ticker: str, country: str, override: str | None) -> str:
    """Map a ticker to its Stooq symbol (e.g. AAPL -> aapl.us, IFX.DE -> ifx.de)."""
    if override:
        return override.lower()
    base = ticker.split(".")[0].lower()
    suffix = STOOQ_SUFFIX_BY_COUNTRY.get(country.upper(), ".us")
    return f"{base}{suffix}"


def pull_stooq(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch a daily CSV from Stooq. Stooq has a low, undocumented daily quota -> cache hard."""
    d1, d2 = start.replace("-", ""), end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = resp.text.strip()
        if text.lower().startswith("exceeded") or "<html" in text.lower() or text == "":
            print(f"[warn] stooq returned no usable data for {symbol} (quota/CAPTCHA?).")
            return None
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"]).set_index("Date")
        return df if not df.empty else None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] stooq failed for {symbol}: {type(exc).__name__}: {exc}")
        return None


# --------------------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------------------

def split_sanity_check(raw: pd.DataFrame | None) -> int:
    """
    Flag close-to-close moves > SPLIT_JUMP_THRESHOLD on the RAW (unadjusted) close that do
    NOT coincide with a recorded stock split within +/- 2 trading days. These are candidate
    mis-applied-split errors (yfinance issue #1531) to spot-check against filings.
    Returns the number of flagged days.
    """
    if raw is None or raw.empty or "Close" not in raw.columns:
        return 0
    close = raw["Close"].astype("float64").squeeze()
    pct = close.pct_change().abs()
    big_moves = pct[pct > SPLIT_JUMP_THRESHOLD].index

    split_days: set = set()
    if "Stock Splits" in raw.columns:
        splits = raw["Stock Splits"].squeeze()
        split_days = set(splits[splits != 0].index)

    flags = 0
    for day in big_moves:
        pos = raw.index.get_loc(day)
        window = raw.index[max(0, pos - 2): min(len(raw), pos + 3)]
        if not (set(window) & split_days):
            flags += 1
    return flags


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, out_dir: Path, ticker: str, kind: str) -> None:
    """Persist immediately so a pull is never repeated. kind in {'adjusted','raw'}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace(".", "_")
    df.to_parquet(out_dir / f"{safe}__{kind}.parquet")


def already_cached(out_dir: Path, ticker: str) -> bool:
    safe = ticker.replace(".", "_")
    return (out_dir / f"{safe}__adjusted.parquet").exists()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Pull daily prices for the value-chain universe.")
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/prices"))
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    universe = load_universe(args.universe)
    results: list[PullResult] = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"]).strip()
        country = str(row.get("country", "US"))
        has_price = bool(row.get("has_price", True))
        override = row.get("stooq_symbol")
        override = str(override) if isinstance(override, str) and override else None

        # Private firms (e.g. Bosch) are KG nodes with no price series.
        if not has_price:
            results.append(PullResult(ticker, "skipped", 0, 0, "has_price=false (private)"))
            print(f"[skip] {ticker}: private, no price series.")
            continue

        if already_cached(args.out, ticker) and not args.force:
            results.append(PullResult(ticker, "yfinance", -1, 0, "cached"))
            print(f"[skip] {ticker}: already cached.")
            continue

        # --- primary: yfinance ---
        adj, raw = pull_yfinance(ticker, args.start, args.end)
        if adj is not None and not adj.empty:
            flags = split_sanity_check(raw)
            save_parquet(adj, args.out, ticker, "adjusted")
            if raw is not None and not raw.empty:
                save_parquet(raw, args.out, ticker, "raw")
            note = f"{flags} split-sanity flag(s) -> spot-check vs filings" if flags else ""
            results.append(PullResult(ticker, "yfinance", len(adj), flags, note))
            print(f"[ok]   {ticker}: {len(adj)} rows (yfinance){' | ' + note if note else ''}")
            time.sleep(POLITE_PAUSE_SECS)
            continue

        # --- fallback: Stooq ---
        sym = stooq_symbol(ticker, country, override)
        sdf = pull_stooq(sym, args.start, args.end)
        if sdf is not None and not sdf.empty:
            save_parquet(sdf, args.out, ticker, "adjusted")
            results.append(PullResult(ticker, "stooq", len(sdf), 0, f"via {sym}"))
            print(f"[ok]   {ticker}: {len(sdf)} rows (stooq:{sym})")
        else:
            results.append(PullResult(ticker, "failed", 0, 0, "both sources failed"))
            print(f"[FAIL] {ticker}: yfinance and stooq both failed -> manual check needed.")
        time.sleep(POLITE_PAUSE_SECS)

    # --- QC report ---
    report = pd.DataFrame([r.__dict__ for r in results])
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "_pull_report.csv", index=False)

    priced = report[report["source"] != "skipped"]
    n_priced = len(priced)
    n_clean = len(priced[priced["source"].isin(["yfinance", "stooq"])])
    coverage = (n_clean / n_priced * 100) if n_priced else 0.0
    n_flagged = int(report["n_split_flags"].clip(lower=0).sum())
    n_failed = int((report["source"] == "failed").sum())

    print("\n" + "=" * 60)
    print("QC SUMMARY")
    print("=" * 60)
    print(f"  firms needing a price series : {n_priced}")
    print(f"  clean series obtained        : {n_clean} ({coverage:.1f}%)")
    print(f"  split-sanity flags           : {n_flagged}  (spot-check these vs filings)")
    print(f"  failed (manual follow-up)    : {n_failed}")
    print(f"  report written               : {args.out / '_pull_report.csv'}")
    print("-" * 60)
    if coverage >= 95.0:
        print("  THRESHOLD MET (>=95%): proceed to Stage 2 (edges).")
    else:
        print("  THRESHOLD NOT MET (<95%): resolve failures before Stage 2.")
    print("=" * 60)
    return 0 if coverage >= 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
