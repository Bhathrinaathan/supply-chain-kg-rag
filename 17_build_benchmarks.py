#!/usr/bin/env python3
"""
17_build_benchmarks.py  —  Non-self-referential abnormal-return benchmarks (Stage 3.5, Task L1).

Produces THREE daily abnormal-return (AR) series per firm so the affected-label can be
triangulated instead of resting on one benchmark (rag_architecture.md §2):

  1. loo  — Leave-one-out sector mean (PRIMARY). AR = r_firm - mean(r of OTHER same-sector
            universe firms). Non-self-referential by construction (focal firm excluded).
  2. ff   — Fama-French 5-factor + momentum residual (ROBUSTNESS). Betas estimated on a trailing
            window, refit periodically → strictly pre-event, no look-ahead.
  3. pca  — PCA approximate-factor residual (ROBUSTNESS). Loadings fit on the TRAIN split only,
            projected forward; residual after removing k common factors.

LEAKAGE SAFETY (the whole point): every AR at day t uses only information available by day t.
  - loo: contemporaneous cross-section of OTHER firms (no future data).
  - ff : betas from a trailing window ending BEFORE the refit day; factors are day-t.
  - pca: loadings frozen on train (< T1); projection uses the day-t cross-section only.

INPUTS
  data/raw/prices/<TICKER>__adjusted.parquet   (from 02; MultiIndex columns handled)
  config/universe.csv                          (ticker, sector[, market_cap])
OUTPUTS
  data/labeled/benchmarks/loo_abnormal_returns.parquet   (long: ticker,date,ret,ar)
  data/labeled/benchmarks/ff_abnormal_returns.parquet
  data/labeled/benchmarks/pca_abnormal_returns.parquet

USAGE
  pip install pandas pyarrow numpy pandas_datareader
  python 17_build_benchmarks.py --train-end 2024-07-01 --pca-k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------- returns panel
def load_returns_panel(prices_dir: Path, tickers: list[str]) -> pd.DataFrame:
    """Wide daily simple-return panel [date x ticker] from adjusted-close parquets."""
    series = {}
    for t in tickers:
        f = prices_dir / f"{t.replace('.', '_')}__adjusted.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        # Flatten MultiIndex columns like ('Close','AAPL') -> 'Close'.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if "Close" not in df.columns:
            continue
        close = df["Close"].astype("float64")
        series[t] = close.pct_change()
    panel = pd.DataFrame(series).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


# ---------------------------------------------------------------------------- 1. LOO sector mean
def loo_sector_abnormal(panel: pd.DataFrame, sector: dict[str, str],
                        weight: dict[str, float] | None) -> pd.DataFrame:
    """
    AR = r_firm - leave-one-out mean of OTHER same-sector firms.
    Equal-weight by default; value-weight if `weight` (static market caps) is supplied.
    Sectors with a single firm fall back to the all-universe LOO mean.
    """
    ar = pd.DataFrame(index=panel.index, columns=panel.columns, dtype="float64")
    by_sector: dict[str, list[str]] = {}
    for t in panel.columns:
        by_sector.setdefault(sector.get(t, "UNKNOWN"), []).append(t)

    all_firms = list(panel.columns)
    for sec, firms in by_sector.items():
        peers = firms if len(firms) > 1 else all_firms      # singleton -> broad fallback
        sub = panel[peers]
        if weight:
            w = np.array([weight.get(f, 1.0) for f in peers], dtype="float64")
            for f in firms:
                others = [p for p in peers if p != f]
                wo = np.array([weight.get(p, 1.0) for p in others])
                bench = (sub[others] * wo).sum(axis=1) / np.where(
                    sub[others].notna().mul(wo).sum(axis=1) == 0, np.nan,
                    sub[others].notna().mul(wo).sum(axis=1))
                ar[f] = panel[f] - bench
        else:
            for f in firms:
                others = [p for p in peers if p != f]
                bench = sub[others].mean(axis=1)            # equal-weight LOO mean
                ar[f] = panel[f] - bench
    return _to_long(panel, ar)


# ---------------------------------------------------------------------------- shared regression
def periodic_refit_residual(excess: pd.DataFrame, factors: pd.DataFrame,
                            est_window: int = 252, refit_every: int = 21) -> pd.DataFrame:
    """
    Residual AR from regressing each firm's excess return on `factors`.
    Betas are re-estimated every `refit_every` days on the trailing `est_window` days
    (ending the day BEFORE the block) and held constant across the block -> no look-ahead.
    """
    common = excess.index.intersection(factors.index)
    excess, F = excess.loc[common], factors.loc[common]
    Fx = np.column_stack([np.ones(len(F)), F.to_numpy()])       # intercept + factors
    ar = pd.DataFrame(index=common, columns=excess.columns, dtype="float64")
    n = len(common)

    for tk in excess.columns:
        y = excess[tk].to_numpy()
        out = np.full(n, np.nan)
        for start in range(est_window, n, refit_every):
            lo, hi = start - est_window, start                  # trailing estimation window
            block_hi = min(start + refit_every, n)
            Xtr, ytr = Fx[lo:hi], y[lo:hi]
            m = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
            if m.sum() < est_window // 2:
                continue
            beta, *_ = np.linalg.lstsq(Xtr[m], ytr[m], rcond=None)
            blk = slice(start, block_hi)
            pred = Fx[blk] @ beta
            out[blk] = y[blk] - pred                             # residual = abnormal return
        ar[tk] = out
    return ar


# ---------------------------------------------------------------------------- 2. Fama-French
def fetch_ff_factors(start: str, end: str) -> pd.DataFrame:
    """FF5 + momentum daily factors from the Kenneth French Data Library (decimal returns)."""
    from pandas_datareader.famafrench import FamaFrenchReader
    ff5 = FamaFrenchReader("F-F_Research_Data_5_Factors_2x3_daily", start=start, end=end).read()[0]
    mom = FamaFrenchReader("F-F_Momentum_Factor_daily", start=start, end=end).read()[0]
    # PeriodIndex (returned by pandas_datareader) must be converted before join
    if hasattr(ff5.index, "to_timestamp"):
        ff5.index = ff5.index.to_timestamp()
    if hasattr(mom.index, "to_timestamp"):
        mom.index = mom.index.to_timestamp()
    ff = ff5.join(mom, how="inner") / 100.0                     # percentages -> decimals
    ff.columns = [c.strip() for c in ff.columns]
    ff.index = pd.to_datetime(ff.index)
    return ff


def ff_abnormal(panel: pd.DataFrame, ff: pd.DataFrame,
                est_window: int, refit_every: int) -> pd.DataFrame:
    factor_cols = [c for c in ff.columns if c.upper() != "RF"]
    rf = ff["RF"].reindex(panel.index).ffill().fillna(0.0)
    excess = panel.sub(rf, axis=0)
    ar = periodic_refit_residual(excess, ff[factor_cols].reindex(panel.index),
                                 est_window, refit_every)
    return _to_long(panel, ar)


# ---------------------------------------------------------------------------- 3. PCA factors
def pca_abnormal(panel: pd.DataFrame, train_end: str, k: int,
                 est_window: int, refit_every: int) -> pd.DataFrame:
    """
    Fit k PCA loadings on the TRAIN split only (dates < train_end), project all dates onto them
    to form factor-return series, then regress each firm on those factors (periodic refit).
    Note: a single firm contributes ~1/N to any component, so residual self-reference is minor
    (documented caveat in rag_architecture.md §2).
    """
    train = panel.loc[panel.index < pd.to_datetime(train_end)].dropna(axis=1, how="all")
    Xtr = train.fillna(0.0).to_numpy()
    mu = Xtr.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    load = Vt[:k]                                               # (k x N) firm loadings on train
    cols = list(train.columns)

    P = panel[cols].fillna(0.0).to_numpy() - mu                 # center with TRAIN means
    factor_returns = pd.DataFrame(P @ load.T, index=panel.index,
                                  columns=[f"pc{i+1}" for i in range(k)])
    ar = periodic_refit_residual(panel[cols], factor_returns, est_window, refit_every)
    return _to_long(panel[cols], ar)


# ---------------------------------------------------------------------------- helpers
def _to_long(panel: pd.DataFrame, ar: pd.DataFrame) -> pd.DataFrame:
    panel2 = panel.copy()
    panel2.index.name = "date"
    ar2 = ar.copy()
    ar2.index.name = "date"
    r = panel2.reset_index().melt(id_vars="date", var_name="ticker", value_name="ret")
    a = ar2.reset_index().melt(id_vars="date", var_name="ticker", value_name="ar")
    out = r.merge(a, on=["date", "ticker"])
    return out.dropna(subset=["ar"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, default=Path("data/raw/prices"))
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/labeled/benchmarks"))
    ap.add_argument("--train-end", default="2024-07-01", help="PCA loadings fit on dates < this")
    ap.add_argument("--pca-k", type=int, default=3)
    ap.add_argument("--est-window", type=int, default=252)
    ap.add_argument("--refit-every", type=int, default=21)
    ap.add_argument("--weight", choices=["equal", "value"], default="equal")
    args = ap.parse_args()

    uni = pd.read_csv(args.universe)
    uni = uni[uni.get("has_price", True) != False]              # noqa: E712
    tickers = uni["ticker"].astype(str).tolist()

    # Load sector from dedicated sector_map.csv (universe.csv has no sector column)
    sector_map_path = args.universe.parent / "sector_map.csv"
    if sector_map_path.exists():
        sm = pd.read_csv(sector_map_path)
        sector = dict(zip(sm["ticker"].astype(str), sm["sector"].astype(str)))
        print(f"[info] sector map: {len(sector)} tickers, {len(set(sector.values()))} sectors")
    else:
        print(f"[warn] {sector_map_path} not found — all firms in one LOO group")
        sector = {t: "all" for t in tickers}

    weight = None
    if args.weight == "value" and "market_cap" in uni.columns:
        weight = dict(zip(uni["ticker"].astype(str), uni["market_cap"].astype(float)))
    elif args.weight == "value":
        print("[warn] --weight value requested but no market_cap column; using equal weights.")

    panel = load_returns_panel(args.prices, tickers)
    print(f"[info] return panel: {panel.shape[1]} firms x {panel.shape[0]} days "
          f"({panel.index.min().date()} -> {panel.index.max().date()})")

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. LOO (primary) — always available, no external data.
    loo = loo_sector_abnormal(panel, sector, weight)
    loo.to_parquet(args.out / "loo_abnormal_returns.parquet")
    print(f"[ok] loo  -> {len(loo):,} firm-day ARs")

    # 2. FF (robustness) — needs Ken French download.
    try:
        ff = fetch_ff_factors(str(panel.index.min().date()), str(panel.index.max().date()))
        ffa = ff_abnormal(panel, ff, args.est_window, args.refit_every)
        ffa.to_parquet(args.out / "ff_abnormal_returns.parquet")
        print(f"[ok] ff   -> {len(ffa):,} firm-day ARs")
    except Exception as exc:  # noqa: BLE001 -- FF is robustness, not critical path
        print(f"[warn] FF benchmark skipped ({type(exc).__name__}: {exc}). "
              "Install pandas_datareader / check network; LOO is sufficient to proceed.")

    # 3. PCA (robustness).
    try:
        pca = pca_abnormal(panel, args.train_end, args.pca_k, args.est_window, args.refit_every)
        pca.to_parquet(args.out / "pca_abnormal_returns.parquet")
        print(f"[ok] pca  -> {len(pca):,} firm-day ARs (k={args.pca_k}, fit on <{args.train_end})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PCA benchmark skipped ({type(exc).__name__}: {exc}).")

    print("\n[done] benchmarks in", args.out, "-> next: 18_relabel_abnormal.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
