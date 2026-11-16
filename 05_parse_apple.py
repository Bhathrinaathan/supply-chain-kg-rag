#!/usr/bin/env python3
"""
05_parse_apple.py  —  Apple Supplier List -> supplier_of(supplier -> AAPL) edges.

Apple publishes a yearly PDF listing its top-200 suppliers (~98% of direct spend). Each row
is a supplier + a manufacturing location; multiple facility rows map to the same parent, so
we COLLAPSE to the parent company and resolve to a ticker where one exists in our universe.
One list per fiscal year -> each edge carries the correct valid_from (fiscal-year start).

Download the PDFs first (one per FY) from apple.com and place them in --pdf-dir named like
`apple_suppliers_FY2022.pdf`. Parsing uses pdfplumber table extraction (see pdf-reading
guidance: data-heavy table PDF -> extract_tables()).

OUTPUTS:
    data/raw/apple/apple_edges.csv   (supplier_name, supplier_ticker?, valid_from, source_url)

USAGE:
    pip install pdfplumber
    python 05_parse_apple.py --pdf-dir data/raw/apple --universe config/universe.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except ImportError:  # keep the failure message actionable
    print("[FATAL] pip install pdfplumber")
    raise

APPLE_TICKER = "AAPL"
FY_RE = re.compile(r"FY(\d{4})", re.IGNORECASE)


def build_name_index(uni: pd.DataFrame) -> list[tuple[str, str]]:
    """(lowercased name/alias, ticker) pairs for fuzzy supplier->ticker resolution.
    Minimum 4 chars enforced to avoid spurious substring hits from very short tokens
    (e.g. NaN aliases or 2-letter abbreviations matching Chinese place names).
    """
    pairs: list[tuple[str, str]] = []
    for _, r in uni.iterrows():
        ticker = str(r["ticker"])
        name = str(r["name"]) if not pd.isna(r["name"]) else ""
        if len(name) >= 4:
            pairs.append((name.lower(), ticker))
        aliases_raw = r.get("name_aliases")
        aliases = "" if pd.isna(aliases_raw) else str(aliases_raw)
        for a in filter(None, (x.strip() for x in aliases.split("|"))):
            if len(a) >= 4:
                pairs.append((a.lower(), ticker))
    return pairs


def resolve_ticker(supplier_name: str, index: list[tuple[str, str]]) -> str:
    """Best-effort match. Unmatched -> ''.
    Uses word-boundary at START of the token so 'micron technology' does not
    match 'unimicron technology' (no word boundary before 'micron' there),
    but 'coherent corp' still matches 'coherent corporation' (boundary at start).
    """
    s = supplier_name.lower()
    for name, ticker in index:
        # name starts at a word boundary inside supplier string
        if re.search(r'\b' + re.escape(name), s):
            return ticker
        # supplier string starts at a word boundary inside universe name
        if re.search(r'\b' + re.escape(s), name):
            return ticker
    return ""


_SKIP_EXACT = {"supplier name", "supplier", "name",
               "primary locations where manufacturing for apple occurs",
               "(continued)"}
_SKIP_PREFIX = ("supplier list", "© ", "apple inc", "apple supplier",
                "clean energy")
_SKIP_STARTSWITH_PAREN = True   # drop sub-entries like "(Foxconn)", "(Foxlink)"

def parse_pdf(path: Path) -> list[str]:
    """
    Extract supplier names from one Apple list PDF.
    Apple's list is a two-column borderless layout — extract_tables() returns nothing.
    We crop each page to the left (supplier-name) column and collect non-empty text lines.

    Column boundary varies by page orientation:
      Landscape (width > 700pt, FY2021+): names at x0~36, locations at x0~470 → crop at 460.
      Portrait  (width ≤ 700pt, FY2020):  names at x0~76, locations at x0~297 → crop at 270.
    Page 0 is the cover/intro — skip it; supplier data starts on page 1.
    """
    names: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[1:]:        # skip cover page
            crop_x = 270 if page.width <= 700 else 460
            left = page.crop((0, 0, crop_x, page.height))
            text = left.extract_text() or ""
            for line in text.splitlines():
                cell = line.strip()
                if not cell or len(cell) < 3:
                    continue
                cl = cell.lower()
                if cl in _SKIP_EXACT:
                    continue
                if any(cl.startswith(p) for p in _SKIP_PREFIX):
                    continue
                if _SKIP_STARTSWITH_PAREN and cell.startswith("("):
                    continue
                names.append(cell)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/raw/apple"))
    ap.add_argument("--universe", type=Path, default=Path("config/universe.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/apple/apple_edges.csv"))
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("apple_suppliers_FY*.pdf"))
    if not pdfs:
        print(f"[FATAL] no apple_suppliers_FY*.pdf in {args.pdf_dir}. "
              "Download from apple.com first.")
        return 2

    uni = pd.read_csv(args.universe)
    index = build_name_index(uni)

    # Pass 1: parse all PDFs into per-year supplier sets
    year_suppliers: dict[str, set[str]] = {}
    fy_order: list[str] = []
    for pdf in pdfs:
        m = FY_RE.search(pdf.name)
        fy = m.group(1) if m else "unknown"
        names = parse_pdf(pdf)
        year_suppliers[fy] = set(names)
        fy_order.append(fy)
        print(f"[ok] {pdf.name}: {len(set(names))} unique suppliers (FY{fy})")

    fy_order_sorted = sorted(fy_order)  # chronological

    # Pass 2: build rows with valid_from and valid_to
    rows = []
    for i, fy in enumerate(fy_order_sorted):
        valid_from = f"{fy}-10-01" if fy != "unknown" else ""
        next_fy = fy_order_sorted[i + 1] if i + 1 < len(fy_order_sorted) else None
        next_set = year_suppliers.get(next_fy, set()) if next_fy else None

        for name in sorted(year_suppliers[fy]):
            # valid_to: set to next FY start if supplier dropped off in next year
            if next_set is not None and name not in next_set:
                valid_to = f"{next_fy}-10-01"
            else:
                valid_to = ""
            rows.append({
                "supplier_name": name,
                "supplier_ticker": resolve_ticker(name, index),
                "customer_ticker": APPLE_TICKER,
                "type": "supplier_of",
                "valid_from": valid_from,
                "valid_to": valid_to,
                "fiscal_year": fy,
                "method": "apple",
                "source_url": f"https://www.apple.com/supplier-responsibility/ (FY{fy} list)",
                "confidence": 0.9,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    matched = (df["supplier_ticker"] != "").sum()
    with_valid_to = (df["valid_to"] != "").sum()
    print(f"\n[done] {len(df)} supplier rows -> {args.out}")
    print(f"       {matched} matched to a universe ticker; {with_valid_to} edges have valid_to (dropped off).")
    print("       Unmatched suppliers become KG nodes without a price series (has_price=false).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
