#!/usr/bin/env python3
"""
download_apple_pdfs.py  —  Fetch correct Apple Supplier List PDFs into data/raw/apple/.

The Supplier List is a ~300 KB standalone PDF distinct from the Progress Reports.
Run with the project venv:
    .venv/bin/python3 download_apple_pdfs.py

Confirmed URLs as of June 2026. FY2024 must be downloaded manually — see note below.
"""
from pathlib import Path
import sys

try:
    import httpx
except ImportError:
    sys.exit("httpx not found — run: .venv/bin/pip install httpx")

DEST = Path("data/raw/apple")
DEST.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.apple.com/supplier-responsibility/",
    "Accept": "application/pdf,*/*",
}

DOWNLOADS = [
    (
        "apple_suppliers_FY2020.pdf",
        "https://www.apple.com/euro/supplier-responsibility/k/generic/pdf/Apple-Supplier-List.pdf",
    ),
    (
        "apple_suppliers_FY2021.pdf",
        "https://www.apple.com/supply-chain/pdf/Apple-FY21-Supplier-List.pdf",
    ),
    (
        "apple_suppliers_FY2023.pdf",
        "https://s203.q4cdn.com/367071867/files/doc_downloads/2024/04/Apple-Supplier-List.pdf",
    ),
    # FY2024: Apple's official redirect (supplychainreports.apple/Apple-Supplier-List)
    # still serves FY2023 as of June 2026. Download manually:
    #   1. Open https://www.apple.com/supply-chain/ in Chrome
    #   2. Find "Supplier List" under Additional Reports → download the PDF
    #   3. Save it here as:  data/raw/apple/apple_suppliers_FY2024.pdf
    #   Expected size: ~300 KB. If the downloaded file is >1 MB it is the Progress Report — wrong file.
]

def download(filename: str, url: str) -> None:
    dest = DEST / filename
    print(f"[{filename}] fetching ...", end=" ", flush=True)
    with httpx.Client(follow_redirects=True, timeout=60, headers=HEADERS) as client:
        r = client.get(url)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "pdf" not in ct.lower() and not r.content[:4] == b"%PDF":
            print(f"WARN: unexpected content-type '{ct}' — may be wrong file")
        size_kb = len(r.content) / 1024
        if size_kb > 2000:
            print(f"WARN: {size_kb:.0f} KB — this looks like a Progress Report, not the Supplier List (~300 KB)")
        dest.write_bytes(r.content)
        print(f"done  ({size_kb:.0f} KB) -> {dest}")

def main() -> int:
    ok = True
    for filename, url in DOWNLOADS:
        try:
            download(filename, url)
        except Exception as e:
            print(f"FAIL: {e}")
            ok = False

    print()
    # Check FY2024
    fy24 = DEST / "apple_suppliers_FY2024.pdf"
    if not fy24.exists():
        print("⚠️  FY2024 missing — download manually (see comment in this script)")
        ok = False
    else:
        size_kb = fy24.stat().st_size / 1024
        if size_kb > 2000:
            print(f"⚠️  FY2024 is {size_kb:.0f} KB — likely the Progress Report. Replace with the Supplier List (~300 KB).")
            ok = False
        else:
            print(f"✓  FY2024 looks correct ({size_kb:.0f} KB)")

    print()
    present = sorted(DEST.glob("apple_suppliers_FY*.pdf"))
    print(f"Files in {DEST}:")
    for p in present:
        print(f"  {p.name:40s}  {p.stat().st_size/1024:8.0f} KB")

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
