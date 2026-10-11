"""
iShares fund loader (ETFs and physical-commodity ETCs).

- Downloads each fund's Excel file into ./ishares_xls/ (overwritten every run)
- Parses every sheet into pandas DataFrames ready for analysis
- Handles products with no holdings / exposure data (e.g. physical gold ETCs)
"""
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------
# 1) FUNDS: ticker -> download link  (add as many as you like)
# ----------------------------------------------------------------------
FUNDS = {
    "SWDA": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=251882&component=fundDownloadV2&userType=individual"),
    "IWMO": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=270051&component=fundDownloadV2&userType=individual"),
    "ACWI": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=251850&component=fundDownloadV2&userType=individual"),
    "SEMA": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=251858&component=fundDownloadV2&userType=individual"),
    "SGLN": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=258441&component=fundDownloadV2&userType=individual"),
    "IWQU": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=270054&component=fundDownloadV2&userType=individual"),
    "MVOL": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=251382&component=fundDownloadV2&userType=individual"),
    "IWVL": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=270048&component=fundDownloadV2&userType=individual"),
    "WSML": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=296576&component=fundDownloadV2&userType=individual"),
    "SSLN": ("https://www.blackrock.com/varnish-api/uk-retail01-product-data/product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=ishares-uk&locale=en_GB&portfolioId=258443&component=fundDownloadV2&userType=individual"),
    # "IWO": "PASTE_LINK_HERE",
}

# Folder sits next to this script, whatever directory you run it from
OUT_DIR = Path(__file__).resolve().parent / "ishares_xls"
HEADERS = {"User-Agent": "Mozilla/5.0"}

NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
SS = "{urn:schemas-microsoft-com:office:spreadsheet}"

HOLDINGS_COLS = ["Issuer Ticker", "Name", "Sector", "Asset Class", "Market Value",
                 "Weight (%)", "Notional Value", "Nominal", "Market Currency"]


# ----------------------------------------------------------------------
# 2) Download (always overwrites the previous file)
# ----------------------------------------------------------------------
def download_fund(ticker: str, url: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f"{ticker}.xls"
    tmp = final.with_suffix(".tmp")

    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    if b"Workbook" not in r.content[:2000]:  # guard against HTML/error pages
        raise ValueError(f"{ticker}: response is not an iShares Excel file "
                         f"(Content-Type: {r.headers.get('Content-Type')})")

    tmp.write_bytes(r.content)
    shutil.move(str(tmp), str(final))  # replaces existing file only on success
    return final


# ----------------------------------------------------------------------
# 3) Parsing helpers
# ----------------------------------------------------------------------
def read_sheets(path: Path) -> dict:
    """SpreadsheetML (.xls that is really XML) -> {sheet_name: list of rows}."""
    root = ET.parse(path).getroot()
    sheets = {}
    for ws in root.findall("ss:Worksheet", NS):
        rows = []
        for row in ws.findall(".//ss:Row", NS):
            vals = []
            for cell in row.findall("ss:Cell", NS):
                idx = cell.get(SS + "Index")
                if idx:
                    vals += [None] * (int(idx) - 1 - len(vals))
                d = cell.find("ss:Data", NS)
                vals.append(d.text.strip() if d is not None and d.text else None)
            rows.append(vals)
        sheets[ws.get(SS + "Name")] = rows
    return sheets


def num(x):
    """'USD 8,851,906,841.52' / '1,011,825,245' / '-19.54' -> float."""
    if x is None:
        return None
    s = re.sub(r"[^0-9.\-]", "", str(x))
    try:
        return float(s)
    except ValueError:
        return None


def to_date(x):
    """Handles '08/Oct/2026', '30/Sept/2026', 'Thu, 08 Oct 2026', 'as of 08/Oct/2026'."""
    if x is None:
        return pd.NaT
    s = re.sub(r"^as of\s*", "", str(x).strip(), flags=re.I).replace("Sept", "Sep")
    s = re.sub(r"^[A-Za-z]{3},\s*", "", s)
    return pd.to_datetime(s, errors="coerce")


def first(rows, pred):
    return next((i for i, r in enumerate(rows) if r and pred(r)), None)


def as_of_after(rows, label):
    for r in rows:
        if r and r[0] == label and len(r) > 1:
            return to_date(r[1])
    return pd.NaT


def get_fact(d: dict, *names, default=None):
    """Look up the first matching label in Key Facts (labels differ between ETFs and ETCs)."""
    kf = d["key_facts"].set_index("Item")["Value"]
    for n in names:
        if n in kf.index:
            return kf[n]
    return default


# ----------------------------------------------------------------------
# 4) Sheet -> DataFrame parsers (all tolerate empty / missing sheets)
# ----------------------------------------------------------------------
def parse_holdings(rows, ticker):
    """Returns an empty frame with the standard columns for funds with no holdings
    (e.g. physical gold ETCs, where the Holdings sheet is blank)."""
    as_of = as_of_after(rows, "as of")
    h = first(rows, lambda r: r[0] == "Issuer Ticker")
    if h is None:
        df = pd.DataFrame(columns=HOLDINGS_COLS)
    else:
        cols = rows[h]
        body = []
        for r in rows[h + 1:]:
            if not r or r[0] is None or len(r) < 9:
                break
            body.append(r[:len(cols)])
        df = pd.DataFrame(body, columns=cols)
        for c in ["Market Value", "Notional Value", "Nominal", "Weight (%)"]:
            df[c] = df[c].map(num)
    df.insert(0, "Fund", ticker)
    df.insert(1, "As Of", as_of)
    return df


def parse_nav(rows):
    data = [(to_date(r[0]), num(r[1])) for r in rows
            if r and len(r) > 1 and r[1] and pd.notna(to_date(r[0]))]
    return (pd.DataFrame(data, columns=["Date", "NAV"])
            .sort_values("Date").reset_index(drop=True))


def parse_growth(rows):
    if not rows:
        return pd.DataFrame(columns=["Date", "Fund", "Benchmark"])
    names = rows[0][1:3]
    data = [(to_date(r[0]), num(r[1]), num(r[2])) for r in rows[1:]
            if r and len(r) > 2 and pd.notna(to_date(r[0]))]
    df = pd.DataFrame(data, columns=["Date", "Fund", "Benchmark"])
    df.attrs["names"] = names
    return df.sort_values("Date").reset_index(drop=True)


def parse_key_value(rows):
    out = []
    for r in rows:
        if r and len(r) > 1 and r[0]:
            as_of = to_date(r[2]) if len(r) > 2 and r[2] else pd.NaT
            out.append((r[0], r[1], as_of))
    return pd.DataFrame(out, columns=["Item", "Value", "As Of"])


def parse_exposure(rows):
    """Returns (sector_df, geography_df); both empty if the sheet has no data."""
    results, name, as_of, started = {}, None, pd.NaT, False
    for r in rows:
        if not r:
            started = False
            continue
        if r[0] in ("Sector", "Geography/Locations"):
            name, started, as_of, results[name] = r[0], False, pd.NaT, []
        elif r[0] == "as of":
            as_of = to_date(r[1])
        elif r[0] == "Type":
            started = True
        elif started and name and len(r) > 1 and num(r[1]) is not None:
            results[name].append((r[0], num(r[1]), r[2] if len(r) > 2 else None, as_of))
    sector = pd.DataFrame(results.get("Sector", []), columns=["Sector", "Weight (%)", "Code", "As Of"])
    geo = pd.DataFrame(results.get("Geography/Locations", []),
                       columns=["Country", "Weight (%)", "Code", "As Of"])
    return sector.drop(columns="Code"), geo


def parse_performance(rows):
    out, section, as_of, header = [], None, pd.NaT, None
    for r in rows:
        if not r:
            continue
        if r[0] in ("Discrete", "Annualised", "Cumulative", "Calendar Year"):
            section, as_of, header = r[0], pd.NaT, None
        if "as of" in r:
            as_of = to_date(r[r.index("as of") + 1])
            continue
        if r[0] is None and len(r) > 1 and any(r[1:]):
            header = [re.sub(r"\s+", " ", c) if c else c for c in r]
        elif r[0] in ("Total Return (%)", "Benchmark (%)") and header:
            for p, v in zip(header[1:], r[1:]):
                if p and num(v) is not None:
                    out.append((section, as_of, p, r[0], num(v)))
    return pd.DataFrame(out, columns=["Section", "As Of", "Period", "Series", "Return (%)"])


def parse_fund(path: Path, ticker: str) -> dict:
    s = read_sheets(path)
    sector, geo = parse_exposure(s.get("Exposure Breakdowns", []))
    growth_key = next((k for k in s if k.startswith("Growth")), None)
    out = {
        "name": s["Fund Header"][0][0],
        "holdings": parse_holdings(s.get("Holdings", []), ticker),
        "nav": parse_nav(s.get("Historical NAVs", [])),
        "growth_10k": parse_growth(s.get(growth_key, [])),
        "key_facts": parse_key_value(s.get("Key Facts", [])),
        "characteristics": parse_key_value(s.get("Portfolio Characteristics", [])),
        "sector": sector,
        "geography": geo,
        "performance": parse_performance(s.get("Performance Returns", [])),
    }
    # Fund-type metadata (labels differ between ETFs and ETCs)
    out["asset_class"] = get_fact(out, "Asset Class")               # 'Equity', 'Commodity', ...
    out["currency"] = get_fact(out, "Share Class Currency", "Base Currency", "Fund Base Currency")
    out["inception"] = to_date(get_fact(out, "Share Class Launch Date", "Fund Launch Date"))
    out["has_holdings"] = not out["holdings"].empty
    out["benchmark"] = get_fact(out, "Benchmark Index", "Index")
    return out


# ----------------------------------------------------------------------
# 5) Main: refresh everything, return {ticker: {table: DataFrame}}
# ----------------------------------------------------------------------
def load_all(refresh: bool = True) -> dict:
    data = {}
    for ticker, url in FUNDS.items():
        path = OUT_DIR / f"{ticker}.xls"
        if refresh:
            try:
                path = download_fund(ticker, url)
            except Exception as e:
                if path.exists():
                    print(f"[{ticker}] download failed ({e}); using last saved file")
                else:
                    print(f"[{ticker}] download failed ({e}); skipping")
                    continue
        if not path.exists():
            print(f"[{ticker}] no saved file; skipping")
            continue
        data[ticker] = parse_fund(path, ticker)
    return data


if __name__ == "__main__":
    data = load_all()

    # Holdings of all funds that have them, stacked into one DataFrame
    frames = [d["holdings"] for d in data.values() if d["has_holdings"]]
    all_holdings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    for t, d in data.items():
        print(f"\n=== {t}: {d['name']} ({d['asset_class']}, {d['currency']}) ===")
        print("NAV history:", d["nav"].shape, "| inception", d["inception"].date())
        if d["has_holdings"]:
            h = d["holdings"]
            print("Holdings:", h.shape, "| as of", h["As Of"].iloc[0].date())
            print(h.head())
            print("Weights sum:", round(h["Weight (%)"].sum(), 2))
            print(d["sector"])
        else:
            print("No holdings / exposure data for this product (NAV and facts only).")
            print(d["key_facts"])