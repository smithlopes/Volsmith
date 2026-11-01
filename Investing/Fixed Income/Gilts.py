"""Download and parse the conventional gilts table from giltsyield.com."""

from __future__ import annotations

import re
from typing import Final

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL: Final = "https://giltsyield.com/bond/"
INDEX_LINKED_URL: Final = "https://giltsyield.com/bond/inflation/"
USER_AGENT: Final = "Mozilla/5.0 (compatible; volsmith-gilts/1.0)"


def _clean_header(value: str) -> str:
    """Convert a displayed table heading to a stable snake_case column name."""
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return {"accr_int": "accrued_interest"}.get(value, value)


def _cell_text(cell) -> str:
    """Return visible cell text, including text nested in links and spans."""
    return " ".join(cell.stripped_strings)


def _numeric(value: str) -> float | None:
    """Extract the first numeric value from a displayed price, yield, or duration."""
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", value.replace("−", "-"))
    return float(match.group(0).replace(",", "")) if match else None


def fetch_gilts(url: str = URL, timeout: int = 30) -> pd.DataFrame:
    """Fetch a giltsyield.com table and return it as a DataFrame.

    Price columns are in pounds per £100 nominal and yield columns are percentages.
    The site's up/down arrows and day counts are omitted from the numeric columns.
    Coupon and yield columns are formatted as percentage strings with two decimals.
    """
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=timeout,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")
    if table is None:
        raise ValueError(f"Could not find a gilts table at {url!r}.")

    header_row = table.find("tr")
    if header_row is None:
        raise ValueError("The gilts table contains no header row.")
    headers = [_clean_header(_cell_text(cell)) for cell in header_row.find_all(["th", "td"])]
    if not headers or "gilt" not in headers:
        raise ValueError("The page structure changed: expected a 'Gilt' table column.")

    records: list[dict[str, str]] = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) != len(headers):
            continue
        records.append(dict(zip(headers, (_cell_text(cell) for cell in cells))))

    if not records:
        raise ValueError("The gilts table contained no data rows.")

    frame = pd.DataFrame.from_records(records)
    for column in ("issued", "matures"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    percentage_columns = {"coupon", "gross_yield", "dmo_yield", "net_yield"}
    numeric_columns = percentage_columns | {
        "clean_price", "accrued_interest", "dirty_price", "mod_duration",
    }
    for column in numeric_columns:
        if column in frame:
            frame[column] = frame[column].map(_numeric).astype("Float64")
    for column in percentage_columns & set(frame.columns):
        frame[column] = frame[column].map(lambda value: f"{value:.2f}%" if pd.notna(value) else pd.NA)

    return frame


def matching_maturities(
    conventional_gilts: pd.DataFrame,
    index_linked_gilts: pd.DataFrame,
) -> pd.DataFrame:
    """Return all securities sharing an exact maturity across the two tables.

    Every original column is retained.  The additional ``security_type`` column
    identifies which source table each security came from.
    """
    if "matures" not in conventional_gilts or "matures" not in index_linked_gilts:
        raise ValueError("Both gilts DataFrames must contain a 'matures' column.")

    conventional = conventional_gilts.copy()
    conventional.insert(0, "security_type", "Conventional")
    index_linked = index_linked_gilts.copy()
    index_linked.insert(0, "security_type", "Index-linked")

    shared_maturities = pd.Index(conventional["matures"].dropna()).intersection(
        pd.Index(index_linked["matures"].dropna())
    )
    if shared_maturities.empty:
        return pd.DataFrame(columns=["security_type", *conventional_gilts.columns])

    matches = pd.concat(
        [
            conventional[conventional["matures"].isin(shared_maturities)],
            index_linked[index_linked["matures"].isin(shared_maturities)],
        ],
        ignore_index=True,
    )
    return matches.sort_values(["matures", "security_type"], ignore_index=True)


if __name__ == "__main__":
    gilts = fetch_gilts(URL)
    index_linked_gilts = fetch_gilts(INDEX_LINKED_URL)
    print("Conventional gilts")
    print(gilts.to_string(index=False))
    print("\nIndex-linked gilts")
    print(index_linked_gilts.to_string(index=False))

    same_maturity = matching_maturities(gilts, index_linked_gilts)
    print("\nSecurities with the exact same maturity in both tables")
    if same_maturity.empty:
        print("None found.")
    else:
        print(same_maturity.to_string(index=False))
