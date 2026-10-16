"""Performance table for selected MSCI USD net total return indices.

Edit the settings below and run this script. It requests index levels with the
``mscidata`` package and prints a table using the same metrics as
``pre_analysis.py``. Index codes are MSCI's codes for these USD indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from mscidata import msci
except ImportError as error:
    raise ImportError(
        "Could not import MSCI Data. Install it with `pip install msci-data`."
    ) from error


# ---------------- settings ----------------
INDICES = {
    "MSCI World": "990100",
    "MSCI All Country": "892400",
    "MSCI Emerging Markets": "891800",
    "MSCI World Momentum": "703755",
    "MSCI World Quality": "702787",
    "MSCI All Country Momentum": "703026",
}

START_DATE = "2000-01-01"  # Request as much available history as MSCI provides.
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
RETURN_VARIANT = "NETR"  # Net total returns (withholding tax deducted).
RISK_FREE_RATE_ANNUAL = 0.04
ANNUALISE_MULTI_YEAR_ROLLS = False
USE_COMMON_START_DATE = True
# ------------------------------------------


def rolling_returns(levels: pd.Series, years: int, annualise: bool) -> pd.Series:
    """Calendar-based rolling return using the last level on/before the target."""
    targets = levels.index - pd.DateOffset(years=years)
    positions = levels.index.searchsorted(targets, side="right") - 1
    valid = targets >= levels.index[0]
    start_levels = pd.Series(
        np.where(valid, levels.to_numpy()[np.clip(positions, 0, None)], np.nan),
        index=levels.index,
    )
    result = levels / start_levels - 1
    if annualise and years > 1:
        result = (1 + result) ** (1 / years) - 1
    return result.dropna()


def drawdown_stats(levels: pd.Series) -> tuple[float, float | str]:
    """Return maximum drawdown and days from its trough to recovery."""
    drawdown = levels / levels.cummax() - 1
    trough_date = drawdown.idxmin()
    prior_peak = levels.cummax().loc[trough_date]
    recovered = levels.loc[trough_date:]
    recovered = recovered.loc[recovered >= prior_peak]
    if recovered.empty:
        return float(drawdown.min()), "Not recovered"
    return float(drawdown.min()), int((recovered.index[0] - trough_date).days)


def normalise_msc_data(raw_data: object) -> pd.DataFrame:
    """Validate and standardise the MSCI response to INDEX_CODE/DATE/LEVEL."""
    if raw_data is None:
        raise ValueError("MSCI returned no data.")
    history = pd.DataFrame(raw_data).copy()
    if history.empty:
        raise ValueError("MSCI returned an empty dataset.")

    history.columns = [str(column).strip().upper() for column in history.columns]
    required = {"INDEX_CODE", "DATE", "LEVEL"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "MSCI response is missing required columns: " + ", ".join(sorted(missing))
        )

    history["INDEX_CODE"] = (
        history["INDEX_CODE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    history["DATE"] = pd.to_datetime(history["DATE"], errors="coerce", utc=True).dt.tz_convert(None)
    history["LEVEL"] = pd.to_numeric(history["LEVEL"], errors="coerce")
    history = history.dropna(subset=["DATE", "LEVEL"])

    if "CURRENCY" in history.columns:
        history["CURRENCY"] = history["CURRENCY"].astype(str).str.strip().str.upper()
        non_usd = history.loc[history["CURRENCY"] != "USD", "CURRENCY"].unique()
        if len(non_usd):
            raise ValueError(
                "MSCI returned non-USD levels ("
                + ", ".join(sorted(non_usd))
                + "); this table requires USD data."
            )

    history = history.sort_values(["INDEX_CODE", "DATE"])
    history = history.drop_duplicates(["INDEX_CODE", "DATE"], keep="last")
    if history.empty:
        raise ValueError("No valid MSCI observations remained after normalisation.")
    return history


def load_msc_levels() -> dict[str, pd.Series]:
    """Download all selected USD net return index levels in one request."""
    codes = list(dict.fromkeys(INDICES.values()))
    print(
        f"Requesting MSCI USD net total return levels for {len(codes)} indices "
        f"from {START_DATE} to {END_DATE}..."
    )
    raw = msci.get_levels(codes, START_DATE, END_DATE, RETURN_VARIANT)
    history = normalise_msc_data(raw)

    levels_by_name = {}
    for name, code in INDICES.items():
        rows = history.loc[history["INDEX_CODE"] == code, ["DATE", "LEVEL"]]
        if rows.empty:
            print(f"[{name}] No history returned for MSCI code {code}; skipping.")
            continue
        series = rows.set_index("DATE")["LEVEL"].astype(float).sort_index()
        if (series <= 0).any():
            raise ValueError(f"{name} contains zero or negative index levels.")
        levels_by_name[name] = series

    if not levels_by_name:
        raise ValueError("MSCI returned no observations for the selected indices.")
    return levels_by_name


def select_analysis_period(
    data: dict[str, pd.Series], use_common_start: bool
) -> tuple[dict[str, pd.Series], pd.Timestamp | None]:
    """Clip indices to their shared first date when the setting is enabled."""
    if not use_common_start:
        return data, None
    common_start = max(series.index.min() for series in data.values())
    clipped = {
        name: series.loc[series.index >= common_start]
        for name, series in data.items()
    }
    return clipped, common_start


def analyse_index(name: str, levels: pd.Series, inception_date: pd.Timestamp) -> dict:
    """Calculate metrics and rolling returns in the pre_analysis table format."""
    levels = levels.sort_index()
    returns = levels.pct_change().dropna()
    if len(levels) < 2 or returns.empty:
        raise ValueError(f"Not enough observations to analyse {name}.")

    years = (levels.index[-1] - levels.index[0]).days / 365.25
    if years <= 0:
        raise ValueError(f"Invalid date range for {name}.")
    observations_per_year = len(returns) / years
    total_return = levels.iloc[-1] / levels.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    annualised_sd = returns.std() * np.sqrt(observations_per_year)

    daily_risk_free = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / observations_per_year) - 1
    excess = returns - daily_risk_free
    sharpe = excess.mean() * observations_per_year / annualised_sd if annualised_sd > 0 else np.nan
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(observations_per_year)
    sortino = excess.mean() * observations_per_year / downside if downside > 0 else np.nan

    max_drawdown, recovery_days = drawdown_stats(levels)
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    row = {
        "Index": name,
        "Type": "Index",
        "Years of Data": round(years, 2),
        "Analysis Start Date": levels.index[0].date(),
        "Inception Date": inception_date.date(),
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann. SD": annualised_sd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_drawdown,
        "Days to Recovery": recovery_days,
    }
    for horizon in (1, 3):
        rolled = rolling_returns(levels, horizon, ANNUALISE_MULTI_YEAR_ROLLS)
        row[f"{horizon}Y roll min"] = rolled.min() if not rolled.empty else np.nan
        row[f"{horizon}Y roll med"] = rolled.median() if not rolled.empty else np.nan
        row[f"{horizon}Y roll max"] = rolled.max() if not rolled.empty else np.nan
    return row


def build_summary(
    full_data: dict[str, pd.Series], analysis_data: dict[str, pd.Series]
) -> pd.DataFrame:
    rows = [
        analyse_index(name, levels, full_data[name].index[0])
        for name, levels in analysis_data.items()
    ]
    summary = pd.DataFrame(rows).set_index("Index")
    summary = summary.sort_values("CAGR", ascending=False, na_position="last")
    if USE_COMMON_START_DATE:
        summary = summary.drop(columns=["Inception Date"])
    else:
        summary = summary.drop(columns=["Analysis Start Date"])
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    percentage_columns = [
        "Total Return", "CAGR", "Ann. SD", "Max DD",
        "1Y roll min", "1Y roll med", "1Y roll max",
        "3Y roll min", "3Y roll med", "3Y roll max",
    ]
    display = summary.copy()
    display[percentage_columns] = display[percentage_columns].apply(
        lambda column: column.map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    )
    display[["Sharpe", "Sortino", "Calmar"]] = display[["Sharpe", "Sortino", "Calmar"]].round(2)
    display["Years of Data"] = display["Years of Data"].map(lambda value: f"{value:.2f}")
    pd.set_option("display.width", 250, "display.max_columns", None)
    print(display.to_string())


def main() -> None:
    full_data = load_msc_levels()
    analysis_data, common_start = select_analysis_period(full_data, USE_COMMON_START_DATE)
    if common_start is not None:
        print(f"Using a common analysis period from {common_start.date()}.")
    else:
        print("Using each index's full available history.")
    print("Returns: MSCI USD net total return index levels (NETR).")
    print_summary(build_summary(full_data, analysis_data))


if __name__ == "__main__":
    main()
