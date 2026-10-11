"""
Preliminary ETF analysis built on etf_loader.py (same folder).
One row per fund, computed from the fund's daily NAV history.
"""
import numpy as np
import pandas as pd

from etf_loader import load_all

# ---------------- settings ----------------
RF_ANNUAL = 0.04                 # risk-free rate for Sharpe/Sortino (e.g. 0.04 = 4%)
ANNUALISE_MULTI_YEAR_ROLLS = False   # 3Y rolling returns shown as annualised (CAGR) if True
REFRESH = True                  # False = use the files already saved, no download
USE_COMMON_START_DATE = True    # True = compare all funds from the latest NAV start date
# ------------------------------------------


def select_analysis_period(data: dict, use_common_start: bool) -> tuple[dict, pd.Timestamp | None]:
    """Return data clipped to a common NAV start, or unchanged for full histories."""
    prepared = {}
    first_dates = []
    for ticker, fund in data.items():
        nav = fund["nav"].copy()
        if not nav.empty:
            nav["Date"] = pd.to_datetime(nav["Date"])
            nav = nav.sort_values("Date")
            first_dates.append(nav["Date"].iloc[0])
        prepared[ticker] = {**fund, "nav": nav}

    if not use_common_start:
        return prepared, None
    if not first_dates:
        raise ValueError("Cannot select a common analysis period: no NAV history was loaded.")

    common_start = max(first_dates)
    for fund in prepared.values():
        fund["nav"] = fund["nav"].loc[fund["nav"]["Date"] >= common_start].reset_index(drop=True)
    return prepared, common_start


def rolling_returns(nav: pd.Series, years: int, annualise: bool) -> pd.Series:
    """Calendar-based rolling return: NAV(t) / NAV(t - N years) - 1, using the last
    available NAV on or before the lookback date."""
    targets = nav.index - pd.DateOffset(years=years)
    pos = nav.index.searchsorted(targets, side="right") - 1
    valid = targets >= nav.index[0]
    start = pd.Series(np.where(valid, nav.values[np.clip(pos, 0, None)], np.nan), index=nav.index)
    r = nav / start - 1
    if annualise and years > 1:
        r = (1 + r) ** (1 / years) - 1
    return r.dropna()


def drawdown_stats(nav: pd.Series):
    """Max drawdown and calendar days from the trough back to the prior peak."""
    peak = nav.cummax()
    dd = nav / peak - 1
    trough_date = dd.idxmin()
    max_dd = dd.min()
    peak_level = peak.loc[trough_date]
    after = nav.loc[trough_date:]
    recovered = after[after >= peak_level]
    if recovered.empty:
        return max_dd, np.nan, trough_date, None          # not yet recovered
    rec_date = recovered.index[0]
    return max_dd, (rec_date - trough_date).days, trough_date, rec_date


def analyse_fund(ticker: str, d: dict) -> dict:
    nav = d["nav"].set_index("Date")["NAV"].sort_index()
    ret = nav.pct_change().dropna()

    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ppy = len(ret) / years                                   # observations per year (~252)

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(ppy)

    rf_d = (1 + RF_ANNUAL) ** (1 / ppy) - 1
    excess = ret - rf_d
    sharpe = excess.mean() * ppy / vol
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(ppy)
    sortino = excess.mean() * ppy / downside

    max_dd, days_rec, _, rec_date = drawdown_stats(nav)
    calmar = cagr / abs(max_dd)

    # inception: share class launch date from Key Facts, else first NAV date
    kf = d["key_facts"].set_index("Item")["Value"]
    incep = pd.to_datetime(str(kf.get("Share Class Launch Date", "")).replace("Sept", "Sep"),
                           errors="coerce")
    if pd.isna(incep):
        incep = nav.index[0]

    out = {
        "Fund": ticker,
        "Type": "ETC" if (d.get("asset_class") or "").lower() == "commodity" else "ETF",
        "Years of Data": round(years, 2),
        "Analysis Start Date": nav.index[0].date(),
        "Inception Date": incep.date(),
        "Total Return": total_ret,
        "CAGR": cagr,
        "Ann. SD": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_dd,
        "Days to Recovery": days_rec if rec_date is not None else "Not recovered",
    }
    for y in (1, 3):
        r = rolling_returns(nav, y, ANNUALISE_MULTI_YEAR_ROLLS)
        out[f"{y}Y roll min"] = r.min()
        out[f"{y}Y roll med"] = r.median()
        out[f"{y}Y roll max"] = r.max()
    return out


def build_summary(data: dict) -> pd.DataFrame:
    summary = (
        pd.DataFrame([analyse_fund(t, d) for t, d in data.items()])
        .set_index("Fund")
        .sort_values("CAGR", ascending=False, na_position="last")
    )
    if USE_COMMON_START_DATE:
        # Show the shared analysis start in place of each fund's inception.
        # Years of Data remains and reflects the common analysis window.
        summary = summary.drop(columns=["Inception Date"])
    else:
        # Full-history mode shows each fund's own inception date only.
        summary = summary.drop(columns=["Analysis Start Date"])
    return summary


def build_correlation_matrix(data: dict) -> pd.DataFrame:
    """Return correlations between funds' daily NAV returns."""
    daily_returns = {
        ticker: d["nav"].set_index("Date")["NAV"].sort_index().pct_change()
        for ticker, d in data.items()
    }
    return pd.DataFrame(daily_returns).corr()


if __name__ == "__main__":
    loaded_data = load_all(refresh=REFRESH)
    data, common_start = select_analysis_period(loaded_data, USE_COMMON_START_DATE)
    if USE_COMMON_START_DATE:
        print(f"Using a common analysis period from {common_start.date()}.")
    else:
        print("Using each fund's full available history; correlations use overlapping dates per pair.")
    summary = build_summary(data)

    pct_cols = ["Total Return", "CAGR", "Ann. SD", "Max DD",
                "1Y roll min", "1Y roll med", "1Y roll max",
                "3Y roll min", "3Y roll med", "3Y roll max"]
    show = summary.copy()
    show[pct_cols] = show[pct_cols].apply(lambda s: s.map("{:.2%}".format))
    show[["Sharpe", "Sortino", "Calmar"]] = show[["Sharpe", "Sortino", "Calmar"]].round(2)

    pd.set_option("display.width", 250, "display.max_columns", None)
    print(show)             # one row per fund, one column per metric

    print("\nDaily NAV return correlation matrix:")
    corr = build_correlation_matrix(data)
    corr_display = corr.map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
    for ticker in corr_display.index.intersection(corr_display.columns):
        corr_display.loc[ticker, ticker] = ""
    print(corr_display)
