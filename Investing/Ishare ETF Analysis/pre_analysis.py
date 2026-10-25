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
REFRESH = False                  # False = use the files already saved, no download

START_DATE = "2021-01-01"   # None or "common" = latest NAV start across funds
                            # "full"            = each fund's own full history
                            # "2021-06-30"      = specific date
END_DATE = None             # None or "full"    = each fund's last NAV
                            # "common"          = earliest last-NAV date across funds
                            # "2025-12-31"      = specific date
# ------------------------------------------


def select_analysis_period(data: dict, start_date=None, end_date=None):
    """Clip each fund's NAV to [start, end]. Returns (data, start, end), where start/end
    are the resolved Timestamps (None = no clipping on that side)."""
    prepared, first_dates, last_dates = {}, [], []
    for ticker, fund in data.items():
        nav = fund["nav"].copy()
        if not nav.empty:
            nav["Date"] = pd.to_datetime(nav["Date"])
            nav = nav.sort_values("Date")
            first_dates.append(nav["Date"].iloc[0])
            last_dates.append(nav["Date"].iloc[-1])
        prepared[ticker] = {**fund, "nav": nav}
    if not first_dates:
        raise ValueError("Cannot select an analysis period: no NAV history was loaded.")

    common_start, common_end = max(first_dates), min(last_dates)

    # ---- resolve start ----
    s = start_date.lower() if isinstance(start_date, str) else start_date
    if s is None or s == "common":
        start = common_start
    elif s == "full":
        start = None
    else:
        start = pd.Timestamp(start_date)
        if start < common_start:
            print(f"Note: START_DATE {start.date()} is before the common start "
                  f"{common_start.date()}; later-launching funds begin at their own first NAV.")

    # ---- resolve end ----
    e = end_date.lower() if isinstance(end_date, str) else end_date
    if e is None or e == "full":
        end = None
    elif e == "common":
        end = common_end
    else:
        end = pd.Timestamp(end_date)

    if start is not None and end is not None and start >= end:
        raise ValueError(f"Start date {start.date()} must be before end date {end.date()}.")

    for ticker, fund in prepared.items():
        nav = fund["nav"]
        if start is not None:
            nav = nav.loc[nav["Date"] >= start]
        if end is not None:
            nav = nav.loc[nav["Date"] <= end]
        if len(nav) < 2:
            raise ValueError(f"{ticker}: fewer than 2 NAV points in the selected period.")
        fund["nav"] = nav.reset_index(drop=True)
    return prepared, start, end


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
        "Analysis End Date": nav.index[-1].date(),
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


def build_summary(data: dict, windowed: bool) -> pd.DataFrame:
    summary = (
        pd.DataFrame([analyse_fund(t, d) for t, d in data.items()])
        .set_index("Fund")
        .sort_values("CAGR", ascending=False, na_position="last")
    )
    if windowed:
        # A start/end window is applied: show analysis start, not inception.
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
    data, start, end = select_analysis_period(loaded_data, START_DATE, END_DATE)

    print(f"Analysis period: {start.date() if start is not None else 'each fund start'} "
          f"to {end.date() if end is not None else 'each fund end'}.")
    if start is None:
        print("Using each fund's full available history; correlations use overlapping dates per pair.")

    summary = build_summary(data, windowed=start is not None)

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