"""Analyse the daily return difference between IWMO and SWDA."""
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_loader import load_all


COMPARISON_START = pd.Timestamp("2014-01-01")
TICKERS = ("IWMO", "SWDA")
TAIL_LEVELS = (0.01, 0.05, 0.10)
RF_ANNUAL = 0.04
ANNUALISE_MULTI_YEAR_ROLLS = False


def load_returns() -> pd.DataFrame:
    """Load saved NAV histories and return aligned daily returns."""
    funds = load_all(refresh=False)
    frames = {}

    for ticker in TICKERS:
        if ticker not in funds:
            raise RuntimeError(f"No saved NAV workbook found for {ticker}")
        frame = funds[ticker]["nav"].copy()
        frame["Return"] = frame["NAV"].pct_change()
        frames[ticker] = frame.set_index("Date")["Return"]

    returns = pd.concat(frames, axis=1).dropna()
    returns = returns.loc[returns.index >= COMPARISON_START].copy()
    returns["Difference"] = returns["IWMO"] - returns["SWDA"]
    return returns


def print_statistics(returns: pd.DataFrame) -> None:
    """Print annualised historical relative VaR and distribution statistics."""
    daily_difference = returns["Difference"]
    statistics = {
        "Relative return skewness": f"{daily_difference.skew():.4f}",
        "Relative return excess kurtosis": f"{daily_difference.kurt():.4f}",
    }
    for tail in TAIL_LEVELS:
        daily_relative_var = -daily_difference.quantile(tail)
        yearly_relative_var = daily_relative_var * sqrt(252)
        statistics[f"{tail:.0%} yearly relative VaR"] = f"{yearly_relative_var:.2%}"

    print("\nRelative return distribution and VaR:")
    print(pd.Series(statistics).to_string(header=False))


def print_calendar_year_returns(returns: pd.DataFrame) -> None:
    """Print compounded calendar-year returns since the IWMO comparison start."""
    daily = returns[["IWMO", "SWDA"]].copy()
    daily["60/40 IWMO/SWDA"] = 0.60 * daily["IWMO"] + 0.40 * daily["SWDA"]
    daily["80/20 IWMO/SWDA"] = 0.80 * daily["IWMO"] + 0.20 * daily["SWDA"]

    annual = (1 + daily).groupby(daily.index.year).prod() - 1
    annual.index.name = "Year"
    annual = annual.reset_index()

    current_year = pd.Timestamp.now().year
    first_year = int(annual["Year"].min())
    annual.insert(
        1,
        "Period",
        annual["Year"].map(
            lambda year: (
                "Incomplete inception year / YTD"
                if year == first_year == current_year
                else "Incomplete inception year"
                if year == first_year
                else "YTD"
                if year == current_year
                else "Full year"
            )
        ),
    )

    display = annual.copy()
    return_columns = [column for column in display.columns if column not in ("Year", "Period")]
    for column in return_columns:
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    print("\nCalendar-year returns since IWMO inception:")
    print(display.to_string(index=False))


def rolling_returns(nav: pd.Series, years: int) -> pd.Series:
    """Calculate calendar-based rolling returns using the last available NAV."""
    targets = nav.index - pd.DateOffset(years=years)
    positions = nav.index.searchsorted(targets, side="right") - 1
    valid = targets >= nav.index[0]
    starts = pd.Series(
        np.where(valid, nav.values[np.clip(positions, 0, None)], np.nan),
        index=nav.index,
    )
    result = nav / starts - 1
    if ANNUALISE_MULTI_YEAR_ROLLS and years > 1:
        result = (1 + result) ** (1 / years) - 1
    return result.dropna()


def drawdown_stats(nav: pd.Series) -> tuple[float, float | str]:
    """Return maximum drawdown and calendar days from trough to recovery."""
    peak = nav.cummax()
    drawdown = nav / peak - 1
    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.min()
    peak_level = peak.loc[trough_date]
    recovered = nav.loc[trough_date:][lambda values: values >= peak_level]
    if recovered.empty:
        return max_drawdown, "Not recovered"
    recovery_date = recovered.index[0]
    return max_drawdown, (recovery_date - trough_date).days


def build_performance_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Build pre-analysis-style performance metrics for all four strategies."""
    daily = returns[["IWMO", "SWDA"]].copy()
    daily["60/40 IWMO/SWDA"] = 0.60 * daily["IWMO"] + 0.40 * daily["SWDA"]
    daily["80/20 IWMO/SWDA"] = 0.80 * daily["IWMO"] + 0.20 * daily["SWDA"]
    navs = (1 + daily).cumprod()
    summaries = []

    for strategy, nav in navs.items():
        nav = nav.dropna()
        strategy_returns = nav.pct_change().dropna()
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        periods_per_year = len(strategy_returns) / years
        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        cagr = (1 + total_return) ** (1 / years) - 1
        annualised_sd = strategy_returns.std() * np.sqrt(periods_per_year)

        risk_free_daily = (1 + RF_ANNUAL) ** (1 / periods_per_year) - 1
        excess = strategy_returns - risk_free_daily
        sharpe = excess.mean() * periods_per_year / annualised_sd
        downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(periods_per_year)
        sortino = excess.mean() * periods_per_year / downside

        max_drawdown, days_to_recovery = drawdown_stats(nav)
        calmar = cagr / abs(max_drawdown)
        summary = {
            "Fund": strategy,
            "Years of Data": round(years, 2),
            "Analysis Start Date": nav.index[0].date(),
            "Analysis End Date": nav.index[-1].date(),
            "Total Return": total_return,
            "CAGR": cagr,
            "Ann. SD": annualised_sd,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Max DD": max_drawdown,
            "Days to Recovery": days_to_recovery,
        }
        for window_years in (1, 3):
            rolling = rolling_returns(nav, window_years)
            summary[f"{window_years}Y roll min"] = rolling.min()
            summary[f"{window_years}Y roll med"] = rolling.median()
            summary[f"{window_years}Y roll max"] = rolling.max()
        summaries.append(summary)

    return pd.DataFrame(summaries).set_index("Fund")


def print_performance_summary(returns: pd.DataFrame) -> None:
    """Print the full performance summary in a console-friendly format."""
    summary = build_performance_summary(returns)
    percentage_columns = [
        "Total Return", "CAGR", "Ann. SD", "Max DD",
        "1Y roll min", "1Y roll med", "1Y roll max",
        "3Y roll min", "3Y roll med", "3Y roll max",
    ]
    display = summary.copy()
    display[percentage_columns] = display[percentage_columns].apply(
        lambda column: column.map("{:.2%}".format)
    )
    display[["Sharpe", "Sortino", "Calmar"]] = display[
        ["Sharpe", "Sortino", "Calmar"]
    ].round(2)
    display["Analysis Start Date"] = display["Analysis Start Date"].map(str)
    display["Analysis End Date"] = display["Analysis End Date"].map(str)

    print("\nPerformance summary:")
    print(display.to_string())


def rolling_threshold_counts(
    daily_returns: pd.Series,
    threshold: float = -0.10,
) -> pd.Series:
    """Count threshold breaches in each calendar-based rolling one-year window."""
    daily_returns = daily_returns.dropna().sort_index()
    targets = daily_returns.index - pd.DateOffset(years=1)
    positions = daily_returns.index.searchsorted(targets, side="right") - 1
    counts = []

    for endpoint, start_target, start_position in zip(
        daily_returns.index, targets, positions
    ):
        if start_position < 0 or start_target < daily_returns.index[0]:
            continue
        window = daily_returns.loc[
            (daily_returns.index >= daily_returns.index[start_position])
            & (daily_returns.index <= endpoint)
        ]
        counts.append((endpoint, int((window <= threshold).sum())))

    return pd.Series(dict(counts), dtype="int64", name="Days")


def print_rolling_10pct_frequency(returns: pd.DataFrame) -> None:
    """Print rolling one-year frequencies of daily returns at or below -10%."""
    rows = []
    for ticker in TICKERS:
        counts = rolling_threshold_counts(returns[ticker])
        rows.append(
            {
                "Fund": ticker,
                "Rolling 1Y periods": len(counts),
                "Min days <= -10%": counts.min(),
                "Median days <= -10%": counts.median(),
                "Average days <= -10%": counts.mean(),
                "Max days <= -10%": counts.max(),
                "Periods with >=1 day": f"{(counts > 0).mean():.1%}",
            }
        )

    print("\nDaily returns <= -10% within rolling one-year periods:")
    print(pd.DataFrame(rows).set_index("Fund").to_string())


def rolling_holding_period_returns(nav: pd.Series, months: int = 1) -> pd.Series:
    """Calculate calendar-based holding-period returns using prior available NAVs."""
    nav = nav.dropna().sort_index()
    targets = nav.index - pd.DateOffset(months=months)
    positions = nav.index.searchsorted(targets, side="right") - 1
    valid = targets >= nav.index[0]
    starts = pd.Series(
        np.where(valid, nav.values[np.clip(positions, 0, None)], np.nan),
        index=nav.index,
    )
    return (nav / starts - 1).dropna()


def print_rolling_one_month_returns(returns: pd.DataFrame) -> None:
    """Print one-month holding-period return statistics from the common start."""
    common_navs = (1 + returns[list(TICKERS)]).cumprod()
    rows = []
    for ticker in TICKERS:
        rolling = rolling_holding_period_returns(common_navs[ticker])
        rows.append(
            {
                "Fund": ticker,
                "Common start": common_navs.index[0].date(),
                "End date": common_navs.index[-1].date(),
                "Periods": len(rolling),
                "Periods <= -10%": int((rolling <= -0.10).sum()),
                "% periods <= -10%": (rolling <= -0.10).mean(),
                "Min 1M return": rolling.min(),
                "Median 1M return": rolling.median(),
                "Mean 1M return": rolling.mean(),
                "Max 1M return": rolling.max(),
            }
        )

    display = pd.DataFrame(rows).set_index("Fund")
    percentage_columns = [
        "% periods <= -10%",
        "Min 1M return", "Median 1M return", "Mean 1M return", "Max 1M return",
    ]
    display[percentage_columns] = display[percentage_columns].apply(
        lambda column: column.map("{:.2%}".format)
    )
    display["Common start"] = display["Common start"].map(str)
    display["End date"] = display["End date"].map(str)

    print("\nRolling one-month holding-period returns from the common start:")
    print(display.to_string())


def print_active_analysis(returns: pd.DataFrame) -> None:
    """Print IWMO's active performance versus SWDA and an optimal long-only mix."""
    daily_assets = returns[["IWMO", "SWDA"]]
    active_daily = daily_assets["IWMO"] - daily_assets["SWDA"]
    active_return = active_daily.mean() * 252
    active_risk = active_daily.std() * np.sqrt(252)
    information_ratio = active_return / active_risk

    print("\nActive performance: IWMO versus SWDA benchmark")
    print(pd.Series({
        "Annualised active return": f"{active_return:.2%}",
        "Annualised active risk": f"{active_risk:.2%}",
        "Information ratio": f"{information_ratio:.2f}",
    }).to_string(header=False))

    # Historical tangency portfolio, constrained to long-only IWMO/SWDA weights
    # summing to 100%, with SWDA as the 100% benchmark portfolio.
    annualised_mean = daily_assets.mean() * 252
    annualised_covariance = daily_assets.cov() * 252
    rf = RF_ANNUAL
    candidate_weights = np.linspace(0.0, 1.0, 10_001)
    candidate_portfolio_weights = np.column_stack(
        [candidate_weights, 1.0 - candidate_weights]
    )
    candidate_returns = candidate_portfolio_weights @ annualised_mean.to_numpy()
    candidate_variances = np.einsum(
        "ij,jk,ik->i",
        candidate_portfolio_weights,
        annualised_covariance.to_numpy(),
        candidate_portfolio_weights,
    )
    candidate_volatility = np.sqrt(candidate_variances)
    candidate_sharpe = (candidate_returns - rf) / candidate_volatility
    optimal_position = int(np.nanargmax(candidate_sharpe))
    optimal_weights = candidate_portfolio_weights[optimal_position]

    optimal_portfolio_daily = daily_assets.to_numpy() @ optimal_weights
    optimal_active_daily = optimal_portfolio_daily - daily_assets["SWDA"].to_numpy()
    optimal_active_return = optimal_active_daily.mean() * 252
    optimal_active_risk = optimal_active_daily.std(ddof=1) * np.sqrt(252)
    optimal_information_ratio = optimal_active_return / optimal_active_risk

    weights = pd.DataFrame(
        {
            "Portfolio weight": optimal_weights,
            "Active weight vs SWDA": [optimal_weights[0], optimal_weights[1] - 1],
        },
        index=["IWMO", "SWDA"],
    )
    weights = weights.apply(lambda column: column.map("{:.2%}".format))
    print("\nOptimal long-only portfolio (RF = 4.00%; weights sum to 100%):")
    print(weights.to_string())
    print(pd.Series({
        "Optimal active risk target": f"{optimal_active_risk:.2%}",
        "Optimal active return": f"{optimal_active_return:.2%}",
        "Optimal information ratio": f"{optimal_information_ratio:.2f}",
    }).to_string(header=False))


def plot_rolling_information_ratio(returns: pd.DataFrame) -> None:
    """Plot the one-year rolling Information Ratio of IWMO versus SWDA."""
    active_daily = returns["IWMO"] - returns["SWDA"]
    targets = active_daily.index - pd.DateOffset(years=1)
    positions = active_daily.index.searchsorted(targets, side="right") - 1
    ratios = []

    for endpoint_position, (endpoint, start_target, start_position) in enumerate(
        zip(active_daily.index, targets, positions)
    ):
        if start_position < 0 or start_target < active_daily.index[0]:
            continue
        window = active_daily.iloc[start_position:endpoint_position + 1]
        active_risk = window.std(ddof=1) * np.sqrt(252)
        if active_risk == 0 or pd.isna(active_risk):
            continue
        active_return = window.mean() * 252
        ratios.append((endpoint, active_return / active_risk))

    rolling_information_ratio = pd.Series(
        dict(ratios), name="Information Ratio"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        rolling_information_ratio.index,
        rolling_information_ratio,
        color="tab:purple",
        linewidth=1.0,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Rolling one-year Information Ratio: IWMO vs SWDA")
    ax.set_xlabel("Window end date")
    ax.set_ylabel("Information Ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()


def plot_returns(returns: pd.DataFrame) -> None:
    """Plot the return-difference time series and its distribution."""
    difference = returns["Difference"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns.index, difference * 100, color="tab:blue", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Daily return difference: IWMO − SWDA")
    ax.set_xlabel(
        f"Comparison from {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )
    ax.set_ylabel("Difference in daily return (percentage points)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    histogram_fig, histogram_ax = plt.subplots(figsize=(10, 6))
    daily_difference = difference * 100
    histogram_ax.hist(
        daily_difference,
        bins=80,
        color="tab:blue",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        density=True,
    )
    daily_difference.plot.kde(
        ax=histogram_ax,
        color="tab:orange",
        linewidth=2,
        label="KDE",
    )
    for tail in TAIL_LEVELS:
        threshold = daily_difference.quantile(tail)
        histogram_ax.axvline(
            threshold,
            linewidth=1.2,
            linestyle="--",
            label=f"{tail:.0%} tail: {threshold:.2f}%",
        )
    histogram_ax.axvline(0, color="black", linewidth=0.8)
    histogram_ax.set_title("Distribution of daily return differences: IWMO − SWDA")
    histogram_ax.set_xlabel("Daily return difference (percentage points)")
    histogram_ax.set_ylabel("Density")
    histogram_ax.legend()
    histogram_ax.grid(axis="y", alpha=0.25)
    histogram_fig.tight_layout()

    plt.show()


def main() -> None:
    returns = load_returns()
    print_calendar_year_returns(returns)
    print_performance_summary(returns)
    print_rolling_10pct_frequency(returns)
    print_rolling_one_month_returns(returns)
    print_active_analysis(returns)
    print_statistics(returns)
    plot_rolling_information_ratio(returns)
    plot_returns(returns)


if __name__ == "__main__":
    main()
