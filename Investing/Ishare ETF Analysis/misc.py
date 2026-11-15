"""Analyse the daily return difference between IWMO and SWDA."""
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd

from etf_loader import load_all


COMPARISON_START = pd.Timestamp("2014-01-01")
TICKERS = ("IWMO", "SWDA")
TAIL_LEVELS = (0.01, 0.05, 0.10)


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
        print(f"{ticker}: loaded {len(frame):,} daily observations")

    returns = pd.concat(frames, axis=1).dropna()
    returns = returns.loc[returns.index >= COMPARISON_START].copy()
    returns["Difference"] = returns["IWMO"] - returns["SWDA"]
    return returns


def print_statistics(returns: pd.DataFrame) -> None:
    """Print annualised historical relative VaR and distribution statistics."""
    daily_difference = returns["Difference"]
    print(f"Relative return skewness: {daily_difference.skew():.4f}")
    print(f"Relative return excess kurtosis: {daily_difference.kurt():.4f}")

    for tail in TAIL_LEVELS:
        daily_relative_var = -daily_difference.quantile(tail)
        yearly_relative_var = daily_relative_var * sqrt(252)
        print(
            f"{tail:.0%} yearly relative VaR (IWMO − SWDA): "
            f"{yearly_relative_var:.2%}"
        )


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
    print_statistics(returns)
    plot_returns(returns)


if __name__ == "__main__":
    main()
