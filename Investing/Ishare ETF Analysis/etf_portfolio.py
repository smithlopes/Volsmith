"""ETF portfolio mix, historical risk/performance, and look-through holdings.

Edit ASSET_CLASS_WEIGHTS and FUND_WEIGHTS_WITHIN_CLASS below, then run this
file. Both tiers must sum to 1.0. Holdings are aggregated across funds, so an
issuer held by multiple funds appears once with its total portfolio weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_loader import load_all


# Tier 1: allocation across asset classes (must sum to 100%).
ASSET_CLASS_WEIGHTS = {
    "Equity": 0.70,
    "Commodity": 0.10,
    "Fixed Income": 0.20,
}

# Tier 2: fund allocation within each asset class (each class must sum to 100%).
# Example: SWDA's total portfolio weight is 80% * 50% = 40%.
FUND_WEIGHTS_WITHIN_CLASS = {
    "Equity": {
        "SWDA": 0.50,
        "IWMO": 0.50,
    },
    "Commodity": {
        "SGLN": 1.00,
    },
    # These are user-defined model sleeves, not tickers downloaded by etf_loader.
    "Fixed Income": {
        "FI_BOND": 1.0,
        "FI_CASH": 0.00,
    },
}

# Fixed-income simulation assumptions (decimal form). Set the expected annual
# return and annual volatility for each modeled sleeve. The return path is
# simulated daily with a lognormal model and a fixed seed, so it is repeatable.
# This is a simple scenario model, not a bond-pricing or duration model.
FI_ASSUMED_ANNUAL_RETURNS = {
    "FI_BOND": 0.045,
    "FI_CASH": 0.035,
}
FI_ASSUMED_ANNUAL_VOLATILITIES = {
    "FI_BOND": 0.060,
    "FI_CASH": 0.005,
}
FI_RANDOM_SEED = 20261010

RISK_FREE_RATE_ANNUAL = 0.04
REFRESH = True  # Refreshes current holdings; loader falls back to saved files if download fails.
TOP_HOLDINGS = 10
TRADING_DAYS_PER_YEAR = 252


def build_portfolio_weights(data: dict) -> tuple[dict[str, float], pd.DataFrame]:
    """Combine asset-class and within-class weights into total fund weights."""
    if not ASSET_CLASS_WEIGHTS:
        raise ValueError("ASSET_CLASS_WEIGHTS is empty.")
    if any(not np.isfinite(w) or w < 0 for w in ASSET_CLASS_WEIGHTS.values()):
        raise ValueError("Asset-class weights must be finite, non-negative numbers.")
    if not np.isclose(sum(ASSET_CLASS_WEIGHTS.values()), 1.0, atol=1e-6):
        raise ValueError("ASSET_CLASS_WEIGHTS must sum to 1.0 (100%).")

    weights: dict[str, float] = {}
    rows = []
    for asset_class, class_weight in ASSET_CLASS_WEIGHTS.items():
        class_funds = FUND_WEIGHTS_WITHIN_CLASS.get(asset_class, {})
        if not class_funds:
            raise ValueError(f"Add fund weights for asset class {asset_class!r}.")
        if any(not np.isfinite(w) or w < 0 for w in class_funds.values()):
            raise ValueError(f"Fund weights within {asset_class} must be finite and non-negative.")
        if not np.isclose(sum(class_funds.values()), 1.0, atol=1e-6):
            raise ValueError(f"Fund weights within {asset_class} must sum to 1.0 (100%).")

        for ticker, within_class_weight in class_funds.items():
            is_fi_model = asset_class == "Fixed Income" and ticker in FI_ASSUMED_ANNUAL_RETURNS
            if ticker not in data and not is_fi_model:
                raise ValueError(f"No loaded data for {ticker}; check the fund ticker.")
            total_weight = class_weight * within_class_weight
            weights[ticker] = total_weight
            rows.append({
                "Asset Class": asset_class,
                "Class Weight": class_weight,
                "Fund": ticker,
                "Weight within Class": within_class_weight,
                "Portfolio Weight": total_weight,
            })

    if not np.isclose(sum(weights.values()), 1.0, atol=1e-6):
        raise ValueError("Combined fund weights do not sum to 1.0.")
    return weights, pd.DataFrame(rows)


def fund_returns(data: dict, tickers: list[str]) -> pd.DataFrame:
    series = {}
    model_tickers = [ticker for ticker in tickers if ticker in FI_ASSUMED_ANNUAL_RETURNS]
    market_calendar = None
    if model_tickers:
        calendars = [
            pd.to_datetime(data[ticker]["nav"]["Date"])
            for ticker in data
            if not data[ticker]["nav"].empty
        ]
        if not calendars:
            raise ValueError("Cannot create fixed-income return series without any fund NAV dates.")
        market_calendar = pd.DatetimeIndex(
            sorted(set().union(*(set(index) for index in calendars)))
        )

    for ticker in tickers:
        if ticker in FI_ASSUMED_ANNUAL_RETURNS:
            annual_return = FI_ASSUMED_ANNUAL_RETURNS[ticker]
            annual_volatility = FI_ASSUMED_ANNUAL_VOLATILITIES.get(ticker)
            if not np.isfinite(annual_return) or annual_return <= -1:
                raise ValueError(f"Invalid assumed annual return for {ticker}: {annual_return}")
            if annual_volatility is None or not np.isfinite(annual_volatility) or annual_volatility < 0:
                raise ValueError(f"Invalid or missing annual volatility assumption for {ticker}.")
            rng_seed = FI_RANDOM_SEED + sum(ord(char) for char in ticker)
            rng = np.random.default_rng(rng_seed)
            daily_volatility = annual_volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
            daily_log_drift = (
                np.log1p(annual_return) - 0.5 * annual_volatility**2
            ) / TRADING_DAYS_PER_YEAR
            daily_log_returns = rng.normal(
                loc=daily_log_drift,
                scale=daily_volatility,
                size=len(market_calendar),
            )
            simulated_returns = np.expm1(daily_log_returns)
            series[ticker] = pd.Series(simulated_returns, index=market_calendar, name=ticker)
            continue
        nav_frame = data[ticker]["nav"]
        if nav_frame.empty:
            raise ValueError(f"No NAV history found for {ticker}.")
        nav = nav_frame.set_index("Date")["NAV"].sort_index()
        nav = nav[~nav.index.duplicated(keep="last")]
        series[ticker] = nav.pct_change()
    # Require a common date range with observations for every selected fund.
    return pd.DataFrame(series).dropna(how="any")


def drawdown_and_recovery(nav: pd.Series) -> tuple[float, float | str]:
    drawdown = nav / nav.cummax() - 1
    trough_date = drawdown.idxmin()
    peak_at_trough = nav.cummax().loc[trough_date]
    recovered = nav.loc[trough_date:]
    recovered = recovered[recovered >= peak_at_trough]
    days = (recovered.index[0] - trough_date).days if not recovered.empty else "Not recovered"
    return float(drawdown.min()), days


def performance_row(name: str, returns: pd.Series, weight: float | None = None) -> dict:
    returns = returns.dropna()
    if returns.empty:
        raise ValueError(f"Not enough overlapping return history for {name}.")
    nav = (1 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    rf_daily = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = returns - rf_daily
    sharpe = float(excess.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if returns.std(ddof=1) > 0 else np.nan
    max_dd, recovery = drawdown_and_recovery(nav)
    row = {
        "Asset": name,
        "Weight": weight,
        "Start Date": returns.index[0].date(),
        "End Date": returns.index[-1].date(),
        "CAGR": cagr,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Days to Recovery": recovery,
        "Total Return": float(nav.iloc[-1] - 1),
    }
    return row


def build_performance_table(data: dict, weights: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = fund_returns(data, list(weights))
    portfolio_returns = returns.mul(pd.Series(weights), axis="columns").sum(axis=1)
    rows = [performance_row(ticker, returns[ticker], weights[ticker]) for ticker in weights]
    rows.append(performance_row("PORTFOLIO", portfolio_returns, 1.0))
    return pd.DataFrame(rows).set_index("Asset"), returns


def build_lookthrough_holdings(data: dict, weights: dict[str, float]) -> pd.DataFrame:
    frames = []
    no_holdings = []
    for ticker, fund_weight in weights.items():
        # FI sleeves are modeled return streams and have no underlying holdings
        # data in the ETF loader.
        if ticker not in data:
            no_holdings.append(ticker)
            continue
        holdings = data[ticker].get("holdings", pd.DataFrame()).copy()
        if holdings.empty or "Weight (%)" not in holdings:
            no_holdings.append(ticker)
            continue
        holdings["Weight (%)"] = pd.to_numeric(holdings["Weight (%)"], errors="coerce")
        holdings = holdings.dropna(subset=["Weight (%)"])
        if holdings.empty:
            no_holdings.append(ticker)
            continue
        holdings["Portfolio Weight (%)"] = holdings["Weight (%)"] * fund_weight / 100
        holdings["Fund"] = ticker
        # Ticker is the preferred key; fall back to holding name when absent.
        ticker_col = holdings.get("Issuer Ticker", pd.Series(index=holdings.index, dtype=object))
        name_col = holdings.get("Name", pd.Series(index=holdings.index, dtype=object))
        holdings["_key"] = ticker_col.fillna("").astype(str).str.strip()
        missing_key = holdings["_key"].eq("")
        holdings.loc[missing_key, "_key"] = name_col.fillna("Unknown holding").astype(str).str.strip()
        holdings["_name"] = name_col.fillna(holdings["_key"])
        frames.append(holdings)

    if not frames:
        return pd.DataFrame(columns=["Holding", "Ticker", "Portfolio Weight (%)", "Funds", "As Of"])

    all_holdings = pd.concat(frames, ignore_index=True)
    grouped = all_holdings.groupby("_key", dropna=False).agg(
        Holding=("_name", "first"),
        **{"Portfolio Weight (%)": ("Portfolio Weight (%)", "sum")},
        Funds=("Fund", lambda values: ", ".join(sorted(set(values)))),
        **{"As Of": ("As Of", "max")},
    ).reset_index()
    ticker_lookup = all_holdings.groupby("_key")["Issuer Ticker"].first()
    grouped["Ticker"] = grouped["_key"].map(ticker_lookup)
    grouped = grouped.sort_values("Portfolio Weight (%)", ascending=False).head(TOP_HOLDINGS)
    grouped = grouped[["Holding", "Ticker", "Portfolio Weight (%)", "Funds", "As Of"]]
    grouped.attrs["no_holdings_funds"] = no_holdings
    return grouped.reset_index(drop=True)


def display_performance(table: pd.DataFrame) -> None:
    display = table.copy()
    for column in ("Weight", "CAGR", "Ann. Volatility", "Max Drawdown", "Total Return"):
        display[column] = display[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "N/A")
    display["Sharpe"] = display["Sharpe"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "N/A")
    print("\nPortfolio and fund performance:")
    print(display.to_string())


def display_construction(construction: pd.DataFrame) -> None:
    shown = construction.copy()
    for column in ("Class Weight", "Weight within Class", "Portfolio Weight"):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}")
    print("\nPortfolio construction:")
    print(shown.to_string(index=False))


def display_holdings(holdings: pd.DataFrame) -> None:
    print(f"\nTop {TOP_HOLDINGS} look-through holdings by portfolio weight:")
    if holdings.empty:
        print("No holdings data found for the selected funds.")
        return
    shown = holdings.copy()
    shown["Portfolio Weight (%)"] = shown["Portfolio Weight (%)"].map(lambda value: f"{value:.2f}%")
    shown["As Of"] = pd.to_datetime(shown["As Of"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(shown.to_string(index=False))
    unavailable = holdings.attrs.get("no_holdings_funds", [])
    if unavailable:
        print("No look-through holdings available for: " + ", ".join(unavailable))


def main() -> None:
    data = load_all(refresh=REFRESH)
    weights, construction = build_portfolio_weights(data)

    display_construction(construction)
    print("\nFixed-income simulation assumptions:")
    if FI_ASSUMED_ANNUAL_RETURNS:
        for ticker, annual_return in FI_ASSUMED_ANNUAL_RETURNS.items():
            annual_volatility = FI_ASSUMED_ANNUAL_VOLATILITIES.get(ticker, np.nan)
            print(
                f"  {ticker}: expected return {annual_return:.2%} p.a.; "
                f"volatility {annual_volatility:.2%} p.a."
            )
    else:
        print("  None configured")
    performance, returns = build_performance_table(data, weights)
    display_performance(performance)

    print("\nFund daily-return correlation matrix (common dates):")
    correlations = returns.corr().map(lambda value: f"{value:.2f}" if pd.notna(value) else "N/A")
    print(correlations.to_string())

    holdings = build_lookthrough_holdings(data, weights)
    display_holdings(holdings)


if __name__ == "__main__":
    main()
