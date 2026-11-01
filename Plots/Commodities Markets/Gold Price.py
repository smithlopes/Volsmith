from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from institutional_exhibit import Frame, THEME as T

try:
    import yfinance as yf
except ImportError as error:
    raise ImportError(
        "Could not import 'yfinance'. Install it with: pip install yfinance"
    ) from error


# =============================================================================
# Configuration
# =============================================================================

GOLD_TICKER = "GC=F"        # COMEX gold futures, US dollars per troy ounce
GBPUSD_TICKER = "GBPUSD=X"  # US dollars per pound sterling
MOVING_AVERAGE_DAYS = 50

# Set either date to None to leave that end of the date range unrestricted.
START_DATE = "2025-01-01"
END_DATE = None

SLIDE_TITLE = "Gold Prices with 50-Day Moving Average"
SOURCE_TEXT = "Source: Yahoo Finance (COMEX gold futures and GBP/USD)"


# =============================================================================
# Download and prepare data
# =============================================================================

start_date = pd.Timestamp(START_DATE) if START_DATE is not None else None
end_date = pd.Timestamp(END_DATE) if END_DATE is not None else None
if start_date is not None and end_date is not None and start_date > end_date:
    raise ValueError("START_DATE must be on or before END_DATE.")

# Download extra observations before the chart start so the first plotted
# 50-day average still uses a full 50 trading-day window.
download_start = (
    start_date - pd.offsets.BDay(MOVING_AVERAGE_DAYS + 20)
    if start_date is not None else None
)
download_end = end_date + pd.Timedelta(days=1) if end_date is not None else None


def download_close(ticker: str) -> pd.Series:
    request = {
        "auto_adjust": False,
        "actions": False,
    }
    if download_start is None:
        request["period"] = "max"
    else:
        request["start"] = download_start.strftime("%Y-%m-%d")
    if download_end is not None:
        request["end"] = download_end.strftime("%Y-%m-%d")

    history = yf.Ticker(ticker).history(**request)
    if history.empty or "Close" not in history:
        raise ValueError(f"Yahoo Finance returned no closing prices for {ticker}.")
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close[~close.index.duplicated(keep="last")].sort_index()


gold_usd_oz = download_close(GOLD_TICKER).rename("gold_usd_oz")
usd_per_gbp = download_close(GBPUSD_TICKER).rename("usd_per_gbp")

# Forward-fill the latest available FX quote onto gold trading dates, then
# convert USD per troy ounce to GBP per 10 grams of 24K gold.
daily = pd.concat([gold_usd_oz, usd_per_gbp], axis=1, sort=True).sort_index()
daily["usd_per_gbp"] = daily["usd_per_gbp"].ffill()
daily = daily.dropna(subset=["gold_usd_oz", "usd_per_gbp"])
GRAMS_PER_TROY_OUNCE = 31.1034768
daily["gold_gbp_per_10g"] = (
    daily["gold_usd_oz"] / daily["usd_per_gbp"]
    * (10 / GRAMS_PER_TROY_OUNCE)
)
daily["ma_50"] = daily["gold_gbp_per_10g"].rolling(
    MOVING_AVERAGE_DAYS, min_periods=MOVING_AVERAGE_DAYS,
).mean()

if start_date is not None:
    daily = daily.loc[daily.index >= start_date]
if end_date is not None:
    daily = daily.loc[daily.index <= end_date]
if daily.empty:
    raise ValueError("No gold and GBP/USD observations remain in the selected date range.")
if daily["ma_50"].notna().sum() == 0:
    raise ValueError("Not enough observations in the selected date range for a 50-day average.")

latest = daily.iloc[-1]
first = daily.iloc[0]
price_change_pct = (latest["gold_gbp_per_10g"] / first["gold_gbp_per_10g"] - 1) * 100
change_word = "rose" if price_change_pct >= 0 else "fell"
ma_gap_pct = (latest["gold_gbp_per_10g"] / latest["ma_50"] - 1) * 100
ma_position = "above" if ma_gap_pct >= 0 else "below"
period_low = daily["gold_gbp_per_10g"].idxmin()
period_high = daily["gold_gbp_per_10g"].idxmax()
recent_start = daily["gold_gbp_per_10g"].iloc[max(0, len(daily) - 22)]
recent_change_pct = (latest["gold_gbp_per_10g"] / recent_start - 1) * 100
recent_change_word = "rose" if recent_change_pct >= 0 else "fell"
subtitle_text = (
    f"From {daily.index[0]:%d %b %Y} to {daily.index[-1]:%d %b %Y}, the GBP price of "
    f"24K gold {change_word} "
    f"from £{first['gold_gbp_per_10g']:,.0f} to £{latest['gold_gbp_per_10g']:,.0f} "
    f"per 10g ({price_change_pct:+.1f}%). Over the period, gold ranged from "
    f"£{daily['gold_gbp_per_10g'].min():,.0f} on {period_low:%d %b %Y} to "
    f"£{daily['gold_gbp_per_10g'].max():,.0f} on {period_high:%d %b %Y}; its average "
    f"price was £{daily['gold_gbp_per_10g'].mean():,.0f}. Over the latest month, it "
    f"{recent_change_word} {abs(recent_change_pct):.1f}%. The latest 50-day average is "
    f"£{latest['ma_50']:,.0f}, with gold {abs(ma_gap_pct):.1f}% {ma_position} the average."
)
note_text = (
    "Note: COMEX front-month gold futures provide an international 24K gold price proxy, "
    "converted from US dollars per troy ounce to GBP per 10g using daily GBP/USD closes "
    "(10g = 0.3215 troy ounces). UK retail prices may differ because of product type, "
    "dealer premiums, applicable taxes and local costs. The blue line is the "
    "50-trading-day simple moving average. The securities are quoted as an example and not as a recommendation."
)


# =============================================================================
# Styled chart
# =============================================================================

f = Frame(
    width=1920,
    height=1080,
    title=SLIDE_TITLE,
    subtitle=subtitle_text,
    source=SOURCE_TEXT,
    note=note_text,
)
ax = f.add_axes()

ax.plot(
    daily.index, daily["gold_gbp_per_10g"],
    color=T.line, linewidth=T.line_w * 72 / 100,
)
ax.plot(
    daily.index, daily["ma_50"],
    color=T.series[0], linewidth=T.line_w * 72 / 100,
)

price_min = daily["gold_gbp_per_10g"].min()
price_max = daily["gold_gbp_per_10g"].max()
period_low_value = float(price_min)
period_high_value = float(price_max)
ax.axhline(
    period_low_value, color=T.series[2], linewidth=T.line_w * 72 / 100,
    linestyle="--", zorder=2,
)
ax.axhline(
    period_high_value, color="#c96f6f", linewidth=T.line_w * 72 / 100,
    linestyle="--", zorder=2,
)
price_span = price_max - price_min
raw_tick_step = price_span / 6 if price_span > 0 else max(price_max * 0.05, 1)
tick_magnitude = 10 ** np.floor(np.log10(raw_tick_step))
normalized_step = raw_tick_step / tick_magnitude
nice_step = next(
    step for step in (1, 2, 2.5, 5, 10) if step >= normalized_step
)
tick_step = nice_step * tick_magnitude
y_min = tick_step * np.floor(price_min / tick_step)
y_max = tick_step * np.ceil(price_max / tick_step)
if y_min == y_max:
    y_min -= tick_step
    y_max += tick_step
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"£{value:,.0f}"))
f.ylabels(ax, "GBP per 10g")

span_months = max(
    1,
    (daily.index[-1].year - daily.index[0].year) * 12
    + daily.index[-1].month - daily.index[0].month,
)
desired_month_step = int(np.ceil(span_months / 9))
if desired_month_step <= 6:
    month_step = next(step for step in (1, 2, 3, 4, 6) if step >= desired_month_step)
else:
    desired_year_step = int(np.ceil(desired_month_step / 12))
    year_step = next(
        (step for step in (1, 2, 3, 5, 10, 20, 50, 100) if step >= desired_year_step),
        desired_year_step,
    )
    month_step = year_step * 12
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_step))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_xlim(daily.index.min(), daily.index.max())

f.add_legend_item(
    Line2D([0], [0], color=T.line, linewidth=T.line_w * 72 / 100),
    "24K gold per 10g",
)
f.add_legend_item(
    Line2D([0], [0], color=T.series[0], linewidth=T.line_w * 72 / 100),
    "50-day moving average",
)
f.add_legend_item(
    Line2D([0], [0], color=T.series[2], linewidth=T.line_w * 72 / 100, linestyle="--"),
    f"Period low £{period_low_value:,.0f}",
)
f.add_legend_item(
    Line2D([0], [0], color="#c96f6f", linewidth=T.line_w * 72 / 100, linestyle="--"),
    f"Period high £{period_high_value:,.0f}",
)
f.fit()

if hasattr(f, "show"):
    f.show()
else:
    # Compatibility with institutional-exhibit 0.4.x, which predates Frame.show().
    from matplotlib._pylab_helpers import Gcf

    manager = plt._get_backend_mod().new_figure_manager_given_figure(1, f.fig)
    Gcf._set_new_active_manager(manager)
    plt.show()
