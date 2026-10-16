from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from fredapi import Fred
from matplotlib.patches import Patch


# =============================================================================
# Configuration
# =============================================================================

SERIES_ID = "A191RL1Q225SBEA"
SERIES_NAME = "US Real GDP Growth"
YIELD_SERIES_ID = "DGS10"
YIELD_SERIES_NAME = "US 10-Year Treasury Yield"

# Set either value to a date such as "2016-01-01" to bound the analysis.
# None uses the full available history on that side of the selected period.
START_DATE = "2015-01-01"  # Example: "2021-01-01"
END_DATE = None  # Example: "2024-12-31"

SLIDE_TITLE = "How Fast Is the US Economy Growing?"
SOURCE_TEXT = "Source: US Bureau of Economic Analysis and Federal Reserve via FRED"
NOTE_TEXT = (
    "Note: GDP is quarter-over-quarter growth in inflation-adjusted output at a "
    "seasonally adjusted annual rate (SAAR), not year-over-year; negative values "
    "indicate contraction. Red shading marks runs of at least two consecutive "
    "negative quarters (a rule of thumb, not official NBER dates). The 10-year "
    "Treasury yield is a daily constant-maturity market yield, not an investment "
    "return. GDP is quarterly and revised; yields are daily. Both share a percent scale."
)

OUTPUT_DIR = Path(__file__).resolve().parent
PNG_FILE = OUTPUT_DIR / "US_Real_GDP_Growth.png"
SAVE_PNG = False

FIG_W_IN = 16
FIG_H_IN = 9
FIG_DPI = 150
CANVAS_W = 1920
CANVAS_H = 1080

LEFT_MARGIN_PX = 90
RIGHT_MARGIN_PX = 1830

BLUE = "#6b8fd1"
ORANGE = "#e8856a"
BG = "#ffffff"
INK = "#000000"
SUBTITLE_INK = "#2d2a2a"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#8c8c8c"
FONT_STACK = (
    "Barclays Effra",
    "Barclays Effra App",
    "Inter",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)

TITLE_SIZE = 26
SUBTITLE_SIZE = 14
FOOT_SIZE = 11
TICK_SIZE = 13

TITLE_BASELINE_PX = 83
SUBTITLE_BASELINES_PX = (124, 155)
SOURCE_BASELINE_PX = 988
NOTE_BASELINES_PX = (1013, 1037)

# =============================================================================
# FRED SSL and API setup
# =============================================================================

API_KEY = os.environ["FRED_API_KEY"]
CA_BUNDLE = certifi.where()
ssl_context = ssl.create_default_context(cafile=CA_BUNDLE)
https_handler = urllib.request.HTTPSHandler(context=ssl_context)
opener = urllib.request.build_opener(https_handler)
urllib.request.install_opener(opener)

fred = Fred(api_key=API_KEY)


# =============================================================================
# Slide coordinate and text helpers
# =============================================================================

def px_x(value_px: float) -> float:
    return value_px / CANVAS_W


def px_y(value_px: float) -> float:
    return 1.0 - value_px / CANVAS_H


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = list(FONT_STACK)

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
fig.patch.set_facecolor(BG)


def get_renderer():
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig._get_renderer()


def measure_width_px(text: str, font_size: float, font_style: str = "normal") -> float:
    probe = fig.text(0, 0, text, fontsize=font_size, style=font_style)
    width = probe.get_window_extent(get_renderer()).width
    probe.remove()
    return width * CANVAS_W / fig.bbox.width


def wrap_to_margins(text: str, font_size: float, max_width_px: float,
                    font_style: str = "normal") -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and measure_width_px(candidate, font_size, font_style) > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# =============================================================================
# Download and prepare GDP growth data
# =============================================================================

print(f"SSL certificate bundle: {CA_BUNDLE}")
print(f"Downloading FRED series {SERIES_ID} ({SERIES_NAME})...")
growth = fred.get_series(SERIES_ID).dropna().sort_index()
growth.index = pd.to_datetime(growth.index)
growth = pd.to_numeric(growth, errors="coerce").dropna()

if growth.empty:
    raise ValueError(f"FRED returned no usable observations for {SERIES_ID}.")

# Find runs of consecutive negative GDP quarters before applying the display dates.
recession_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
negative_run: list[pd.Timestamp] = []
previous_quarter_number: int | None = None

for date, value in growth.items():
    quarter_number = date.to_period("Q").ordinal
    is_consecutive = (
        previous_quarter_number is not None
        and quarter_number == previous_quarter_number + 1
    )

    if value < 0 and (not negative_run or is_consecutive):
        negative_run.append(date)
    else:
        if len(negative_run) >= 2:
            recession_periods.append(
                (negative_run[0], negative_run[-1] + pd.DateOffset(months=3))
            )
        negative_run = [date] if value < 0 else []

    previous_quarter_number = quarter_number

if len(negative_run) >= 2:
    recession_periods.append(
        (negative_run[0], negative_run[-1] + pd.DateOffset(months=3))
    )

print(f"Downloading FRED series {YIELD_SERIES_ID} ({YIELD_SERIES_NAME})...")
treasury_yield = fred.get_series(YIELD_SERIES_ID).dropna().sort_index()
treasury_yield.index = pd.to_datetime(treasury_yield.index)
treasury_yield = pd.to_numeric(treasury_yield, errors="coerce").dropna()

if treasury_yield.empty:
    raise ValueError(f"FRED returned no usable observations for {YIELD_SERIES_ID}.")

available_start = min(growth.index[0], treasury_yield.index[0])
available_end = max(growth.index[-1], treasury_yield.index[-1])
start_date = pd.Timestamp(START_DATE) if START_DATE is not None else available_start
end_date = pd.Timestamp(END_DATE) if END_DATE is not None else available_end

if start_date > end_date:
    raise ValueError("START_DATE must be on or before END_DATE.")

growth = growth.loc[(growth.index >= start_date) & (growth.index <= end_date)]
treasury_yield = treasury_yield.loc[
    (treasury_yield.index >= start_date) & (treasury_yield.index <= end_date)
]

if growth.empty or treasury_yield.empty:
    raise ValueError("One or both FRED series have no observations in the last ten years.")

latest_date = growth.index[-1]
latest_value = float(growth.iloc[-1])
previous_date = growth.index[-2] if len(growth) > 1 else latest_date
previous_value = float(growth.iloc[-2]) if len(growth) > 1 else latest_value
quarter = latest_date.to_period("Q")
previous_quarter = previous_date.to_period("Q")
change_pp = latest_value - previous_value
change_phrase = (
    f"up {change_pp:.1f} percentage points"
    if change_pp >= 0
    else f"down {abs(change_pp):.1f} percentage points"
)
yield_latest_date = treasury_yield.index[-1]
yield_latest_value = float(treasury_yield.iloc[-1])
yield_year_ago = treasury_yield.loc[
    treasury_yield.index <= yield_latest_date - pd.DateOffset(years=1)
]
yield_comparison = float(yield_year_ago.iloc[-1]) if not yield_year_ago.empty else yield_latest_value
yield_change = yield_latest_value - yield_comparison
yield_change_phrase = (
    f"up {yield_change:.1f} percentage points"
    if yield_change >= 0
    else f"down {abs(yield_change):.1f} percentage points"
)
subtitle = (
    f"In {quarter}, US real GDP grew at {latest_value:+.1f}% annualized, "
    f"{change_phrase} from {previous_value:+.1f}% in {previous_quarter}. "
    f"The 10-year Treasury yield was {yield_latest_value:.1f}% on "
    f"{yield_latest_date:%d %b %Y}, {yield_change_phrase} from "
    f"{yield_comparison:.1f}% a year earlier."
)

print(f"Observations: {len(growth)}")
print(f"Latest observation: {latest_date:%Y-%m-%d} ({latest_value:+.2f}%)")
print(f"10-year yield latest: {yield_latest_date:%Y-%m-%d} ({yield_latest_value:.2f}%)")


# =============================================================================
# Title and subtitle
# =============================================================================

fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontweight="bold", color=INK, ha="left", va="baseline",
)

subtitle_font_size = SUBTITLE_SIZE
wrapped_subtitle = wrap_to_margins(
    subtitle, subtitle_font_size, RIGHT_MARGIN_PX - LEFT_MARGIN_PX
)
while len(wrapped_subtitle) > len(SUBTITLE_BASELINES_PX) and subtitle_font_size > 11:
    subtitle_font_size -= 0.5
    wrapped_subtitle = wrap_to_margins(
        subtitle, subtitle_font_size, RIGHT_MARGIN_PX - LEFT_MARGIN_PX
    )
for line, baseline in zip(wrapped_subtitle, SUBTITLE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=subtitle_font_size, color=SUBTITLE_INK, ha="left", va="baseline",
    )


# =============================================================================
# Source and note
# =============================================================================

fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)

note_font_size = FOOT_SIZE
wrapped_note = wrap_to_margins(
    NOTE_TEXT, note_font_size, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
)
while len(wrapped_note) > len(NOTE_BASELINES_PX) and note_font_size > 8:
    note_font_size -= 0.5
    wrapped_note = wrap_to_margins(
        NOTE_TEXT, note_font_size, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
    )
for line, baseline in zip(wrapped_note, NOTE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=note_font_size, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Single chart
# =============================================================================

axes_left, axes_right = 205.0, 1798.0
axes_top, axes_bottom = 220.0, 835.0
ax = fig.add_axes(
    [
        px_x(axes_left),
        px_y(axes_bottom),
        px_x(axes_right - axes_left),
        (axes_bottom - axes_top) / CANVAS_H,
    ],
    zorder=2,
    facecolor="none",
)

for recession_start, recession_end in recession_periods:
    ax.axvspan(
        recession_start,
        recession_end,
        facecolor="#d9534f",
        alpha=0.10,
        edgecolor="none",
        zorder=0.1,
    )

ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
ax.plot(
    growth.index, growth.to_numpy(dtype=float),
    color=BLUE, linewidth=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3,
    label="US Real GDP Growth (annualized, quarterly)",
)
ax.plot(
    treasury_yield.index, treasury_yield.to_numpy(dtype=float),
    color=ORANGE, linewidth=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3,
    label="US 10-Year Treasury Yield (daily)",
)
ax.set_axisbelow(True)
ax.grid(
    True, which="major", axis="both", linestyle="--", linewidth=1, color=GRID, zorder=0,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.spines["bottom"].set_color(AXIS)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

display_end_date = end_date + pd.Timedelta(days=30)
display_years = (display_end_date - start_date).days / 365.25
year_tick_interval = max(1, int(round(display_years / 8)))
ax.xaxis.set_major_locator(
    mdates.YearLocator(year_tick_interval, month=1, day=1)
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(
    axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
    pad=8, colors=INK,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:+.0f}%"))
ax.tick_params(
    axis="y", which="major", length=0, pad=10, labelsize=TICK_SIZE, colors=INK,
)
ax.set_ylabel("Percent", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.set_xlim(start_date, display_end_date)
ax.yaxis.set_label_coords(-0.060, 0.5)
ax.legend(
    handles=[
        *ax.get_legend_handles_labels()[0],
        Patch(facecolor="#d9534f", alpha=0.10, edgecolor="none", label="2+ negative GDP quarters"),
    ],
    labels=[
        *ax.get_legend_handles_labels()[1],
        "2+ negative GDP quarters",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.105),
    ncol=3,
    frameon=False,
    fontsize=11,
    handlelength=2.2,
    columnspacing=1.8,
    handletextpad=0.7,
)


# =============================================================================
# Display / optional save
# =============================================================================

if SAVE_PNG:
    fig.savefig(PNG_FILE, dpi=FIG_DPI, facecolor=BG)
    print(f"Chart saved to:\n{PNG_FILE}")

plt.show()
