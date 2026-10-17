import json
import ssl
from urllib.request import HTTPSHandler, Request, build_opener

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.patches import Patch

try:
    import yfinance as yf
except ImportError as error:
    raise ImportError(
        "Could not import 'yfinance'. Install or upgrade it with:\n\n"
        "    pip install --upgrade yfinance pandas\n"
    ) from error


# =============================================================================
# Configuration
# =============================================================================

# Public, community-maintained daily archive of NSE cash-market FII/DII data.
# The NSE report page itself exposes current/previous-session reports; this
# archive preserves daily history so monthly, quarterly and annual flows can be
# calculated without keeping a local data file.
FLOW_DATA_URL = (
    "https://raw.githubusercontent.com/MrChartist/fii-dii-data/main/data/history.json"
)

# Choose comparison mode: "mom", "qoq", or "yoy".
# MoM is the default: monthly bars and change versus the previous month.
COMPARISON_MODE = "mom"
START_DATE = "2021-01-01"  # Example: "2024-01-01"; None uses all available history.
END_DATE = None    # Example: "2026-09-30"; None uses the latest available date.

if COMPARISON_MODE not in {"mom", "qoq", "yoy"}:
    raise ValueError('COMPARISON_MODE must be "mom", "qoq", or "yoy".')

FREQUENCY = {"mom": "monthly", "qoq": "quarterly", "yoy": "annual"}[COMPARISON_MODE]
COMPARISON_NAME = {
    "mom": "Month-on-month", "qoq": "Quarter-on-quarter", "yoy": "Year-on-year"
}[COMPARISON_MODE]
RESAMPLE_RULE = {"mom": "ME", "qoq": "QE", "yoy": "YE"}[COMPARISON_MODE]
PERIOD_LABEL = {"mom": "month", "qoq": "quarter", "yoy": "year"}[COMPARISON_MODE]

SLIDE_TITLE = "India's FII and DII Net Market Flows"
SOURCE_TEXT = (
    "Source: NSE FII/FPI & DII cash-market data (NSE, BSE and MSEI); "
    "historical archive via Mr. Chartist; Nifty 500: Yahoo Finance (^CRSLDX)"
)
NOTE_TEXT = (
    "Note: Net flow equals purchases minus sales in the cash market; bars show "
    f"the sum for each {PERIOD_LABEL}. Values are in ₹ crore (cr). "
    "The Nifty 500 line shows monthly closing levels in index points. "
    "Positive values indicate net buying and negative values indicate net selling. "
    "The latest period may be partial."
)

FIG_W_IN, FIG_H_IN, FIG_DPI = 16, 9, 150
CANVAS_W, CANVAS_H = 1920, 1080
LEFT_MARGIN_PX, RIGHT_MARGIN_PX = 90, 1830

GREEN = "#55a868"      # FII/FPI
BLUE = "#6b8fd1"       # DII
YELLOW = "#fcb149"     # Nifty 500 line
BG = "#ffffff"
INK = "#000000"
SUBTITLE_INK = "#2d2a2a"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#616970"
FONT_STACK = (
    "Barclays Effra", "Barclays Effra App", "Inter", "Helvetica", "Arial",
    "Liberation Sans", "DejaVu Sans",
)

TITLE_SIZE, SUBTITLE_SIZE, FOOT_SIZE, TICK_SIZE = 26, 13, 10, 13
TITLE_BASELINE_PX = 83
SUBTITLE_BASELINES_PX = (124, 155)
SOURCE_BASELINE_PX = 988
NOTE_BASELINES_PX = (1013, 1037)


# =============================================================================
# Download and prepare data
# =============================================================================

ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = build_opener(HTTPSHandler(context=ssl_context))
request = Request(FLOW_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
with opener.open(request, timeout=30) as response:
    raw_records = json.loads(response.read().decode("utf-8"))

flows = pd.DataFrame(raw_records)
required_columns = {"date", "fii_net", "dii_net"}
if not required_columns.issubset(flows.columns):
    raise ValueError(
        f"Unexpected FII/DII archive structure. Expected {sorted(required_columns)}; "
        f"found {list(flows.columns)}."
    )

flows["date"] = pd.to_datetime(flows["date"], dayfirst=True, errors="coerce")
for column in ("fii_net", "dii_net"):
    flows[column] = pd.to_numeric(flows[column], errors="coerce")
flows = (
    flows.dropna(subset=["date", "fii_net", "dii_net"])
    .drop_duplicates(subset="date", keep="last")
    .set_index("date")
    .sort_index()
)

if START_DATE is not None:
    flows = flows.loc[flows.index >= pd.Timestamp(START_DATE)]
if END_DATE is not None:
    flows = flows.loc[flows.index <= pd.Timestamp(END_DATE)]
if flows.empty:
    raise ValueError("No FII/DII observations remain in the selected date range.")

# Source values are ₹ crore. Sum daily net flows within each selected period;
# do not fill missing dates with zero bars.
flow_columns = ["fii_net", "dii_net"]
for column in ("fii_buy", "fii_sell", "dii_buy", "dii_sell"):
    if column in flows.columns:
        flows[column] = pd.to_numeric(flows[column], errors="coerce")
        flow_columns.append(column)
flows[["fii_net", "dii_net"]] = flows[["fii_net", "dii_net"]].astype(float)
period_flows = flows[flow_columns].resample(RESAMPLE_RULE).sum(min_count=1).dropna(
    subset=["fii_net", "dii_net"]
)
period_flows = period_flows.loc[period_flows.index <= flows.index.max()]
if period_flows.empty:
    raise ValueError(f"No {FREQUENCY} FII/DII observations remain in the selected range.")

# Fetch daily Nifty 500 closes across the selected flow-history window, then
# retain each month's final close. The flow bars continue to use the selected
# comparison frequency.
NIFTY_500_TICKER = "^CRSLDX"
try:
    nifty_raw = yf.download(
        NIFTY_500_TICKER,
        start=flows.index.min().strftime("%Y-%m-%d"),
        end=(flows.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
except Exception as error:
    raise RuntimeError(f"Could not download Nifty 500 history: {error}") from error

if nifty_raw.empty or "Close" not in nifty_raw.columns.get_level_values(0):
    raise ValueError(
        f"Yahoo Finance returned no Nifty 500 closing prices for {NIFTY_500_TICKER}."
    )
nifty_close = nifty_raw["Close"]
if isinstance(nifty_close, pd.DataFrame):
    nifty_close = nifty_close.iloc[:, 0]
nifty_close = pd.to_numeric(nifty_close, errors="coerce").dropna().sort_index()
nifty_close = nifty_close.loc[
    (nifty_close.index >= flows.index.min()) & (nifty_close.index <= flows.index.max())
]

latest = period_flows.iloc[-1]
latest_period = period_flows.index[-1]
if COMPARISON_MODE == "mom":
    latest_period_label = latest_period.to_period("M").strftime("%b %Y")
elif COMPARISON_MODE == "qoq":
    latest_period_label = f"Q{latest_period.to_period('Q').quarter} {latest_period.year}"
else:
    latest_period_label = str(latest_period.year)
comparison_period = {"mom": "M", "qoq": "Q", "yoy": "Y"}[COMPARISON_MODE]
latest_session_count = int((
    flows.index.to_period(comparison_period)
    == latest_period.to_period(comparison_period)
).sum())

def flow_label(value: float, decimals: int = 0) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}₹{abs(value):,.{decimals}f}"


if {"fii_buy", "fii_sell", "dii_buy", "dii_sell"}.issubset(period_flows.columns):
    combined_net = latest["fii_net"] + latest["dii_net"]
    current_line = (
        f"{latest_period_label} (₹ crore; {latest_session_count} sessions): "
        f"FII bought/sold {latest['fii_buy']:,.0f}/{latest['fii_sell']:,.0f} "
        f"(net {flow_label(latest['fii_net'])}); DII bought/sold "
        f"{latest['dii_buy']:,.0f}/{latest['dii_sell']:,.0f} "
        f"(net {flow_label(latest['dii_net'])}); combined net {flow_label(combined_net)}."
    )
else:
    current_line = (
        f"{latest_period_label}: FII net {flow_label(latest['fii_net'])}; "
        f"DII net {flow_label(latest['dii_net'])}."
    )
if len(period_flows) > 1:
    previous = period_flows.iloc[-2]
    previous_period = period_flows.index[-2]
    if COMPARISON_MODE == "mom":
        previous_label = previous_period.to_period("M").strftime("%b %Y")
    elif COMPARISON_MODE == "qoq":
        previous_label = f"Q{previous_period.to_period('Q').quarter} {previous_period.year}"
    else:
        previous_label = str(previous_period.year)
    fii_delta = latest["fii_net"] - previous["fii_net"]
    dii_delta = latest["dii_net"] - previous["dii_net"]
    previous_combined = previous["fii_net"] + previous["dii_net"]
    latest_combined = latest["fii_net"] + latest["dii_net"]
    combined_delta = latest_combined - previous_combined
    comparison_short = {"mom": "MoM", "qoq": "QoQ", "yoy": "YoY"}[COMPARISON_MODE]
    change_line = (
        f"{comparison_short}\u00a0vs\u00a0{previous_label}: FII {flow_label(previous['fii_net'])} "
        f"to {flow_label(latest['fii_net'])} ({flow_label(fii_delta)}); DII "
        f"{flow_label(previous['dii_net'])} to {flow_label(latest['dii_net'])} "
        f"({flow_label(dii_delta)}); combined {flow_label(previous_combined)} to "
        f"{flow_label(latest_combined)} ({flow_label(combined_delta)})."
    )
else:
    change_line = f"Showing the available {FREQUENCY} observation; no prior period is available for comparison."


# =============================================================================
# Canvas text and slide layout
# =============================================================================

def px_x(value_px: float) -> float:
    return value_px / CANVAS_W


def px_y(value_px: float) -> float:
    return 1.0 - value_px / CANVAS_H


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = list(FONT_STACK)
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
fig.patch.set_facecolor(BG)


def renderer():
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig._get_renderer()


def measure_width_px(text: str, font_size: float, font_style: str = "normal") -> float:
    probe = fig.text(0, 0, text, fontsize=font_size, style=font_style)
    width = probe.get_window_extent(renderer()).width * CANVAS_W / fig.bbox.width
    probe.remove()
    return width


def wrap_to_margins(text: str, font_size: float, max_width_px: float,
                    font_style: str = "normal", balance_two_lines: bool = False) -> list[str]:
    result, current = [], ""
    # Split on regular spaces only; non-breaking spaces keep labels such as
    # "MoM vs Aug 2026" together when the line wraps.
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if current and measure_width_px(candidate, font_size, font_style) > max_width_px:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)

    # For subtitles, choose a natural break that makes both lines use the
    # available width. Greedy wrapping can leave the second line much shorter.
    if balance_two_lines and len(result) == 2:
        words = text.split(" ")
        candidates = []
        for split_at in range(1, len(words)):
            first = " ".join(words[:split_at])
            second = " ".join(words[split_at:])
            first_width = measure_width_px(first, font_size, font_style)
            second_width = measure_width_px(second, font_size, font_style)
            if first_width <= max_width_px and second_width <= max_width_px:
                # Prefer a sentence boundary when line lengths are similarly balanced.
                sentence_break = words[split_at - 1].endswith((".", ";"))
                imbalance = abs(first_width - second_width)
                candidates.append((imbalance + (0 if sentence_break else 180), first, second))
        if candidates:
            _, first, second = min(candidates, key=lambda candidate: candidate[0])
            result = [first, second]
    return result


fig.text(px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
         fontsize=TITLE_SIZE, fontweight="bold", color=INK, ha="left", va="baseline")
subtitle_text = f"{current_line} {change_line}"
subtitle_lines = wrap_to_margins(
    subtitle_text, SUBTITLE_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
    balance_two_lines=True,
)
if len(subtitle_lines) > len(SUBTITLE_BASELINES_PX):
    raise ValueError("The dynamic subtitle exceeds the two-line layout; shorten the selected date range or text.")
for line, baseline in zip(subtitle_lines, SUBTITLE_BASELINES_PX):
    fig.text(px_x(LEFT_MARGIN_PX), px_y(baseline), line,
             fontsize=SUBTITLE_SIZE, color=SUBTITLE_INK, ha="left", va="baseline")

source_artist = fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)
note_lines = wrap_to_margins(
    NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
)
for line, baseline in zip(note_lines, NOTE_BASELINES_PX):
    fig.text(px_x(LEFT_MARGIN_PX), px_y(baseline), line,
             fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline")


# =============================================================================
# Chart
# =============================================================================

axes_left, axes_right = 230.0, 1720.0
axes_top, axes_bottom = 200.0, 860.0
ax = fig.add_axes([
    px_x(axes_left), px_y(axes_bottom), px_x(axes_right - axes_left),
    (axes_bottom - axes_top) / CANVAS_H,
], facecolor="none")

dates = mdates.date2num(period_flows.index.to_pydatetime())
if COMPARISON_MODE == "mom":
    bar_width_days = 18
elif COMPARISON_MODE == "qoq":
    bar_width_days = 55
else:
    bar_width_days = 190
pair_offset_days = bar_width_days * 0.28
single_bar_width = bar_width_days * 0.44

ax.bar(dates - pair_offset_days, period_flows["fii_net"], width=single_bar_width,
       color=GREEN, edgecolor="white", linewidth=0.5, zorder=3)
ax.bar(dates + pair_offset_days, period_flows["dii_net"], width=single_bar_width,
       color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)

nifty_ax = ax.twinx()
nifty_ax.patch.set_visible(False)
nifty_monthly = nifty_close.resample("ME").last().dropna()
nifty_dates = mdates.date2num(nifty_monthly.index.to_pydatetime())
nifty_ax.plot(
    nifty_dates,
    nifty_monthly.to_numpy(),
    color=YELLOW,
    linewidth=2.8,
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    zorder=5,
)

ax.set_axisbelow(True)
ax.grid(True, axis="y", linestyle="--", linewidth=1, color=GRID, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.spines["bottom"].set_color(AXIS)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
nifty_ax.spines["top"].set_visible(False)
nifty_ax.spines["left"].set_visible(False)
nifty_ax.spines["bottom"].set_visible(False)
nifty_ax.spines["right"].set_color(AXIS)
nifty_ax.spines["right"].set_linewidth(1.5)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"₹{value:,.0f}cr"))
ax.set_ylabel("Net cash-market flows (₹ crore)", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.yaxis.set_label_coords(-0.085, 0.5)
nifty_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
nifty_ax.set_ylabel("Nifty 500 (index points)", fontsize=TICK_SIZE, color=INK, labelpad=10)
nifty_ax.tick_params(axis="y", which="major", length=0, pad=8, labelsize=TICK_SIZE, colors=INK)

span_years = max((period_flows.index[-1] - period_flows.index[0]).days / 365.25, 0)
if COMPARISON_MODE == "mom":
    # MoM mode has one bar per month, so label each displayed month.
    monthly_ticks = period_flows.index
    ax.set_xticks(mdates.date2num(monthly_ticks.to_pydatetime()))
    ax.set_xticklabels([
        f"{date:%b}" + (f" '{date:%y}" if i == 0 or date.month == 1 else "")
        for i, date in enumerate(monthly_ticks)
    ])
elif span_years <= 2.25:
    # Bars remain monthly in MoM mode; labels stay at quarter-level or coarser.
    # Label only quarters with at least one displayed bar. Center labels in
    # the quarter to prevent the first month's period-end date hiding Q1.
    quarter_periods = period_flows.index.to_period("Q").unique()
    quarter_ticks = pd.DatetimeIndex([
        period.start_time + pd.Timedelta(days=45) for period in quarter_periods
    ])
    ax.set_xticks(mdates.date2num(quarter_ticks.to_pydatetime()))
    ax.set_xticklabels(
        [f"Q{period.quarter} '{period.year % 100:02d}" for period in quarter_periods]
    )
else:
    year_interval = 1 if span_years <= 8 else 2 if span_years <= 16 else 5 if span_years <= 35 else 10
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
               pad=8, colors=INK)
ax.tick_params(axis="y", which="major", length=0, pad=6, labelsize=TICK_SIZE, colors=INK)

edge_pad = single_bar_width / 2 + pair_offset_days + 8
ax.set_xlim(dates[0] - edge_pad, dates[-1] + edge_pad)
nifty_ax.set_xlim(ax.get_xlim())

handles = [
    Patch(facecolor=GREEN, label="FII/FPI net flow"),
    Patch(facecolor=BLUE, label="DII net flow"),
    plt.Line2D(
        [0], [0], color=YELLOW, linewidth=2.8, marker="o", markersize=5,
        markerfacecolor="white", label="Nifty 500 (monthly close)",
    ),
]
legend = ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, -0.105),
                   ncol=3, frameon=False, fontsize=11, handlelength=1.6,
                   columnspacing=2.0, handletextpad=0.7)

# Place the legend in the actual gap beneath tick labels and above the source.
fig.canvas.draw()
renderer_now = fig.canvas.get_renderer()
visible_ticks = [label for label in ax.get_xticklabels() if label.get_visible() and label.get_text()]
if visible_ticks:
    tick_bottom = min(label.get_window_extent(renderer_now).y0 for label in visible_ticks)
    source_top = source_artist.get_window_extent(renderer_now).y1
    legend.set_bbox_to_anchor(
        (0.5, ((tick_bottom + source_top) / 2) / fig.bbox.height),
        transform=fig.transFigure,
    )

plt.show()
