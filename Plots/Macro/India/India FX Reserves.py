from io import BytesIO
import ssl
from urllib.request import HTTPSHandler, Request, build_opener

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.patches import Patch


# =============================================================================
# Configuration
# =============================================================================

RBI_RESERVES_CSV_URL = (
    "https://data-api.dbie.rbihub.in/api/tables/"
    "external_sector/r53_foreign_exchange_reserves_weekly/csv"
)
FRED_USDINR_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"

# Set either to a date like "2015-01-01"; None uses all available data.
START_DATE = "2023-01-01"
END_DATE = None
COMPARISON_MODE = "qoq"  # Choose "yoy" or "qoq"; controls bar and subtitle frequency.

if COMPARISON_MODE not in {"yoy", "qoq"}:
    raise ValueError('COMPARISON_MODE must be either "yoy" or "qoq".')

PERIOD_NAME = "annual" if COMPARISON_MODE == "yoy" else "quarterly"
COMPARISON_NAME = "year-on-year" if COMPARISON_MODE == "yoy" else "quarter-on-quarter"

SLIDE_TITLE = "India's Foreign Exchange Reserves Composition"
SOURCE_TEXT = (
    "Source: Reserve Bank of India via DBIE; USD/INR: Federal Reserve via FRED"
)
NOTE_TEXT = (
    "Note: Stacked bars show foreign currency assets, gold, SDRs and India's reserve "
    f"tranche position in the IMF (RTP), in US$ billions. {PERIOD_NAME.title()} bars use each "
    f"period's final available weekly observation (the latest period may be partial); "
    f"USD/INR is the {PERIOD_NAME} average. A higher USD/INR rate means a weaker rupee."
)

FIG_W_IN = 16
FIG_H_IN = 9
FIG_DPI = 150
CANVAS_W = 1920
CANVAS_H = 1080

LEFT_MARGIN_PX = 90
RIGHT_MARGIN_PX = 1830

BLUE = "#6b8fd1"
ORANGE = "#e8856a"
YELLOW = "#fcb149"
GREEN = "#55a868"
PURPLE = "#9b8ac4"
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

TITLE_SIZE = 26
SUBTITLE_SIZE = 13
FOOT_SIZE = 10
TICK_SIZE = 13

TITLE_BASELINE_PX = 83
SUBTITLE_BASELINES_PX = (124, 155)
SOURCE_BASELINE_PX = 988
NOTE_BASELINES_PX = (1013, 1037)


# =============================================================================
# Download and prepare data
# =============================================================================

ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = build_opener(HTTPSHandler(context=ssl_context))


def download_csv(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(request, timeout=30) as response:
        return pd.read_csv(BytesIO(response.read()))


def to_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


reserves_raw = download_csv(RBI_RESERVES_CSV_URL)
required_rbi_columns = {"row_no", "c1", "c3", "c5", "c7", "c11", "c13"}
if not required_rbi_columns.issubset(reserves_raw.columns):
    raise ValueError(f"Unexpected RBI DBIE CSV columns: {list(reserves_raw.columns)}")

# DBIE exports this bulletin table as a report layout: c1 is the date,
# c5 is foreign currency assets in US$ millions, c7 is gold, c11 is SDRs,
# and c13 is the IMF reserve tranche position (RTP), all in US$ millions.
# Header and financial-year rows are discarded when their dates fail to parse.
reserves = pd.DataFrame({
    "date": pd.to_datetime(reserves_raw["c1"], format="%d-%b-%Y", errors="coerce"),
    "total_reserves_usd_bn": to_number(reserves_raw["c3"]) / 1_000,
    "fx_reserves_usd_bn": to_number(reserves_raw["c5"]) / 1_000,
    "gold_reserves_usd_bn": to_number(reserves_raw["c7"]) / 1_000,
    "sdr_reserves_usd_bn": to_number(reserves_raw["c11"]) / 1_000,
    "rtp_reserves_usd_bn": to_number(reserves_raw["c13"]) / 1_000,
}).dropna()

if reserves.empty:
    raise ValueError("RBI DBIE returned no usable foreign currency or gold observations.")

reserves = reserves.set_index("date").sort_index()

fx_raw = download_csv(FRED_USDINR_CSV_URL)
if not {"observation_date", "DEXINUS"}.issubset(fx_raw.columns):
    raise ValueError(f"Unexpected FRED CSV columns: {list(fx_raw.columns)}")

usd_inr = pd.Series(
    to_number(fx_raw["DEXINUS"]).to_numpy(),
    index=pd.to_datetime(fx_raw["observation_date"], errors="coerce"),
    name="usd_inr",
).dropna().sort_index()

if START_DATE is not None:
    start_date = pd.Timestamp(START_DATE)
    reserves = reserves.loc[reserves.index >= start_date]
    usd_inr = usd_inr.loc[usd_inr.index >= start_date]

if END_DATE is not None:
    end_date = pd.Timestamp(END_DATE)
    reserves = reserves.loc[reserves.index <= end_date]
    usd_inr = usd_inr.loc[usd_inr.index <= end_date]

if reserves.empty or usd_inr.empty:
    raise ValueError("No observations remain in the selected date range.")

# Use final weekly reserve stocks and average exchange rates for each period.
resample_rule = "YE" if COMPARISON_MODE == "yoy" else "QE"
period_reserves = reserves.resample(resample_rule).last()
period_usd_inr = usd_inr.resample(resample_rule).mean()

chart_data = period_reserves.join(period_usd_inr, how="inner").dropna()
if chart_data.empty:
    raise ValueError(f"No overlapping {PERIOD_NAME} reserves and USD/INR observations.")

latest = chart_data.iloc[-1]
latest_year = chart_data.index[-1].year
latest_period = chart_data.index[-1].to_period("Q")
if COMPARISON_MODE == "yoy":
    latest_date_mask = reserves.index.year == latest_year
else:
    latest_date_mask = reserves.index.to_period("Q") == latest_period
latest_observation_date = reserves.loc[latest_date_mask].index.max()


def reserve_share(component: str) -> float:
    return latest[component] / latest["total_reserves_usd_bn"] * 100


def period_label(date: pd.Timestamp) -> str:
    if COMPARISON_MODE == "yoy":
        return str(date.year)
    period = date.to_period("Q")
    return f"Q{period.quarter} {period.year}"


first = chart_data.iloc[0]
first_period = chart_data.index[0]
last_period = chart_data.index[-1]
range_total_change = latest["total_reserves_usd_bn"] - first["total_reserves_usd_bn"]
range_fca_change = latest["fx_reserves_usd_bn"] - first["fx_reserves_usd_bn"]
range_gold_change = latest["gold_reserves_usd_bn"] - first["gold_reserves_usd_bn"]
range_usd_inr_change_pct = (latest["usd_inr"] / first["usd_inr"] - 1) * 100
range_reserve_verb = "rose" if range_total_change >= 0 else "fell"
range_usd_inr_verb = "climbed" if range_usd_inr_change_pct >= 0 else "eased"


def signed_usd_bn(value: float, decimals: int = 0) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}US\\${abs(value):,.{decimals}f}bn"


range_line = (
    f"From {period_label(first_period)} to {period_label(last_period)}, reserves "
    f"{range_reserve_verb} US\\${abs(range_total_change):,.0f}bn to "
    f"US\\${latest['total_reserves_usd_bn']:,.0f}bn (FCA "
    f"{signed_usd_bn(range_fca_change)}, gold {signed_usd_bn(range_gold_change)}); "
    f"USD/INR {range_usd_inr_verb} "
    f"{abs(range_usd_inr_change_pct):.1f}%, signalling rupee weakness."
)

if len(chart_data) > 1:
    previous = chart_data.iloc[-2]
    total_change = latest["total_reserves_usd_bn"] - previous["total_reserves_usd_bn"]
    fx_change = latest["fx_reserves_usd_bn"] - previous["fx_reserves_usd_bn"]
    gold_change = latest["gold_reserves_usd_bn"] - previous["gold_reserves_usd_bn"]
    latest_reserve_verb = "rose" if total_change >= 0 else "fell"
    def signed_component_change(value: float) -> str:
        return signed_usd_bn(value, decimals=1)

    latest_line = (
        f"In {period_label(last_period)}, reserves {latest_reserve_verb} "
        f"US\\${abs(total_change):,.0f}bn {COMPARISON_NAME}, led by FCA "
        f"({signed_component_change(fx_change)}) and gold "
        f"({signed_component_change(gold_change)}). On {latest_observation_date:%d %b}, "
        f"FCA made up {reserve_share('fx_reserves_usd_bn'):.1f}% of reserves, gold "
        f"{reserve_share('gold_reserves_usd_bn'):.1f}%, SDRs "
        f"{reserve_share('sdr_reserves_usd_bn'):.1f}% and RTP "
        f"{reserve_share('rtp_reserves_usd_bn'):.1f}%."
    )
else:
    latest_line = (
        f"As of {latest_observation_date:%d %b %Y}, total reserves were "
        f"US\\${latest['total_reserves_usd_bn']:,.0f}bn, with FCA at "
        f"{reserve_share('fx_reserves_usd_bn'):.1f}% and gold at "
        f"{reserve_share('gold_reserves_usd_bn'):.1f}%."
    )
subtitle_text = f"{range_line} {latest_line}"


# =============================================================================
# Slide coordinates and text
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


def wrap_to_margins(
    text: str, font_size: float, max_width_px: float, font_style: str = "normal"
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and measure_width_px(candidate, font_size, font_style) > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontweight="bold", color=INK, ha="left", va="baseline",
)
subtitle_lines = wrap_to_margins(
    subtitle_text, SUBTITLE_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
)
if len(subtitle_lines) > len(SUBTITLE_BASELINES_PX):
    raise ValueError("The dynamic subtitle exceeds the two-line layout; shorten the commentary.")
for line, baseline in zip(subtitle_lines, SUBTITLE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=SUBTITLE_SIZE, color=SUBTITLE_INK, ha="left", va="baseline",
    )
source_artist = fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)
for line, baseline in zip(
    wrap_to_margins(NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"),
    NOTE_BASELINES_PX,
):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Chart
# =============================================================================

axes_left, axes_right = 230.0, 1720.0
axes_top, axes_bottom = 200.0, 860.0
ax = fig.add_axes(
    [
        px_x(axes_left), px_y(axes_bottom), px_x(axes_right - axes_left),
        (axes_bottom - axes_top) / CANVAS_H,
    ],
    facecolor="none",
)

dates = mdates.date2num(chart_data.index.to_pydatetime())
bar_width_days = 145 if COMPARISON_MODE == "yoy" else 34
bar_bottom = pd.Series(0.0, index=chart_data.index)
reserve_components = (
    ("fx_reserves_usd_bn", BLUE),
    ("gold_reserves_usd_bn", ORANGE),
    ("sdr_reserves_usd_bn", GREEN),
    ("rtp_reserves_usd_bn", PURPLE),
)
for component, color in reserve_components:
    ax.bar(
        dates,
        chart_data[component],
        width=bar_width_days,
        bottom=bar_bottom,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    bar_bottom += chart_data[component]

fx_ax = ax.twinx()
fx_ax.patch.set_visible(False)
fx_ax.plot(
    dates,
    chart_data["usd_inr"],
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

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
    fx_ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(AXIS)
    ax.spines[spine].set_linewidth(1.5)
fx_ax.spines["left"].set_visible(False)
fx_ax.spines["right"].set_visible(True)
fx_ax.spines["right"].set_color(AXIS)
fx_ax.spines["right"].set_linewidth(1.5)

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda value, _: f"US\\${value:,.0f}bn")
)
fx_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"₹{value:,.0f}"))
ax.set_ylabel("Reserves (US$ billions)", fontsize=TICK_SIZE, color=INK, labelpad=10)
fx_ax.set_ylabel("Indian rupees per US dollar", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.yaxis.set_label_coords(-0.090, 0.5)

chart_span_years = (chart_data.index[-1] - chart_data.index[0]).days / 365.25
if COMPARISON_MODE == "qoq" and chart_span_years <= 2.5:
    # For short histories, retain quarter-level detail as the finest interval.
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: (
                f"Q{((date := mdates.num2date(value)).month - 1) // 3 + 1} "
                f"'{date.year % 100:02d}"
            )
        )
    )
else:
    # Show each year when it fits; widen the interval for longer histories.
    if chart_span_years <= 8:
        year_interval = 1
    elif chart_span_years <= 16:
        year_interval = 2
    elif chart_span_years <= 35:
        year_interval = 5
    else:
        year_interval = 10
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(
    axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
    pad=8, colors=INK,
)
ax.tick_params(
    axis="y", which="major", length=0, pad=5, labelsize=TICK_SIZE, colors=INK,
)
fx_ax.tick_params(
    axis="y", which="major", length=0, pad=8, labelsize=TICK_SIZE, colors=INK,
)

# Bound the axis to the selected observation span, adding only enough room
# for the end bars and markers; never extend it merely to show an earlier tick.
edge_padding_days = bar_width_days / 2 + 35
ax.set_xlim(dates[0] - edge_padding_days, dates[-1] + edge_padding_days)
fx_ax.set_xlim(ax.get_xlim())

handles = [
    Patch(facecolor=BLUE, label="Foreign currency assets (FCA)"),
    Patch(facecolor=ORANGE, label="Gold reserves"),
    Patch(facecolor=GREEN, label="SDRs"),
    Patch(facecolor=PURPLE, label="IMF Reserve Tranche Position (RTP)"),
    plt.Line2D(
        [0], [0], color=YELLOW, linewidth=2.8, marker="o", markersize=5,
        markerfacecolor="white", label=f"USD/INR ({PERIOD_NAME} average)",
    ),
]
legend = ax.legend(
    handles=handles,
    loc="center",
    bbox_to_anchor=(0.5, -0.105),
    ncol=5,
    frameon=False,
    fontsize=11,
    handlelength=1.6,
    columnspacing=1.8,
    handletextpad=0.7,
)

# Center the legend in the available space between x tick labels and source.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
visible_x_ticks = [
    label for label in ax.get_xticklabels()
    if label.get_visible() and label.get_text()
]
if visible_x_ticks:
    tick_label_bottom = min(
        label.get_window_extent(renderer).y0 for label in visible_x_ticks
    )
    source_line_top = source_artist.get_window_extent(renderer).y1
    legend_center_y = (tick_label_bottom + source_line_top) / 2
    legend.set_bbox_to_anchor(
        (0.5, legend_center_y / fig.bbox.height), transform=fig.transFigure,
    )

plt.show()
