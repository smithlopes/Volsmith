from __future__ import annotations

from io import BytesIO
import ssl
from urllib.request import HTTPSHandler, Request, build_opener

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from tempfile import gettempdir
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


# =============================================================================
# Configuration
# =============================================================================

OECD_URL = (
    "https://sdmx.oecd.org/public/rest/v1/data/OECD.SDD.NAD,"
    "DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1/"
    "Q.Y.DEU+GBR+USA.S1..B1GQ......GY.?startPeriod=2000-Q1&format=csvfile"
)
COUNTRIES = {"GBR": "United Kingdom", "USA": "United States", "DEU": "Germany"}
COUNTRY_ORDER = ("United Kingdom", "United States", "Germany")

# Set either value to a date such as "2016-01-01" to bound the chart.
# None uses the earliest or latest available observation on that side.
START_DATE = "2022-01-01"
END_DATE = None

SLIDE_TITLE = "How Fast Are These Economies Growing?"
SOURCE_TEXT = "Source: OECD Quarterly National Accounts"
NOTE_TEXT = (
    "Note: Values show year-over-year real GDP growth: each quarter's seasonally adjusted, "
    "chain-linked GDP is compared with the same quarter one year earlier. Subtitle changes "
    "compare those growth rates with the previous quarter. OECD national accounts are revised."
)

FIG_W_IN = 16
FIG_H_IN = 9
FIG_DPI = 150
CANVAS_W = 1920
CANVAS_H = 1080

LEFT_MARGIN_PX = 60
RIGHT_MARGIN_PX = 1860

BLUE = "#6b8fd1"
ORANGE = "#e8856a"
GREEN = "#55a868"
BG = "#ffffff"
INK = "#000000"
SUBTITLE_INK = "#000000"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#434343"
COLORS = {"United Kingdom": BLUE, "United States": ORANGE, "Germany": GREEN}
FONT_STACK = (
    "Inter", "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans",
)

TITLE_SIZE = 28.536
SUBTITLE_SIZE = 12.138
FOOT_SIZE = 9.6
TICK_SIZE = 10.4

TITLE_TOP_PX, SUBTITLE_TOP_PX = 40, 103
SOURCE_TOP_PX, NOTE_TOP_PX = 978, 1002
CHART_AREA_TOP_PX, CHART_AREA_BOTTOM_PX = 173, 956
TITLE_BASELINE_PX = TITLE_TOP_PX + 43
SUBTITLE_BASELINES_PX = (SUBTITLE_TOP_PX + 21, SUBTITLE_TOP_PX + 45)
SOURCE_BASELINE_PX = SOURCE_TOP_PX + 15
NOTE_BASELINES_PX = (NOTE_TOP_PX + 15, NOTE_TOP_PX + 35)


# =============================================================================
# Download and prepare GDP growth data
# =============================================================================

request = Request(
    OECD_URL,
    headers={"Accept": "text/csvfile", "User-Agent": "Mozilla/5.0"},
)
ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = build_opener(HTTPSHandler(context=ssl_context))
with opener.open(request, timeout=30) as response:
    raw_data = pd.read_csv(BytesIO(response.read()))

country_col = next((c for c in raw_data.columns if c.upper() in {"REF_AREA", "LOCATION"}), None)
period_col = next((c for c in raw_data.columns if c.upper() in {"TIME_PERIOD", "TIME"}), None)
value_col = next((c for c in raw_data.columns if c.upper() == "OBS_VALUE"), None)
if country_col is None or period_col is None or value_col is None:
    raise ValueError(f"Unexpected OECD CSV columns: {list(raw_data.columns)}")

data = raw_data[[country_col, period_col, value_col]].rename(
    columns={country_col: "country", period_col: "year", value_col: "growth"}
)
data["country"] = data["country"].map(COUNTRIES)
data["year"] = pd.PeriodIndex(data["year"].astype(str), freq="Q").to_timestamp()
data["growth"] = pd.to_numeric(data["growth"], errors="coerce")
data = data.dropna(subset=["country", "growth"]).sort_values("year")
if data.empty:
    raise ValueError("OECD returned no usable GDP growth observations for GBR, USA, or DEU.")

available_start = data["year"].min()
available_end = data["year"].max()
start_date = pd.Timestamp(START_DATE) if START_DATE is not None else available_start
end_date = pd.Timestamp(END_DATE) if END_DATE is not None else available_end
if start_date > end_date:
    raise ValueError("START_DATE must be on or before END_DATE.")
data = data.loc[(data["year"] >= start_date) & (data["year"] <= end_date)]
if data.empty:
    raise ValueError("OECD returned no GDP growth observations in the selected dates.")

quarterly = data.pivot(index="year", columns="country", values="growth").sort_index()
common = quarterly.dropna(subset=list(COUNTRY_ORDER))
if common.empty:
    raise ValueError("OECD returned no quarter with observations for all three economies.")
latest_period = common.index[-1].to_period("Q")
latest_period_label = f"{latest_period.year}-Q{latest_period.quarter}"
latest_values = common.loc[common.index[-1]]
prior_index = common.index[-2] if len(common) > 1 else None
prior_period = prior_index.to_period("Q") if prior_index is not None else None
prior_period_label = (
    f"{prior_period.year}-Q{prior_period.quarter}" if prior_period is not None else None
)

leader = latest_values.idxmax()
if prior_period is None:
    subtitle = (
        f"{latest_period_label} real GDP growth vs year earlier: "
        f"UK {latest_values['United Kingdom']:+.1f}%, "
        f"US {latest_values['United States']:+.1f}%, "
        f"Germany {latest_values['Germany']:+.1f}%; {leader} grew fastest."
    )
else:
    changes = latest_values - common.loc[prior_index]
    strongest_acceleration = changes.idxmax()
    strongest_slowdown = changes.idxmin()
    subtitle = (
        f"{latest_period_label} real GDP growth year over year: "
        f"UK {latest_values['United Kingdom']:+.1f}%, "
        f"US {latest_values['United States']:+.1f}%, Germany {latest_values['Germany']:+.1f}%; "
        f"{leader} grew fastest. Versus {prior_period_label}, growth accelerated most in "
        f"{strongest_acceleration} ({changes[strongest_acceleration]:+.1f} pp) and eased "
        f"most in {strongest_slowdown} ({changes[strongest_slowdown]:+.1f} pp)."
    )


# =============================================================================
# Slide coordinate and text helpers
# =============================================================================

def px_x(value_px: float) -> float:
    return value_px / CANVAS_W


def px_y(value_px: float) -> float:
    return 1.0 - value_px / CANVAS_H


inter_fonts = Path.home() / "Library" / "Fonts"
inter_regular = inter_fonts / "Inter-VariableFont_opsz,wght.ttf"
inter_italic = inter_fonts / "Inter-Italic-VariableFont_opsz,wght.ttf"
inter_extrabold = Path(gettempdir()) / "volsmith-inter-extrabold.ttf"
if inter_regular.exists():
    font_manager.fontManager.addfont(str(inter_regular))
    static_font = instantiateVariableFont(
        TTFont(str(inter_regular)), {"opsz": 32, "wght": 800}, inplace=False
    )
    for name_id, value in {
        1: "Inter ExtraBold", 2: "Regular", 4: "Inter ExtraBold",
        6: "Inter-ExtraBold", 16: "Inter ExtraBold", 17: "Regular",
    }.items():
        for name_record in static_font["name"].names:
            if name_record.nameID == name_id:
                name_record.string = value.encode(name_record.getEncoding())
    static_font.save(str(inter_extrabold))
    font_manager.fontManager.addfont(str(inter_extrabold))
if inter_italic.exists():
    font_manager.fontManager.addfont(str(inter_italic))

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
# Title, subtitle, source, and note
# =============================================================================

title_font = FontProperties(fname=str(inter_extrabold)) if inter_extrabold.exists() else None
fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontproperties=title_font,
    fontweight="bold" if title_font is None else "normal",
    color=INK, ha="left", va="baseline",
)

wrapped_subtitle = wrap_to_margins(
    subtitle, SUBTITLE_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX
)
for line, baseline in zip(wrapped_subtitle, SUBTITLE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=SUBTITLE_SIZE, color=SUBTITLE_INK, ha="left", va="baseline",
    )

fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)

wrapped_note = wrap_to_margins(
    NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
)
for line, baseline in zip(wrapped_note, NOTE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Single chart
# =============================================================================

axes_left, axes_right = 205.0, 1798.0
axes_top = CHART_AREA_TOP_PX + 17
axes_bottom = CHART_AREA_BOTTOM_PX - 81
ax = fig.add_axes(
    [
        px_x(axes_left), px_y(axes_bottom), px_x(axes_right - axes_left),
        (axes_bottom - axes_top) / CANVAS_H,
    ],
    zorder=2,
    facecolor="none",
)

ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
country_lines = []
for country in COUNTRY_ORDER:
    series = data.loc[data["country"] == country]
    if series.empty:
        raise ValueError(f"OECD returned no GDP growth observations for {country}.")
    line, = ax.plot(
        series["year"], series["growth"].to_numpy(dtype=float),
        color=COLORS[country], linewidth=2.6,
        solid_capstyle="round", solid_joinstyle="round", zorder=3, label=country,
    )
    country_lines.append(line)
    ax.plot(
        series["year"].iloc[-1], series["growth"].iloc[-1], marker="o",
        markersize=6, markerfacecolor="white", markeredgecolor=COLORS[country],
        markeredgewidth=1.8, linestyle="none", zorder=6,
    )

ax.set_axisbelow(True)
ax.grid(True, which="major", axis="both", linestyle="--", linewidth=1, color=GRID, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.spines["bottom"].set_color(AXIS)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

display_end_date = end_date + pd.Timedelta(days=120)
display_years = (display_end_date - start_date).days / 365.25
year_tick_interval = max(1, int(round(display_years / 8)))
ax.xaxis.set_major_locator(mdates.YearLocator(year_tick_interval, month=1, day=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(
    axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
    pad=8, colors=INK,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:+.0f}%"))
ax.tick_params(axis="y", which="major", length=0, pad=10, labelsize=TICK_SIZE, colors=INK)
ax.set_ylabel("Year-over-year change (%)", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.set_xlim(start_date - pd.Timedelta(days=120), display_end_date)
ax.yaxis.set_label_coords(-0.060, 0.5)

# Place one compact latest-data card in the least busy corner of the plot.
# Score each candidate against interpolated line segments, not just quarterly points,
# so the card avoids crossing a series between observations as well.
label_width_px = max(measure_width_px(country, 9) for country in COUNTRY_ORDER)
value_width_px = max(
    measure_width_px(f"{latest_values[country]:+.1f}%", 9) + 2
    for country in COUNTRY_ORDER
)
card_width_px = 8 + 9 + 4 + label_width_px + 12 + value_width_px + 8
card_width = card_width_px / (axes_right - axes_left)
card_height = 0.20
line_samples_by_series = []
for line in country_lines:
    points = line.get_xydata()
    points_axes = ax.transAxes.inverted().transform(ax.transData.transform(points))
    samples = []
    for first, second in zip(points_axes[:-1], points_axes[1:]):
        samples.extend(first + (second - first) * step / 12 for step in range(13))
    if len(points_axes):
        samples.append(points_axes[-1])
    line_samples_by_series.append(samples)


anchor_x, anchor_y = ax.transAxes.inverted().transform(
    ax.transData.transform(
        (mdates.date2num(common.index[-1]), float(latest_values.mean()))
    )
)
edge_gap = 0.015
left_limit, right_limit = 0.02, 0.98 - card_width
bottom_limit, top_limit = 0.03, 0.97 - card_height
x_candidates = []
if anchor_x + edge_gap <= right_limit:
    x_candidates.append(anchor_x + edge_gap)
if anchor_x - card_width - edge_gap >= left_limit:
    x_candidates.append(anchor_x - card_width - edge_gap)
y_candidates = []
for offset in (edge_gap, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42):
    above = min(anchor_y + offset, top_limit)
    below = max(anchor_y - card_height - offset, bottom_limit)
    if above <= top_limit:
        y_candidates.append(above)
    if below >= bottom_limit:
        y_candidates.append(below)
y_candidates = sorted(set(y_candidates))
candidate_positions = [(left, bottom) for left in x_candidates for bottom in y_candidates]
if not candidate_positions:
    candidate_positions = [(right_limit, top_limit)]


def card_overlap_score(position: tuple[float, float]) -> tuple[int, int]:
    left, bottom = position
    right, top = left + card_width, bottom + card_height
    padding = 0.015
    points_inside = 0
    series_crossed = 0
    for samples in line_samples_by_series:
        hits = sum(
            left - padding <= x <= right + padding
            and bottom - padding <= y <= top + padding
            for x, y in samples
        )
        points_inside += hits
        series_crossed += hits > 0
    return series_crossed, points_inside


def card_anchor_distance(position: tuple[float, float]) -> float:
    left, bottom = position
    dx = max(left - anchor_x, 0.0, anchor_x - (left + card_width))
    dy = max(bottom - anchor_y, 0.0, anchor_y - (bottom + card_height))
    return (dx**2 + dy**2) ** 0.5


card_left, card_bottom = min(
    candidate_positions,
    key=lambda position: (card_overlap_score(position), card_anchor_distance(position)),
)
card_right = card_left + card_width
card_top = card_bottom + card_height
if anchor_x < card_left:
    connector_start = (card_left, min(max(anchor_y, card_bottom + 0.04), card_top - 0.04))
elif anchor_x > card_right:
    connector_start = (card_right, min(max(anchor_y, card_bottom + 0.04), card_top - 0.04))
elif anchor_y < card_bottom:
    connector_start = (min(max(anchor_x, card_left + 0.04), card_right - 0.04), card_bottom)
else:
    connector_start = (min(max(anchor_x, card_left + 0.04), card_right - 0.04), card_top)

ax.annotate(
    "",
    # Point to the center of the three endpoint markers, just beyond the data,
    # so the leader identifies the group rather than a particular country line.
    xy=(min(anchor_x + 0.012, 0.99), anchor_y),
    xycoords="axes fraction",
    xytext=connector_start,
    textcoords="axes fraction",
    arrowprops={
        "arrowstyle": "-|>", "color": "#969696", "linewidth": 0.7,
        "mutation_scale": 7, "shrinkA": 1, "shrinkB": 4,
    },
    annotation_clip=False,
    zorder=8,
)

card = ax.inset_axes([card_left, card_bottom, card_width, card_height], zorder=10)
card.patch.set_visible(False)
card.add_patch(
    FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        transform=card.transAxes, facecolor="white", edgecolor="#e3e3e3",
        linewidth=0.8, alpha=0.98, mutation_scale=10, zorder=0,
    )
)
for spine in card.spines.values():
    spine.set_visible(False)
card.set_xticks([])
card.set_yticks([])
card.set_xlim(0, 1)
card.set_ylim(0, 1)
card.text(
    0.07, 0.86, f"Latest GDP growth vs year ago · {latest_period_label}",
    ha="left", va="center", fontsize=8, color="#777777",
)
for row, country in enumerate(COUNTRY_ORDER):
    y = 0.63 - row * 0.21
    marker_x = 12 / card_width_px
    label_x = 21 / card_width_px
    value_x = 1 - 8 / card_width_px
    card.scatter([marker_x], [y], s=17, color=COLORS[country], clip_on=False)
    card.text(label_x, y, country, ha="left", va="center", fontsize=9, color=INK)
    card.text(
        value_x, y, f"{latest_values[country]:+.1f}%", ha="right", va="center",
        fontsize=9, color=INK,
    )

legend = ax.legend(
    loc="center", ncol=3, frameon=False,
    fontsize=11, handlelength=2.2, columnspacing=1.8, handletextpad=0.7,
)
legend.set_bbox_to_anchor(
    (0.5, px_y(CHART_AREA_BOTTOM_PX - 17)), transform=fig.transFigure,
)

plt.show()
