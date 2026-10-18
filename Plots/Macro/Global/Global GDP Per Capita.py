from __future__ import annotations

import json
import ssl
from urllib.request import HTTPSHandler, Request, build_opener

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
from pathlib import Path
from tempfile import gettempdir
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


# =============================================================================
# Configuration
# =============================================================================

# World Bank World Development Indicators: GDP per capita growth (annual %).
INDICATOR = "NY.GDP.PCAP.KD.ZG"
COUNTRIES = {
    "USA": "United States",
    "CHE": "Switzerland",
    "GBR": "United Kingdom",
    "IND": "India",
    "FRA": "France",
    "DEU": "Germany",
    "SGP": "Singapore",
}
COUNTRY_ORDER = tuple(COUNTRIES.values())

# Set either value to a year such as 2010 to bound the chart.
START_YEAR = 2015
END_YEAR = None  # None uses the latest available World Bank observation.

API_URL = (
    "https://api.worldbank.org/v2/country/"
    f"{';'.join(COUNTRIES)}/indicator/{INDICATOR}"
    "?format=json&per_page=20000"
)

SLIDE_TITLE = "Real GDP per Capita Growth"
SOURCE_TEXT = "Source: World Bank, World Development Indicators (NY.GDP.PCAP.KD.ZG)"
NOTE_TEXT = (
    "Note: Annual percentage growth in GDP per capita, based on constant local currency. "
    "This is real output per person growth, not nominal GDP per capita growth. "
    "National accounts data are revised; the latest observation may differ across countries."
)

FIG_W_IN, FIG_H_IN, FIG_DPI = 16, 9, 150
CANVAS_W, CANVAS_H = 1920, 1080
LEFT_MARGIN_PX, RIGHT_MARGIN_PX = 60, 1860

COLORS = {
    "United States": "#6b8fd1",
    "Switzerland": "#8172b2",
    "United Kingdom": "#e8856a",
    "India": "#55a868",
    "France": "#64b5cd",
    "Germany": "#c44e52",
    "Singapore": "#fcb149",
}
BG, INK = "#ffffff", "#000000"
SUBTITLE_INK, AXIS = "#000000", "#7f7f7f"
GRID, FOOT = "#b9b9b9", "#434343"
FONT_STACK = (
    "Inter", "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans",
)

TITLE_SIZE, SUBTITLE_SIZE, FOOT_SIZE, TICK_SIZE = 28.536, 12.138, 9.6, 10.4
TITLE_TOP_PX, SUBTITLE_TOP_PX = 40, 103
SOURCE_TOP_PX, NOTE_TOP_PX = 978, 1002
CHART_AREA_TOP_PX, CHART_AREA_BOTTOM_PX = 173, 956
TITLE_BASELINE_PX = TITLE_TOP_PX + 43
SUBTITLE_BASELINES_PX = (SUBTITLE_TOP_PX + 21, SUBTITLE_TOP_PX + 45)
SOURCE_BASELINE_PX = SOURCE_TOP_PX + 15
NOTE_BASELINES_PX = (NOTE_TOP_PX + 15, NOTE_TOP_PX + 35)


# =============================================================================
# Download and prepare annual real GDP per capita growth
# =============================================================================

ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = build_opener(HTTPSHandler(context=ssl_context))
request = Request(API_URL, headers={"User-Agent": "Mozilla/5.0"})
with opener.open(request, timeout=30) as response:
    payload = json.loads(response.read().decode("utf-8"))


if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
    raise ValueError(f"Unexpected World Bank API response for {INDICATOR}: {payload!r}")

records = []
for item in payload[1]:
    if item is None or item.get("value") is None:
        continue
    iso3 = item.get("countryiso3code") or item.get("country", {}).get("id")
    country = COUNTRIES.get(iso3)
    if country is None:
        continue
    records.append({
        "year": int(item["date"]),
        "country": country,
        "growth": float(item["value"]),
    })

data = pd.DataFrame(records)
if data.empty:
    raise ValueError("World Bank returned no usable GDP per capita growth observations.")

available_start, available_end = int(data["year"].min()), int(data["year"].max())
start_year = START_YEAR if START_YEAR is not None else available_start
end_year = END_YEAR if END_YEAR is not None else available_end
if start_year > end_year:
    raise ValueError("START_YEAR must be on or before END_YEAR.")
data = data.loc[data["year"].between(start_year, end_year)].copy()
if data.empty:
    raise ValueError("World Bank returned no observations in the selected year range.")

data["date"] = pd.to_datetime(data["year"].astype(str) + "-01-01")
wide = data.pivot(index="year", columns="country", values="growth").reindex(
    columns=COUNTRY_ORDER
)
common = wide.dropna(subset=list(COUNTRY_ORDER))
if common.empty:
    raise ValueError("No year has observations for all seven selected countries.")

latest_year = int(common.index[-1])
latest_values = common.iloc[-1]
if len(common) > 1:
    previous_year = int(common.index[-2])
    previous_values = common.iloc[-2]
    growth_changes = latest_values - previous_values
    subtitle = (
        f"{latest_year} real GDP per capita growth: "
        + "; ".join(
            f"{COUNTRIES[iso3]} {latest_values[COUNTRIES[iso3]]:+.1f}%"
            for iso3 in COUNTRIES
        )
        + f". Change vs {previous_year}: "
        + "; ".join(
            f"{country} {growth_changes[country]:+.1f} pp"
            for country in COUNTRY_ORDER
        )
        + "."
    )
else:
    subtitle = (
        f"{latest_year} real GDP per capita growth: "
        + "; ".join(
            f"{COUNTRIES[iso3]} {latest_values[COUNTRIES[iso3]]:+.1f}%"
            for iso3 in COUNTRIES
        )
        + "."
    )


# =============================================================================
# Slide coordinates and renderer-measured text layout
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
    text: str,
    font_size: float,
    max_width_px: float,
    font_style: str = "normal",
    balance_two_lines: bool = False,
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

    # Greedy wrapping can leave one subtitle line much shorter. Find a natural
    # two-line split that uses both lines well, preferring sentence boundaries.
    if balance_two_lines and len(lines) == 2:
        candidates = []
        for split_at in range(1, len(words)):
            first, second = " ".join(words[:split_at]), " ".join(words[split_at:])
            first_width = measure_width_px(first, font_size, font_style)
            second_width = measure_width_px(second, font_size, font_style)
            if first_width <= max_width_px and second_width <= max_width_px:
                sentence_break = words[split_at - 1].endswith((".", ";"))
                score = abs(first_width - second_width) + (0 if sentence_break else 180)
                candidates.append((score, first, second))
        if candidates:
            _, first, second = min(candidates, key=lambda item: item[0])
            lines = [first, second]
    return lines


title_font = FontProperties(fname=str(inter_extrabold)) if inter_extrabold.exists() else None
fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontproperties=title_font,
    fontweight="bold" if title_font is None else "normal",
    color=INK, ha="left", va="baseline",
)
subtitle_lines = wrap_to_margins(
    subtitle,
    SUBTITLE_SIZE,
    RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
    balance_two_lines=True,
)
if len(subtitle_lines) > len(SUBTITLE_BASELINES_PX):
    raise ValueError("The dynamic subtitle exceeds the two-line layout; reduce the selected range or text.")
for line, baseline in zip(subtitle_lines, SUBTITLE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=SUBTITLE_SIZE, color=SUBTITLE_INK, ha="left", va="baseline",
    )

fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)
note_lines = wrap_to_margins(
    NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
)
for line, baseline in zip(note_lines, NOTE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Chart
# =============================================================================

axes_left, axes_right = 205.0, 1798.0
axes_top = CHART_AREA_TOP_PX + 17
axes_bottom = CHART_AREA_BOTTOM_PX - 81
ax = fig.add_axes(
    [
        px_x(axes_left), px_y(axes_bottom), px_x(axes_right - axes_left),
        (axes_bottom - axes_top) / CANVAS_H,
    ],
    facecolor="none",
)

ax.axhline(0, color=AXIS, linewidth=1.1, zorder=1)
for country in COUNTRY_ORDER:
    series = data.loc[data["country"] == country].sort_values("year")
    if series.empty:
        raise ValueError(f"World Bank returned no observations for {country}.")
    ax.plot(
        series["date"],
        series["growth"],
        color=COLORS[country],
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=country,
        zorder=3,
    )

ax.set_axisbelow(True)
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=1, color=GRID, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(AXIS)
    ax.spines[spine].set_linewidth(1.5)

span_years = max(end_year - start_year, 1)
year_interval = 1 if span_years <= 16 else 2 if span_years <= 32 else 5
tick_years = list(range(start_year, end_year + 1, year_interval))
ax.set_xticks(pd.to_datetime([f"{year}-01-01" for year in tick_years]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:+.0f}%"))
ax.tick_params(
    axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
    pad=8, colors=INK,
)
ax.tick_params(axis="y", which="major", length=0, pad=8, labelsize=TICK_SIZE, colors=INK)
ax.set_ylabel("Annual change in real GDP per capita (%)", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.yaxis.set_label_coords(-0.075, 0.5)
ax.set_xlim(pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=120),
            pd.Timestamp(f"{end_year + 1}-01-01") + pd.Timedelta(days=30))

legend = ax.legend(
    loc="center", ncol=7, frameon=False,
    fontsize=11, handlelength=1.8, columnspacing=1.8, handletextpad=0.7,
)
legend.set_bbox_to_anchor(
    (0.5, px_y(CHART_AREA_BOTTOM_PX - 17)), transform=fig.transFigure,
)

plt.show()
