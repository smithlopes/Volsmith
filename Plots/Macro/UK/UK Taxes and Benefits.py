import csv
from io import BytesIO, StringIO
from html.parser import HTMLParser
import re
import ssl
import time
from urllib.parse import quote, unquote, urljoin
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, Request, build_opener

import certifi
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
from matplotlib.patches import Patch
from pathlib import Path
from tempfile import gettempdir
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


# =============================================================================
# Configuration
# =============================================================================

ONS_CSV_API = "https://www.ons.gov.uk/generator?format=csv&uri={uri}"
ONS_SERIES_URI = (
    "/economy/governmentpublicsectorandtaxes/publicsectorfinance/"
    "timeseries/{series}/pusf"
)
TAX_SERIES = "AHHY"      # Total public sector taxes and National Insurance, £m, NSA
CORP_TAX_SERIES = "N445"  # Corporation tax, gross of tax credits, £m, NSA
BENEFIT_SERIES = "ANLY"  # Net social benefits (D62), £m, current prices, not seasonally adjusted (CPNSA)
TAX_LABEL = "Taxes and NI (excl. corporation tax)"
MIGRATION_DATASET_PAGE = (
    "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/"
    "internationalmigration/datasets/"
    "longterminternationalimmigrationemigrationandnetmigrationflowsprovisional"
)

# Set either value to a date such as "2016-01-01" to bound the chart.
# None uses the earliest or latest available observation on that side.
START_DATE = "2021-01-01"  # Example: "2021-01-01"
END_DATE = None  # Example: "2024-12-31"
BAR_FREQUENCY = "quarterly"  # Choose "quarterly" or "yearly"

SLIDE_TITLE = "UK Taxes and Benefits, Excluding Corporation Tax"
SOURCE_TEXT = (
    "Source: ONS Public sector finances time series and Long-term international migration"
)
NOTE_TEXT = (
    "Note: Nominal £, not inflation-adjusted; the tax and benefit series are not seasonally adjusted. "
    "Taxes are total public-sector taxes and National Insurance less corporation tax, including income taxes, VAT, "
    "Council Tax and other taxes. Net social benefits (D62) include National Insurance Fund benefits, social assistance "
    "and public-service pension payments, net of contributions; they are one part of spending, not total government spending. "
    "Axis: £bn below £1tn, £tn from £1tn."
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
YELLOW = "#fcb149"
BG = "#ffffff"
INK = "#000000"
SUBTITLE_INK = "#000000"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#434343"
FONT_STACK = (
    "Inter", "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans",
)

TITLE_SIZE = 28.536
SUBTITLE_SIZE = 12.138
FOOT_SIZE = 9.6
TICK_SIZE = 10.4

# CSS top coordinates with Inter's rendered line-box offsets converted to
# Matplotlib baselines on the 1920 x 1080 design canvas.
TITLE_TOP_PX = 40
SUBTITLE_TOP_PX = 103
SOURCE_TOP_PX = 978
NOTE_TOP_PX = 1002
CHART_AREA_TOP_PX = 173
CHART_AREA_BOTTOM_PX = 956

TITLE_BASELINE_PX = TITLE_TOP_PX + 43
SUBTITLE_BASELINES_PX = (SUBTITLE_TOP_PX + 21, SUBTITLE_TOP_PX + 45)
SOURCE_BASELINE_PX = SOURCE_TOP_PX + 15
NOTE_BASELINES_PX = (NOTE_TOP_PX + 15, NOTE_TOP_PX + 35)


# =============================================================================
# ONS API download and frequency selection
# =============================================================================

ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = build_opener(HTTPSHandler(context=ssl_context))


def read_ons_response(request: Request) -> bytes:
    """Read an ONS response, retrying rate limits and temporary server errors."""
    for attempt in range(4):
        try:
            with opener.open(request, timeout=30) as response:
                return response.read()
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 30) if retry_after else 2 ** (attempt + 1)
            except ValueError:
                delay = 2 ** (attempt + 1)
            time.sleep(delay)


def fetch_series(series_id: str, frequency: str) -> pd.Series:
    series_uri = ONS_SERIES_URI.format(series=series_id.lower())
    url = ONS_CSV_API.format(uri=quote(series_uri, safe=""))
    request = Request(url, headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0"})
    csv_text = read_ons_response(request).decode("utf-8-sig")

    rows: list[tuple[pd.Timestamp, float]] = []
    for row in csv.reader(StringIO(csv_text)):
        if len(row) < 2:
            continue
        period, value = row[0].strip().strip('"'), row[1].strip().strip('"')
        if value in ("", "..", "-", "[x]"):
            continue
        try:
            if frequency == "quarterly" and (
                len(period) == 7 and period[:4].isdigit()
                and period[4:6] == " Q" and period[6] in "1234"
            ):
                date = pd.Period(period.replace(" ", ""), freq="Q").start_time
            elif frequency == "yearly" and len(period) == 4 and period.isdigit():
                date = pd.Timestamp(year=int(period), month=1, day=1)
            else:
                continue
            rows.append((date, float(value.replace(",", ""))))
        except (TypeError, ValueError):
            continue

    if not rows:
        raise ValueError(
            f"ONS returned no usable {frequency} observations for series {series_id}."
        )
    return pd.Series(dict(rows), dtype=float).sort_index()


def fetch_net_migration() -> pd.Series:
    class XlsxLinks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href")
                if href and ".xlsx" in unquote(href).lower():
                    self.links.append(href)

    page_request = Request(
        MIGRATION_DATASET_PAGE,
        headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"},
    )
    page_html = read_ons_response(page_request).decode("utf-8", errors="replace")

    link_parser = XlsxLinks()
    link_parser.feed(page_html)
    if not link_parser.links:
        raise ValueError("Could not find the latest ONS net-migration workbook.")

    workbook_url = urljoin(MIGRATION_DATASET_PAGE, link_parser.links[0])
    workbook_request = Request(
        workbook_url,
        headers={
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "User-Agent": "Mozilla/5.0",
        },
    )
    workbook = BytesIO(read_ons_response(workbook_request))

    table = pd.read_excel(workbook, sheet_name="1", header=None)
    month_numbers = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
    migration_rows: list[tuple[pd.Timestamp, float]] = []
    for _, row in table.iterrows():
        if len(row) < 3 or str(row.iloc[0]).strip().lower() != "net migration":
            continue
        period_match = re.fullmatch(
            r"YE\s+(Mar|Jun|Sep|Dec)\s+(\d{2})(?:\s+[RP])?",
            str(row.iloc[1]).strip(),
        )
        if not period_match:
            continue
        month_name, year_suffix = period_match.groups()
        year = 2000 + int(year_suffix)
        date = pd.Period(year=year, month=month_numbers[month_name], freq="M").end_time
        value = pd.to_numeric(row.iloc[2], errors="coerce")
        if pd.notna(value):
            migration_rows.append((date.normalize(), float(value)))

    if not migration_rows:
        raise ValueError("Could not parse net-migration observations from the latest ONS workbook.")
    return pd.Series(dict(migration_rows), dtype=float).sort_index()


if BAR_FREQUENCY not in {"quarterly", "yearly"}:
    raise ValueError('BAR_FREQUENCY must be either "quarterly" or "yearly".')

print(f"Downloading ONS series {TAX_SERIES} (total taxes and National Insurance)...")
taxes = fetch_series(TAX_SERIES, BAR_FREQUENCY)
print(f"Downloading ONS series {CORP_TAX_SERIES} (corporation tax to exclude)...")
corporation_tax = fetch_series(CORP_TAX_SERIES, BAR_FREQUENCY)
taxes = taxes.sub(corporation_tax)
print(f"Downloading ONS series {BENEFIT_SERIES} (net social benefits)...")
benefits = fetch_series(BENEFIT_SERIES, BAR_FREQUENCY)
print("Downloading ONS long-term net-migration estimates...")
net_migration = fetch_net_migration()

plot_data = pd.concat(
    [taxes.rename(TAX_LABEL), benefits.rename("Net social benefits")],
    axis=1,
).dropna()
plot_data = plot_data.sort_index()
if plot_data.empty:
    raise ValueError(f"ONS returned no overlapping {BAR_FREQUENCY} observations for both series.")

start_date = pd.Timestamp(START_DATE) if START_DATE is not None else plot_data.index.min()
end_date = pd.Timestamp(END_DATE) if END_DATE is not None else plot_data.index.max()
if start_date > end_date:
    raise ValueError("START_DATE must be on or before END_DATE.")
plot_data = plot_data.loc[(plot_data.index >= start_date) & (plot_data.index <= end_date)]
if BAR_FREQUENCY == "yearly":
    included_years = set(plot_data.index.year)
    net_migration = net_migration.loc[
        (net_migration.index.month == 12)
        & net_migration.index.year.isin(included_years)
    ]
else:
    net_migration = net_migration.loc[
        (net_migration.index >= start_date) & (net_migration.index <= end_date)
    ]
if plot_data.empty:
    raise ValueError("ONS returned no observations in the selected dates.")
if net_migration.empty:
    raise ValueError("ONS returned no net-migration observations in the selected dates.")

# ONS reports these values in £ millions; plot them in £ trillions internally.
# The axis formatter displays amounts below £1tn as £bn.
plot_data = plot_data / 1_000_000


def format_money(value_trillions: float, signed: bool = False) -> str:
    sign = ""
    if signed:
        sign = "+" if value_trillions >= 0 else "−"
    magnitude = abs(value_trillions)
    if magnitude < 1:
        amount = f"£{magnitude * 1_000:,.0f}bn"
    else:
        trillions = f"{magnitude:,.1f}".rstrip("0").rstrip(".")
        amount = f"£{trillions}tn"
    return f"{sign}{amount}"


latest_date = plot_data.index[-1]
if BAR_FREQUENCY == "quarterly":
    latest_period = latest_date.to_period("Q")
    latest_period_label = f"{latest_period.year}-Q{latest_period.quarter}"
    comparison_label = "previous quarter"
else:
    latest_period_label = str(latest_date.year)
    comparison_label = "previous year"
latest = plot_data.iloc[-1]
if len(plot_data) > 1:
    previous = plot_data.iloc[-2]
    tax_change = latest[TAX_LABEL] - previous[TAX_LABEL]
    benefit_change = latest["Net social benefits"] - previous["Net social benefits"]
    subtitle = (
        f"In {latest_period_label}, taxes and NI (excluding corporation tax) raised "
        f"{format_money(latest.iloc[0])} and net social benefits cost "
        f"{format_money(latest.iloc[1])}. Versus the {comparison_label}, revenue changed "
        f"{format_money(tax_change, signed=True)} and benefit spending changed "
        f"{format_money(benefit_change, signed=True)}."
    )
else:
    subtitle = (
        f"In {latest_period_label}, taxes and NI (excluding corporation tax) raised "
        f"{format_money(latest.iloc[0])} and net social benefits cost "
        f"{format_money(latest.iloc[1])}."
    )


# =============================================================================
# Slide coordinate and text helpers
# =============================================================================

def px_x(value_px: float) -> float:
    return value_px / CANVAS_W


def px_y(value_px: float) -> float:
    return 1.0 - value_px / CANVAS_H


def wrap_to_margins(
    text: str, font_size: float, max_width_px: float, font_style: str = "normal"
) -> list[str]:
    probe = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
    renderer = probe.canvas.get_renderer()
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        artist = probe.text(0, 0, candidate, fontsize=font_size, style=font_style)
        width = artist.get_window_extent(renderer).width * CANVAS_W / probe.bbox.width
        artist.remove()
        if current and width > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    plt.close(probe)
    return lines


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

title_font = FontProperties(fname=str(inter_extrabold)) if inter_extrabold.exists() else None
fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontproperties=title_font,
    fontweight="bold" if title_font is None else "normal",
    color=INK, ha="left", va="baseline",
)
for line, baseline in zip(
    wrap_to_margins(subtitle, SUBTITLE_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX),
    SUBTITLE_BASELINES_PX,
):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=SUBTITLE_SIZE, color=SUBTITLE_INK, ha="left", va="baseline",
    )
source_artist = fig.text(
    px_x(LEFT_MARGIN_PX), px_y(SOURCE_BASELINE_PX), SOURCE_TEXT,
    fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
)
for line, baseline in zip(
    wrap_to_margins(
        NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic"
    ),
    NOTE_BASELINES_PX,
):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Paired annual or quarterly bar chart
# =============================================================================

# Leave room inside the slide margin for the secondary y-axis ticks and label.
axes_left, axes_right = 205.0, 1720.0
axes_top = CHART_AREA_TOP_PX + 17
axes_bottom = CHART_AREA_BOTTOM_PX - 81
ax = fig.add_axes(
    [
        px_x(axes_left), px_y(axes_bottom), px_x(axes_right - axes_left),
        (axes_bottom - axes_top) / CANVAS_H,
    ],
    facecolor="none",
)

dates = mdates.date2num(plot_data.index.to_pydatetime())
bar_width_days = 34 if BAR_FREQUENCY == "quarterly" else 182.5
date_padding_days = 50 if BAR_FREQUENCY == "quarterly" else 200
if BAR_FREQUENCY == "yearly":
    dates = mdates.date2num(
        (plot_data.index + pd.DateOffset(months=6)).to_pydatetime()
    )
net_migration_dates = mdates.date2num(net_migration.index.to_pydatetime())
net_migration_thousands = net_migration.to_numpy(dtype=float) / 1_000
tax_values = plot_data[TAX_LABEL].to_numpy()
benefit_values = plot_data["Net social benefits"].to_numpy()
ax.bar(
    dates - bar_width_days / 2, tax_values, width=bar_width_days,
    color=BLUE, edgecolor="white", linewidth=0.5, zorder=3,
)
ax.bar(
    dates + bar_width_days / 2, benefit_values, width=bar_width_days,
    color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3,
)

migration_ax = ax.twinx()
migration_ax.patch.set_visible(False)
migration_ax.plot(
    net_migration_dates, net_migration_thousands, color=YELLOW, linewidth=2.8,
    marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
    zorder=5, label="Net migration (year ending)",
)

ax.set_axisbelow(True)
ax.grid(True, axis="y", linestyle="--", linewidth=1, color=GRID, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.spines["bottom"].set_color(AXIS)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
migration_ax.spines["top"].set_visible(False)
migration_ax.spines["left"].set_visible(False)
migration_ax.spines["bottom"].set_visible(False)
migration_ax.spines["right"].set_color(AXIS)
migration_ax.spines["right"].set_linewidth(1.5)

date_locator = mdates.AutoDateLocator(minticks=5, maxticks=9, interval_multiples=True)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
chart_min = min(dates[0], net_migration_dates[0])
chart_max = max(dates[-1], net_migration_dates[-1])
ax.set_xlim(chart_min - date_padding_days, chart_max + date_padding_days)
migration_ax.set_xlim(ax.get_xlim())
ax.tick_params(
    axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2,
    pad=8, colors=INK,
)
def format_quarterly_amount(value: float, _position: int) -> str:
    return format_money(value)


ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_quarterly_amount))
# Keep the y-axis label fixed and tuck tick labels closer to the spine to
# provide more whitespace between the tick labels and the axis label.
ax.tick_params(axis="y", which="major", length=0, pad=5, labelsize=TICK_SIZE, colors=INK)
period_name = "Quarterly" if BAR_FREQUENCY == "quarterly" else "Annual"
ax.set_ylabel(f"{period_name} amount, current prices", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.yaxis.set_label_coords(-0.060, 0.5)
migration_ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda value, _: f"{value:,.0f}k")
)
migration_ax.tick_params(
    axis="y", which="major", length=0, pad=8, labelsize=TICK_SIZE, colors=INK,
)
migration_ax.set_ylabel(
    "Net migration (thousands, year ending)",
    fontsize=TICK_SIZE, color=INK, labelpad=10,
)
handles = [
    Patch(facecolor=BLUE, label=TAX_LABEL),
    Patch(facecolor=ORANGE, label="Net social benefits"),
    plt.Line2D(
        [0], [0], color=YELLOW, linewidth=2.8, marker="o", markersize=5,
        markerfacecolor="white", label="Net migration (year ending)",
    ),
]
legend = ax.legend(
    handles=handles,
    loc="center", ncol=3, frameon=False,
    fontsize=11, handlelength=1.6, columnspacing=1.8, handletextpad=0.7,
)

# Keep the legend inside the specified chart-and-legend area (ending at y=956).
legend.set_bbox_to_anchor(
    (0.5, px_y(CHART_AREA_BOTTOM_PX - 17)), transform=fig.transFigure,
)

plt.show()

