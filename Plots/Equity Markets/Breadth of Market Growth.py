from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import time
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


try:
    from mscidata import msci
except ImportError as error:
    raise ImportError(
        "\nCould not import 'msci' from the 'mscidata' package.\n\n"
        "Install or upgrade the required packages using:\n\n"
        "    pip install --upgrade msci-data pandas\n\n"
        f"Original error: {error}"
    ) from error


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_VERSION = "SIDE_BY_SIDE_PANELS_SLIDE_MATCH_v8_ARROW_PATH_2026-10-07"
print(f"Running script version: {SCRIPT_VERSION}")

YEAR: int | None = None
# True uses the latest 12 calendar months, including the current partial month.
# False uses the prior 12 completed calendar months.
INCLUDE_CURRENT_MONTH = True
SAVE_CSV = False
DOWNLOAD_LOOKBACK_DAYS = 10
MSCI_RETURN_VARIANT = "NETR"

MSCI_WORLD_CAP_WEIGHTED = "990100"
MSCI_WORLD_EQUAL_WEIGHTED = "129857"
MSCI_EM_CAP_WEIGHTED = "891800"
MSCI_EM_EQUAL_WEIGHTED = "129856"

BLUE = "#6b8fd1"
ORANGE = "#e8856a"

SERIES_DEFINITIONS = [
    (MSCI_WORLD_CAP_WEIGHTED, MSCI_WORLD_EQUAL_WEIGHTED, "MSCI World", BLUE),
    (MSCI_EM_CAP_WEIGHTED, MSCI_EM_EQUAL_WEIGHTED, "MSCI Emerging Markets", ORANGE),
]

ALL_INDEX_CODES = sorted(
    {
        code
        for cap_code, equal_code, _, _ in SERIES_DEFINITIONS
        for code in (cap_code, equal_code)
    }
)

SERIES_COLOURS = {label: colour for _, _, label, colour in SERIES_DEFINITIONS}

OUTPUT_DIR = Path(__file__).resolve().parent
PNG_FILE = OUTPUT_DIR / "MSCI_EqualWeight_vs_CapWeight_Ratio.png"
CSV_FILE = OUTPUT_DIR / "MSCI_EqualWeight_vs_CapWeight_Ratio_Data.csv"

FIG_W_IN = 16
FIG_H_IN = 9
FIG_DPI = 150
CANVAS_W = 1920
CANVAS_H = 1080

LEFT_MARGIN_PX = 90
RIGHT_MARGIN_PX = 1920 - 90

BANNER_RADIUS_PX = 7
BANNER_TEXT_BASELINE_PX = 226.7

NEUTRAL_BAND_PCT = 1.0

FONT_STACK = (
    "Barclays Effra",
    "Barclays Effra App",
    "Inter",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)

BG = "#ffffff"
INK = "#000000"
SUBTITLE_INK = "#2d2a2a"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#8c8c8c"
TREND = "#fcb149"
BANNER_TEXT = "#ffffff"

TITLE_SIZE = 26
SUBTITLE_SIZE = 14
FOOT_SIZE = 11
TICK_SIZE = 13
BANNER_SIZE = 18

SLIDE_TITLE = "Is Global Equity Growth Concentrated or Broad-Based?"
SOURCE_TEXT = "Source: MSCI"

NOTE_TEXT = (
    "Note: Ratio = daily MSCI Net Total Return (USD, net of withholding "
    "tax) of the cap-weighted index / equal-weighted index: World "
    "(990100 / 129857), EM (891800 / 129856). Dashed line = linear fit "
    "to the log ratio; slope = fitted % change in the ratio per year "
    "(positive = cap-weighted outpacing equal-weighted, negative = the "
    "reverse). Indices cannot be invested in directly and exclude fees "
    "and costs. Separate y-axes."
)

TITLE_BASELINE_PX = 83
SUBTITLE_BASELINES_PX = (124, 155)
SOURCE_BASELINE_PX = 981
NOTE_BASELINES_PX = [1013, 1037]

CALLOUT_SIZE = 10
SLOPE_UNIT_SUFFIX = " p.a."
ARROW_CLEARANCE_PX = 9.0
ARROW_OVERLAP_WEIGHT = 8
PROMINENCE_MUCH = 2.0
PROMINENCE_MORE = 1.25

PANEL_GEOMETRY = [
    {
        "axes": (225.0, 941.3, 261.2, 853.7),
        "banner": (225.0, 945.0, 192.0, 240.0),
    },
    {
        "axes": (1083.6, 1798.0, 265.0, 858.5),
        "banner": (1081.0, 1800.0, 192.0, 240.0),
    },
]


# =============================================================================
# Matplotlib typography
# =============================================================================

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = list(FONT_STACK)


# =============================================================================
# Slide coordinate helpers
# =============================================================================

def px_x(value_px: float) -> float:
    """Convert slide x-coordinate in pixels to figure-relative x."""
    return value_px / CANVAS_W


def px_y(value_px: float) -> float:
    """Convert slide y-coordinate from top-left pixels to figure y."""
    return 1.0 - (value_px / CANVAS_H)


# =============================================================================
# Date range
# =============================================================================

today = pd.Timestamp.today().normalize()

if YEAR is not None:
    start_date = pd.Timestamp(f"{YEAR}-01-01")
    end_date = pd.Timestamp(f"{YEAR}-12-31")
else:
    if INCLUDE_CURRENT_MONTH:
        end_date = today
        start_date = today.to_period("M").start_time - pd.DateOffset(months=11)
    else:
        end_date = today.replace(day=1) - pd.Timedelta(days=1)
        start_date = end_date.to_period("M").start_time - pd.DateOffset(months=11)

download_start_date = start_date - pd.Timedelta(days=DOWNLOAD_LOOKBACK_DAYS)
download_end_date = min(end_date, today)

if download_start_date >= download_end_date:
    raise ValueError(
        "The calculated download start date must be earlier than the "
        "calculated download end date."
    )


# =============================================================================
# MSCI data download
# =============================================================================

print()
print("Downloading MSCI Net Total Return index data...")
print(
    f"Requested period: {start_date.strftime('%Y-%m-%d')} to "
    f"{end_date.strftime('%Y-%m-%d')}"
)
print(
    f"Requesting {len(ALL_INDEX_CODES)} unique MSCI index code(s) "
    "in one batched request."
)

download_start_str = download_start_date.strftime("%Y-%m-%d")
download_end_str = download_end_date.strftime("%Y-%m-%d")

download_t0 = time.perf_counter()

try:
    raw_data = msci.get_levels(
        ALL_INDEX_CODES,
        download_start_str,
        download_end_str,
        MSCI_RETURN_VARIANT,
    )
except Exception as error:
    raise RuntimeError(
        f"The batched MSCI data request failed. Original error: {error}"
    ) from error

download_seconds = time.perf_counter() - download_t0

if raw_data is None:
    raise ValueError("No data returned from MSCI.")

hist = raw_data.copy() if isinstance(raw_data, pd.DataFrame) else pd.DataFrame(raw_data)

if hist.empty:
    raise ValueError("MSCI returned an empty dataset.")

print(f"MSCI download completed in {download_seconds:.2f}s")
print(f"Raw observations returned: {len(hist):,}")


# =============================================================================
# Data preparation
# =============================================================================

hist.columns = [str(column).strip().upper() for column in hist.columns]

required_columns = {"INDEX_CODE", "DATE", "LEVEL"}
missing_columns = required_columns - set(hist.columns)

if missing_columns:
    raise ValueError(
        "MSCI data is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

hist["INDEX_CODE"] = (
    hist["INDEX_CODE"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

hist["DATE"] = pd.to_datetime(
    hist["DATE"],
    errors="coerce",
    utc=True,
).dt.tz_convert(None)

hist["LEVEL"] = pd.to_numeric(hist["LEVEL"], errors="coerce")

hist = (
    hist
    .dropna(subset=["DATE", "LEVEL"])
    .sort_values(["INDEX_CODE", "DATE"])
    .drop_duplicates(["INDEX_CODE", "DATE"], keep="last")
    .reset_index(drop=True)
)

if hist.empty:
    raise ValueError("MSCI data is empty after normalising.")

missing_codes = sorted(set(ALL_INDEX_CODES) - set(hist["INDEX_CODE"]))

for missing_code in missing_codes:
    print(f"No MSCI history returned for code {missing_code}.")


# =============================================================================
# Ratio construction and log-trend fit
# =============================================================================

def get_level_series(index_code: str) -> pd.Series:
    return (
        hist.loc[hist["INDEX_CODE"] == index_code, ["DATE", "LEVEL"]]
        .set_index("DATE")["LEVEL"]
    )


def fit_log_trend(dates, values):
    x_days = mdates.date2num(dates)
    y_values = np.asarray(values, dtype=float)
    usable = ~np.isnan(y_values) & (y_values > 0)

    if usable.sum() < 2 or x_days[usable][-1] == x_days[usable][0]:
        return 0.0, None

    x_origin = x_days[usable][0]
    slope_per_day, intercept = np.polyfit(
        x_days[usable] - x_origin,
        np.log(y_values[usable]),
        1,
    )

    annual_pct = float(np.expm1(slope_per_day * 365.25) * 100)
    fitted_values = np.exp(intercept + slope_per_day * (x_days - x_origin))

    return annual_pct, fitted_values


ratio_frame = pd.DataFrame()
ratio_columns: list[str] = []
ratio_summary: dict[str, dict[str, float]] = {}

for cap_code, equal_code, series_label, _ in SERIES_DEFINITIONS:
    cap_series = get_level_series(cap_code)
    equal_series = get_level_series(equal_code)

    if cap_series.empty or equal_series.empty:
        print(f"Skipping {series_label}: missing cap-weighted or equal-weighted data.")
        continue

    combined = (
        pd.concat([cap_series, equal_series], axis=1, keys=["Cap_Weighted", "Equal_Weighted"])
        .sort_index()
        .ffill()
        .dropna()
    )

    combined = combined.loc[
        (combined.index >= start_date) & (combined.index <= end_date)
    ]

    if combined.empty:
        print(f"Skipping {series_label}: no data in the selected period.")
        continue

    combined["Ratio"] = combined["Cap_Weighted"] / combined["Equal_Weighted"]

    ratio_frame[series_label] = combined["Ratio"]
    ratio_columns.append(series_label)

    trend_pct, _ = fit_log_trend(combined.index, combined["Ratio"])

    ratio_summary[series_label] = {
        "start_value": float(combined["Ratio"].iloc[0]),
        "end_value": float(combined["Ratio"].iloc[-1]),
        "trend_pct": float(trend_pct),
    }

if ratio_frame.empty:
    raise ValueError("There is no ratio data available for the selected period.")

ratio_frame = ratio_frame.sort_index()

if SAVE_CSV:
    ratio_frame.to_csv(CSV_FILE, index=True)
    print(f"CSV saved to:\n{CSV_FILE}")


# =============================================================================
# Narrative
# =============================================================================

def classify_breadth(trend_pct: float) -> str:
    if trend_pct > NEUTRAL_BAND_PCT:
        return "concentrated"

    if trend_pct < -NEUTRAL_BAND_PCT:
        return "broad-based"

    return "balanced"


def format_ratio(value: float) -> str:
    if abs(value) < 0.5:
        return f"{value:.3f}x"

    return f"{value:.2f}x"


def join_with_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]

    return ", ".join(items[:-1]) + " and " + items[-1]


TREND_WORD = {
    "concentrated": "trended upward",
    "broad-based": "trended downward",
    "balanced": "was broadly flat",
}

period_phrase = f"in {YEAR}" if YEAR is not None else "over the last 12 months"


def prominence_phrase(labels):
    if len(labels) < 2:
        return ""

    ranked = sorted(
        labels,
        key=lambda label: abs(ratio_summary[label]["trend_pct"]),
        reverse=True,
    )
    strongest, weakest = ranked[0], ranked[-1]

    strongest_size = abs(ratio_summary[strongest]["trend_pct"])
    weakest_size = max(abs(ratio_summary[weakest]["trend_pct"]), 1e-9)
    steepness = strongest_size / weakest_size

    if steepness >= PROMINENCE_MUCH:
        adverb = "much more so"
    elif steepness >= PROMINENCE_MORE:
        adverb = "more so"
    else:
        return ", to a similar degree in both markets"

    return f", {adverb} in {strongest} than in {weakest}"


breadth_status = {
    label: classify_breadth(ratio_summary[label]["trend_pct"])
    for label in ratio_columns
}
start_values = {label: ratio_summary[label]["start_value"] for label in ratio_columns}
end_values = {label: ratio_summary[label]["end_value"] for label in ratio_columns}

unique_statuses = set(breadth_status.values())
both_markets = " in both markets" if len(ratio_columns) > 1 else ""

if len(unique_statuses) == 1:
    status = unique_statuses.pop()

    detail = join_with_and(
        [
            f"{'from ' if position == 0 else ''}"
            f"{format_ratio(start_values[label])} to "
            f"{format_ratio(end_values[label])} in {label}"
            for position, label in enumerate(ratio_columns)
        ]
    )

    observation = (
        f"the cap-weighted to equal-weighted ratio "
        f"{TREND_WORD[status]}{both_markets}, {detail}"
    )

    if status == "concentrated":
        conclusion = (
            "growth has been concentrated in the largest companies "
            f"({period_phrase}{prominence_phrase(ratio_columns)})"
        )
    elif status == "broad-based":
        conclusion = (
            "growth has been broad-based across companies "
            f"({period_phrase}{prominence_phrase(ratio_columns)})"
        )
    else:
        conclusion = (
            "growth has not been clearly concentrated or broad-based "
            f"({period_phrase}), with large and average companies broadly in line"
        )
else:
    parts = [
        f"{TREND_WORD[breadth_status[label]]} in {label} "
        f"(from {format_ratio(start_values[label])} to {format_ratio(end_values[label])})"
        for label in ratio_columns
    ]

    observation = (
        "the cap-weighted to equal-weighted ratio "
        f"({period_phrase}): " + " but ".join(parts)
    )

    status_phrases = []

    for status_name in ("concentrated", "broad-based", "balanced"):
        status_labels = [
            label for label in ratio_columns if breadth_status[label] == status_name
        ]

        if status_labels:
            status_phrases.append(f"{status_name} in {join_with_and(status_labels)}")

    conclusion = "growth has been mixed: " + " but ".join(status_phrases)

subtitle = f"We can see {observation}, and hence {conclusion}."

headline_text = SLIDE_TITLE
subtitle_lines = [subtitle]


# =============================================================================
# Figure
# =============================================================================

number_of_panels = min(len(ratio_columns), len(PANEL_GEOMETRY))

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
fig.patch.set_facecolor(BG)


def get_renderer():
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig._get_renderer()


# =============================================================================
# Slope callout
# =============================================================================

def format_slope(annual_pct):
    if abs(annual_pct) < 0.05:
        return "0.0%"

    if abs(annual_pct) < 10:
        return f"{annual_pct:+.1f}%"

    return f"{annual_pct:+.0f}%"


def add_slope_callout(ax, dates, values, fitted_values, annual_pct):
    x_num = mdates.date2num(dates)
    values = np.asarray(values, dtype=float)
    usable = ~np.isnan(values)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    def to_axes(x_data, y_data):
        return np.column_stack(
            (
                (x_data - x_lo) / (x_hi - x_lo),
                (y_data - y_lo) / (y_hi - y_lo),
            )
        )

    sample_x = np.linspace(x_num[0], x_num[-1], 600)
    series_pts = to_axes(
        sample_x,
        np.interp(sample_x, x_num[usable], values[usable]),
    )
    trend_pts = to_axes(
        sample_x,
        np.interp(sample_x, x_num, fitted_values),
    )

    pad = 0.55
    label = f"Slope: {format_slope(annual_pct)}{SLOPE_UNIT_SUFFIX}"

    callout = ax.text(
        0.5,
        0.5,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=CALLOUT_SIZE,
        fontweight="bold",
        color=BANNER_TEXT,
        zorder=7,
        bbox=dict(
            boxstyle=f"round,pad={pad},rounding_size=0.9",
            facecolor=TREND,
            edgecolor="none",
        ),
    )

    axes_box = ax.get_window_extent()
    text_box = callout.get_window_extent(get_renderer())

    pad_px = pad * CALLOUT_SIZE * fig.dpi / 72
    box_w = (text_box.width + 2 * pad_px) / axes_box.width
    box_h = (text_box.height + 2 * pad_px) / axes_box.height

    axes_px = np.array([axes_box.width, axes_box.height])

    def points_inside(points, cx, cy, margin=0.02):
        return int(
            (
                (points[:, 0] >= cx - box_w / 2 - margin)
                & (points[:, 0] <= cx + box_w / 2 + margin)
                & (points[:, 1] >= cy - box_h / 2 - margin)
                & (points[:, 1] <= cy + box_h / 2 + margin)
            ).sum()
        )

    def densify(pts, step_px=1.5):
        pieces = []

        for p0, p1 in zip(pts[:-1], pts[1:]):
            length_px = np.hypot(*((p1 - p0) * axes_px))
            n_steps = max(int(np.ceil(length_px / step_px)), 1)
            pieces.append(
                p0 + (p1 - p0) * np.linspace(0, 1, n_steps, endpoint=False)[:, None]
            )

        pieces.append(pts[-1:])

        return np.vstack(pieces)

    series_dense_px = densify(to_axes(x_num[usable], values[usable])) * axes_px

    clearance_px = int(round(ARROW_CLEARANCE_PX))
    grid_w = int(np.ceil(axes_px[0])) + 1
    grid_h = int(np.ceil(axes_px[1])) + 1

    series_raster = np.zeros((grid_w, grid_h), dtype=np.int32)
    series_x = np.clip(np.round(series_dense_px[:, 0]).astype(int), 0, grid_w - 1)
    series_y = np.clip(np.round(series_dense_px[:, 1]).astype(int), 0, grid_h - 1)
    series_raster[series_x, series_y] = 1

    def widen(grid, radius):
        window = 2 * radius + 1
        padded = np.pad(grid, ((radius, radius), (0, 0)))
        running = np.cumsum(padded, axis=0, dtype=np.int32)
        running = np.vstack((np.zeros((1, grid.shape[1]), np.int32), running))

        return (running[window:] - running[:-window]) > 0

    blocked = widen(
        widen(series_raster, clearance_px).astype(np.int32).T,
        clearance_px,
    ).T

    def arrow_endpoints(cx, cy, tx, ty):
        dx, dy = tx - cx, ty - cy
        leave_at = min(
            (box_w / 2) / abs(dx) if dx else np.inf,
            (box_h / 2) / abs(dy) if dy else np.inf,
        )

        return np.array([cx + leave_at * dx, cy + leave_at * dy]), np.array([tx, ty])

    def arrow_hits(start_pt, end_pt):
        length_px = np.hypot(*((end_pt - start_pt) * axes_px))
        n_samples = max(int(length_px / 2), 8)
        t_values = np.linspace(0, 1, n_samples)
        t_values = t_values[t_values < 1 - 12.0 / max(length_px, 1.0)]
        samples = (start_pt + (end_pt - start_pt) * t_values[:, None]) * axes_px
        cells_x = np.clip(np.round(samples[:, 0]).astype(int), 0, grid_w - 1)
        cells_y = np.clip(np.round(samples[:, 1]).astype(int), 0, grid_h - 1)

        return int(blocked[cells_x, cells_y].sum())

    target_offsets = (-0.08, -0.16, -0.24, 0.0, 0.08, 0.16, 0.24)
    best = None

    for cy in np.linspace(box_h / 2 + 0.04, 1 - box_h / 2 - 0.04, 9):
        for cx in np.linspace(box_w / 2 + 0.03, 1 - box_w / 2 - 0.03, 7):
            fill_overlap = 10 * (
                points_inside(series_pts, cx, cy)
                + points_inside(trend_pts, cx, cy)
            )

            for offset in target_offsets:
                target_x = float(np.clip(cx + offset, 0.04, 0.96))
                target_y = float(
                    np.interp(target_x, trend_pts[:, 0], trend_pts[:, 1])
                )

                gap = np.hypot(
                    max(abs(target_x - cx) - box_w / 2, 0.0),
                    max(abs(target_y - cy) - box_h / 2, 0.0),
                )

                start_pt, end_pt = arrow_endpoints(cx, cy, target_x, target_y)

                score = (
                    fill_overlap
                    + ARROW_OVERLAP_WEIGHT * arrow_hits(start_pt, end_pt)
                    + gap
                    + (50 if gap < 0.07 else 0)
                    + 0.25 * abs(offset + 0.08)
                )

                if best is None or score < best[0]:
                    best = (score, cx, cy, target_x, target_y)

    _, cx, cy, target_x, target_y = best

    callout.set_position((cx, cy))

    start, end = arrow_endpoints(cx, cy, target_x, target_y)

    head_px = 26.0
    direction_px = (end - start) * axes_px
    head_start = end - (direction_px / np.hypot(*direction_px) * head_px) / axes_px

    ax.add_artist(
        FancyArrowPatch(
            tuple(start),
            tuple(head_start),
            transform=ax.transAxes,
            arrowstyle="-",
            linestyle=(0, (1.0, 1.7)),
            linewidth=2.0,
            capstyle="round",
            color=TREND,
            shrinkA=5,
            shrinkB=0,
            zorder=6,
        )
    )

    ax.add_artist(
        FancyArrowPatch(
            tuple(head_start),
            tuple(end),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=30,
            linewidth=2.0,
            capstyle="round",
            joinstyle="round",
            color=TREND,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


# =============================================================================
# Overlay (banners only)
# =============================================================================

overlay = fig.add_axes([0, 0, 1, 1], zorder=0)
overlay.set_xlim(0, CANVAS_W)
overlay.set_ylim(CANVAS_H, 0)
overlay.axis("off")


# =============================================================================
# Text measurement / wrapping
# =============================================================================

def measure_width_px(text, font_size, font_style="normal"):
    probe = fig.text(0, 0, text, fontsize=font_size, style=font_style)
    width = probe.get_window_extent(get_renderer()).width
    probe.remove()

    return width * CANVAS_W / fig.bbox.width


def wrap_to_margins(text, font_size, max_width_px, font_style="normal"):
    text = text.replace(" / ", "\u00a7/\u00a7")
    words = text.split()

    lines = []
    current = ""

    for word in words:
        if not current:
            current = word
            continue

        candidate = f"{current} {word}"

        if measure_width_px(candidate.replace("\u00a7", " "), font_size, font_style) > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        lines.append(current)

    return [line.replace("\u00a7", " ") for line in lines]


# =============================================================================
# Title
# =============================================================================

fig.text(
    px_x(LEFT_MARGIN_PX),
    px_y(TITLE_BASELINE_PX),
    headline_text,
    fontsize=TITLE_SIZE,
    fontweight="bold",
    color=INK,
    ha="left",
    va="baseline",
)


# =============================================================================
# Subtitle
# =============================================================================

subtitle_font_size = SUBTITLE_SIZE
wrapped_subtitle = wrap_to_margins(
    " ".join(subtitle_lines),
    subtitle_font_size,
    RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
)

while len(wrapped_subtitle) > len(SUBTITLE_BASELINES_PX) and subtitle_font_size > 11:
    subtitle_font_size -= 0.5
    wrapped_subtitle = wrap_to_margins(
        " ".join(subtitle_lines),
        subtitle_font_size,
        RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
    )

for line, baseline in zip(wrapped_subtitle, SUBTITLE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX),
        px_y(baseline),
        line,
        fontsize=subtitle_font_size,
        color=SUBTITLE_INK,
        ha="left",
        va="baseline",
    )


# =============================================================================
# Source
# =============================================================================

fig.text(
    px_x(LEFT_MARGIN_PX),
    px_y(SOURCE_BASELINE_PX),
    SOURCE_TEXT,
    fontsize=FOOT_SIZE,
    style="italic",
    color=FOOT,
    ha="left",
    va="baseline",
)


# =============================================================================
# Note
# =============================================================================

note_font_size = FOOT_SIZE
wrapped_note = wrap_to_margins(
    NOTE_TEXT,
    note_font_size,
    RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
    "italic",
)

while len(wrapped_note) > len(NOTE_BASELINES_PX) and note_font_size > 8:
    note_font_size -= 0.5
    wrapped_note = wrap_to_margins(
        NOTE_TEXT,
        note_font_size,
        RIGHT_MARGIN_PX - LEFT_MARGIN_PX,
        "italic",
    )

for line, baseline in zip(wrapped_note, NOTE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX),
        px_y(baseline),
        line,
        fontsize=note_font_size,
        style="italic",
        color=FOOT,
        ha="left",
        va="baseline",
    )


# =============================================================================
# Plot panels
# =============================================================================

for series_label, geometry in zip(ratio_columns[:number_of_panels], PANEL_GEOMETRY):
    series_colour = SERIES_COLOURS[series_label]

    axes_left, axes_right, axes_top, axes_bottom = geometry["axes"]

    axes = fig.add_axes(
        [
            px_x(axes_left),
            px_y(axes_bottom),
            px_x(axes_right - axes_left),
            (axes_bottom - axes_top) / CANVAS_H,
        ],
        zorder=2,
        facecolor="none",
    )

    series_data = ratio_frame[series_label].dropna()
    plot_dates = series_data.index
    plot_values = series_data.to_numpy(dtype=float)

    # Ratio line
    axes.plot(
        plot_dates,
        plot_values,
        color=series_colour,
        linewidth=2.6,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )

    # Trend line
    slope_pct, trend_values = fit_log_trend(plot_dates, plot_values)

    if trend_values is not None:
        axes.plot(
            plot_dates,
            trend_values,
            color=TREND,
            linewidth=2.4,
            linestyle="--",
            zorder=5,
        )

    # Grid
    axes.set_axisbelow(True)

    axes.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=1,
        color=GRID,
        zorder=0,
    )

    # Spines
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    axes.spines["left"].set_color(AXIS)
    axes.spines["bottom"].set_color(AXIS)

    axes.spines["left"].set_linewidth(1.5)
    axes.spines["bottom"].set_linewidth(1.5)

    # X axis
    axes.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    axes.tick_params(
        axis="x",
        which="major",
        labelsize=TICK_SIZE,
        length=5,
        width=1.2,
        pad=8,
        colors=INK,
    )

    for tick in axes.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    # Y axis
    axes.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))

    axes.tick_params(
        axis="y",
        which="major",
        length=0,
        pad=10,
        labelsize=TICK_SIZE,
        colors=INK,
    )

    # Y label - first panel only
    if series_label == ratio_columns[0]:
        axes.set_ylabel(
            "Ratio (Cap Weighted / Equal Weighted)",
            fontsize=TICK_SIZE,
            color=INK,
            labelpad=16,
        )

    # X limits
    axes.set_xlim(
        plot_dates[0],
        plot_dates[-1] + pd.Timedelta(days=10),
    )

    # Slope callout
    if trend_values is not None:
        add_slope_callout(axes, plot_dates, plot_values, trend_values, slope_pct)

    # Banner
    banner_left, banner_right, banner_top, banner_bottom = geometry["banner"]

    banner = FancyBboxPatch(
        (banner_left, banner_top),
        banner_right - banner_left,
        banner_bottom - banner_top,
        boxstyle=f"round,pad=0,rounding_size={BANNER_RADIUS_PX}",
        facecolor=series_colour,
        edgecolor="none",
        linewidth=0,
        mutation_aspect=1,
    )

    overlay.add_patch(banner)

    overlay.text(
        (banner_left + banner_right) / 2,
        BANNER_TEXT_BASELINE_PX,
        series_label,
        ha="center",
        va="baseline",
        fontsize=BANNER_SIZE,
        fontweight="bold",
        color=BANNER_TEXT,
    )


# =============================================================================
# Display
# =============================================================================

plt.show()
