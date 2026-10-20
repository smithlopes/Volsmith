from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import os
import ssl
import urllib.request
from pathlib import Path
from tempfile import gettempdir

import certifi
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fredapi import Fred
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


# =============================================================================
# Configuration
# =============================================================================

# Hard-coded lookup period (yyyy-mm-dd)
START_DATE = "2021-01-01"
END_DATE = None

# Treasury constant-maturity series on FRED (daily, percent)
TENORS = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
    "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
    "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}
TRADING_DAYS = 252
MIN_OBS = 20         # minimum daily changes needed to report a vol
RUN_DATE = pd.Timestamp.today().normalize()
RECENT_START = RUN_DATE - pd.DateOffset(years=1)

SLIDE_TITLE = "Term Structure of Interest Rate Volatility"
SOURCE_TEXT = "Source: Federal Reserve (H.15 Treasury constant maturities) via FRED"
NOTE_TEXT = (
    "Note: Realized volatility is the standard deviation of daily changes in US "
    "Treasury constant-maturity yields (bp), annualized by the square root of 252, "
    "over {period}; Recent is the latest one year. It is backward-looking, not "
    "implied (MOVE) volatility. Short-end volatility mainly reflects uncertainty "
    "around monetary policy, while long-end volatility reflects uncertainty around "
    "the real economy and inflation."
)

OUTPUT_DIR = Path(__file__).resolve().parent
PNG_FILE = OUTPUT_DIR / "IR_Vol_Term_Structure.png"
SAVE_PNG = False

# Slide canvas: change CANVAS_H only (extra height goes to the plot area)
CANVAS_W = 1920
CANVAS_H = 1080
FIG_DPI = 150
FIG_W_IN = CANVAS_W / 120
FIG_H_IN = CANVAS_H / 120
PX_TO_PT = 0.6  # 72 pt/in divided by 120 px/in

LEFT_MARGIN_PX = 60
RIGHT_MARGIN_PX = 1860

BLUE = "#6b8fd1"
ORANGE = "#e8856a"
BG, INK, SUBTITLE_INK = "#ffffff", "#000000", "#000000"
AXIS, GRID, FOOT = "#7f7f7f", "#b9b9b9", "#8c8c8c"
FONT_STACK = ("Inter", "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans")

# CSS px sizes from the spec converted to points
TITLE_SIZE = 47.56 * PX_TO_PT      # 28.536
SUBTITLE_SIZE = 20.23 * PX_TO_PT   # 12.138
FOOT_SIZE = 16 * PX_TO_PT          # 9.6
TICK_SIZE = 10.4

# CSS tops from the spec
TITLE_TOP_PX = 40
SUBTITLE_TOP_PX = 103
SOURCE_TOP_PX = CANVAS_H - 102
NOTE_TOP_PX = CANVAS_H - 78
CHART_AREA_TOP_PX = 173
CHART_AREA_BOTTOM_PX = CANVAS_H - 124

# CSS tops converted to Matplotlib baselines (Inter line-box offsets)
TITLE_BASELINE_PX = TITLE_TOP_PX + 43
SUBTITLE_BASELINES_PX = (SUBTITLE_TOP_PX + 21, SUBTITLE_TOP_PX + 45)
SOURCE_BASELINE_PX = SOURCE_TOP_PX + 15
NOTE_BASELINES_PX = (NOTE_TOP_PX + 15, NOTE_TOP_PX + 35)

# Y-axis title placement (px)
YLABEL_WIDTH_PX = 18   # approx. height of the rotated y-axis title text
YLABEL_GAP_PX = 80     # distance from y-axis to the y-axis title (as in the earlier version)


# =============================================================================
# FRED SSL and API setup
# =============================================================================

API_KEY = os.environ["FRED_API_KEY"]
CA_BUNDLE = certifi.where()
ssl_context = ssl.create_default_context(cafile=CA_BUNDLE)
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
)
fred = Fred(api_key=API_KEY)


# =============================================================================
# Slide coordinate, font and text helpers
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


def wrap_to_margins(text: str, font_size: float, max_width_px: float,
                    font_style: str = "normal") -> list[str]:
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


def fit_sentences(sentences: list[str], font_size: float, max_width_px: float,
                  max_lines: int) -> list[str]:
    """Add sentences in priority order while the wrapped text stays within max_lines."""
    kept = ""
    for sentence in sentences:
        candidate = f"{kept} {sentence}".strip()
        if len(wrap_to_margins(candidate, font_size, max_width_px)) <= max_lines:
            kept = candidate
        else:
            break
    if not kept:
        raise ValueError("Even the first commentary sentence exceeds the line limit.")
    return wrap_to_margins(kept, font_size, max_width_px)


# =============================================================================
# Download yields and compute realized vol by tenor
# =============================================================================

start_date = pd.Timestamp(START_DATE)
end_date = pd.Timestamp(END_DATE) if END_DATE is not None else RUN_DATE
if start_date > end_date:
    raise ValueError("START_DATE must be on or before END_DATE.")

yields: dict[str, pd.Series] = {}
for label, series_id in TENORS.items():
    print(f"Downloading FRED series {series_id} ({label})...")
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    yields[label] = pd.to_numeric(s, errors="coerce").dropna().sort_index()

windows = {
    k: v.loc[(v.index >= start_date) & (v.index <= end_date)] for k, v in yields.items()
}
non_empty = [w for w in windows.values() if not w.empty]
if not non_empty:
    raise ValueError("No FRED observations in the selected dates.")
first_obs = min(w.index[0] for w in non_empty)
last_obs = max(w.index[-1] for w in non_empty)

full_vol: dict[str, float] = {}
recent_vol: dict[str, float] = {}
obs_count: dict[str, int] = {}
for label, window in windows.items():
    changes_bp = window.diff().dropna() * 100.0  # percent -> bp
    obs_count[label] = len(changes_bp)
    if len(changes_bp) >= MIN_OBS:
        full_vol[label] = float(changes_bp.std(ddof=1) * np.sqrt(TRADING_DAYS))
    recent_changes = changes_bp.loc[changes_bp.index >= RECENT_START]
    if len(recent_changes) >= MIN_OBS:
        recent_vol[label] = float(recent_changes.std(ddof=1) * np.sqrt(TRADING_DAYS))

if not full_vol:
    raise ValueError("No tenor has enough observations in the selected dates.")

vol_full = pd.Series(full_vol).reindex(TENORS.keys())
vol_recent = pd.Series(recent_vol).reindex(TENORS.keys())

print(f"Window: {first_obs:%Y-%m-%d} to {last_obs:%Y-%m-%d}")
print(pd.DataFrame(
    {"full_bp": vol_full, "recent_bp": vol_recent, "obs": pd.Series(obs_count)}
).round(1))

NOTE_TEXT = NOTE_TEXT.format(period=f"{first_obs:%d %b %Y} to {last_obs:%d %b %Y}")

# =============================================================================
# Analyst commentary (dynamic, fitted to two lines)
# =============================================================================

peak = vol_full.idxmax()
trough = vol_full.idxmin()
peak_v, trough_v = float(vol_full[peak]), float(vol_full[trough])

if peak in ("1M", "3M", "6M", "1Y"):
    shape = "front-end led"
elif peak in ("2Y", "3Y", "5Y"):
    shape = "belly-led"
else:
    shape = "long-end led"


def have(series: pd.Series, *keys: str) -> bool:
    return all(k in series.index and pd.notna(series[k]) for k in keys)


sentences = [
    f"From {first_obs:%b %Y} to {last_obs:%b %Y}, Treasury rate volatility was {shape}, "
    f"peaking at the {peak} tenor ({peak_v:.0f}bp annualized) and lowest at {trough} "
    f"({trough_v:.0f}bp)."
]

if have(vol_full, "2Y", "10Y", "30Y"):
    spread = vol_full["2Y"] - vol_full["10Y"]
    direction = "above" if spread >= 0 else "below"
    sentences.append(
        f"2Y vol of {vol_full['2Y']:.0f}bp sits {abs(spread):.0f}bp {direction} 10Y "
        f"({vol_full['10Y']:.0f}bp), while 30Y vol is {vol_full['30Y']:.0f}bp."
    )

if have(vol_recent, "10Y") and have(vol_full, "10Y"):
    r10, f10 = float(vol_recent["10Y"]), float(vol_full["10Y"])
    pct = (r10 / f10 - 1) * 100
    regime = "above" if pct >= 0 else "below"
    sentence = (
        f"Over the latest year, 10Y vol was {r10:.0f}bp, {abs(pct):.0f}% {regime} "
        f"the period average"
    )
    if have(vol_recent, "2Y") and have(vol_full, "2Y"):
        r2, f2 = float(vol_recent["2Y"]), float(vol_full["2Y"])
        sentence += f", and 2Y vol was {r2:.0f}bp ({(r2 / f2 - 1) * 100:+.0f}%)"
    sentences.append(sentence + ".")

wrapped_subtitle = fit_sentences(
    sentences, SUBTITLE_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, len(SUBTITLE_BASELINES_PX)
)


# =============================================================================
# Title, subtitle, source and note
# =============================================================================

title_font = FontProperties(fname=str(inter_extrabold)) if inter_extrabold.exists() else None
fig.text(
    px_x(LEFT_MARGIN_PX), px_y(TITLE_BASELINE_PX), SLIDE_TITLE,
    fontsize=TITLE_SIZE, fontproperties=title_font,
    fontweight="bold" if title_font is None else "normal",
    color=INK, ha="left", va="baseline",
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

wrapped_note = wrap_to_margins(NOTE_TEXT, FOOT_SIZE, RIGHT_MARGIN_PX - LEFT_MARGIN_PX, "italic")
if len(wrapped_note) > len(NOTE_BASELINES_PX):
    raise ValueError(f"Note wraps to {len(wrapped_note)} lines; shorten NOTE_TEXT.")
for line, baseline in zip(wrapped_note, NOTE_BASELINES_PX):
    fig.text(
        px_x(LEFT_MARGIN_PX), px_y(baseline), line,
        fontsize=FOOT_SIZE, style="italic", color=FOOT, ha="left", va="baseline",
    )


# =============================================================================
# Chart
# =============================================================================

# Y-axis title sits on the left margin; the axes start YLABEL_GAP_PX to its right,
# and the plot runs out to the right margin.
ylabel_right_px = LEFT_MARGIN_PX + YLABEL_WIDTH_PX
axes_left = float(ylabel_right_px + YLABEL_GAP_PX)   # 158
axes_right = float(RIGHT_MARGIN_PX)                  # 1860
axes_top = CHART_AREA_TOP_PX + 17
axes_bottom = CHART_AREA_BOTTOM_PX - 120
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

x = np.arange(len(TENORS))
ax.plot(
    x, vol_full.to_numpy(dtype=float),
    color=BLUE, linewidth=2.6, marker="o", markersize=7,
    solid_capstyle="round", solid_joinstyle="round", zorder=3,
    label=f"Realized vol, {first_obs:%b %Y} to {last_obs:%b %Y}",
)
if vol_recent.notna().any():
    ax.plot(
        x, vol_recent.to_numpy(dtype=float),
        color=ORANGE, linewidth=2.2, marker="o", markersize=6,
        solid_capstyle="round", solid_joinstyle="round", zorder=3,
        label=f"Recent (1 year to {RUN_DATE:%b %Y})",
    )

ax.set_axisbelow(True)
ax.grid(True, which="major", axis="both", linestyle="--", linewidth=1, color=GRID, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
    ax.spines[side].set_linewidth(1.5)

ax.set_xticks(x)
ax.set_xticklabels(list(TENORS.keys()))
ax.set_xlim(-0.5, len(TENORS) - 0.5)
ax.set_xlabel("Treasury maturity", fontsize=TICK_SIZE, color=INK, labelpad=10)
ax.tick_params(axis="x", which="major", labelsize=TICK_SIZE, length=5, width=1.2, pad=8, colors=INK)

ymax = np.nanmax(np.r_[vol_full.to_numpy(dtype=float), vol_recent.to_numpy(dtype=float)])
ax.set_ylim(0, ymax * 1.15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax.tick_params(axis="y", which="major", length=0, pad=10, labelsize=TICK_SIZE, colors=INK)
ax.set_ylabel("Annualized volatility (bp)", fontsize=TICK_SIZE, color=INK, labelpad=10)
# fixed pixel gap between the y-axis and its title, independent of the axes width
ax.yaxis.set_label_coords(-YLABEL_GAP_PX / (axes_right - axes_left), 0.5)

legend = ax.legend(
    loc="center", ncol=2, frameon=False, fontsize=11,
    handlelength=2.2, columnspacing=1.8, handletextpad=0.7,
)
legend.set_bbox_to_anchor(
    (0.5, px_y(CHART_AREA_BOTTOM_PX - 15)), transform=fig.transFigure,
)


# =============================================================================
# Display / optional save
# =============================================================================

if SAVE_PNG:
    fig.savefig(PNG_FILE, dpi=FIG_DPI, facecolor=BG)
    print(f"Chart saved to:\n{PNG_FILE}")

plt.show()