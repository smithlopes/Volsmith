---
name: smiths-chart-style
description: Smith's house style for every chart and data visualization (line, bar, column, stacked, area, scatter, pie/donut, heatmap, waterfall, multi-panel) made with matplotlib or similar. Use whenever creating, restyling or reviewing a chart, graph, plot or figure, especially finance, P&L, returns, strategy-comparison, derivatives or market charts, so output matches his preferred look - pure white background, dashed grid, pill-shaped end labels, bold centred title hierarchy, soft muted palette, uniform padding on all four sides, and a standard disclaimer footnote.
---

# Smith's Chart Style: Complete Guide

This file is the single source of truth for how Smith likes his charts to look. It contains the design rules, the exact values to use, a reusable helper module (copy-paste ready), worked examples for each chart type, and a pre-delivery checklist. Apply it to **every** chart, whatever the type, library or subject.

The look in one sentence: **a clean white canvas, soft muted colours, dashed light-grey gridlines, a bold centred title with a lighter subtitle, direct pill-shaped value labels at the end of each series, a bold zero line, a disclaimer footnote, and generous equal padding on all four sides.**

---

## 1. Non-negotiables

These apply to every chart. If a request conflicts with one of them, follow the request but keep the rest.

1. **Background is pure white `#ffffff`** on the figure *and* the axes. Pass `facecolor` to `savefig` as well, otherwise exports can come out transparent or grey.
2. **Gridlines are dashed (`--`)**, light grey `#d0d0d0`, 1px, always drawn *behind* the data (`zorder=0` and `ax.set_axisbelow(True)`).
3. **Equal padding on all four sides of the saved image.** Always save with `bbox_inches="tight"` and `pad_inches=0.6`. Title, y-label, footnote and end-labels must never touch or clip at the image edge.
4. **No chart junk.** Hide top and right spines. No chart border, no 3D, no gradients, no shadows, no gradient fills, no default matplotlib blue/orange cycle.
5. **Title block is centred and bold**, subtitle is centred and regular weight. Never left-align the main title.
6. **Footnote is mandatory** whenever a specific security, company, fund or index is named: `The securities are quoted as an example and not as a recommendation.` Place it bottom-left, aligned with the left edge of the plot.
7. **Direct labelling over legends** where practical: end-of-series pill labels for line charts, value labels on bars, labels beside donut slices.
8. **Muted palette only** (section 4). Never pure red, pure green, neon, or the matplotlib defaults.

---

## 2. Canvas and layout

| Property | Value |
|---|---|
| Figure size | 16 x 9 inches (landscape) |
| DPI | 150 (use 200+ if the chart will be projected or printed) |
| Background | `#ffffff` (figure and axes) |
| Output format | PNG by default; SVG/PDF if asked |
| Saved padding | `pad_inches=0.6` with `bbox_inches="tight"` |
| Plot area margins (before tight crop) | left 0.10, right 0.84, top 0.80, bottom 0.24 |

The right margin is intentionally large (0.84) to leave room for the pill end-labels. For charts without end-labels (bar, heatmap) it can be widened to about 0.92.

### Vertical rhythm (figure coordinates)

- Title: y = 0.94
- Subtitle: y = 0.90
- Plot area: from y = 0.24 (bottom) to y = 0.80 (top)
- Legend: just below the x-axis label (`bbox_to_anchor=(0.5, -0.14)` relative to axes)
- Footnote: y = 0.06, x = 0.10

---

## 3. Typography

Font stack: `["Inter", "Helvetica", "Arial", "DejaVu Sans"]`. Inter is preferred; fall back silently if it is not installed (matplotlib will log `findfont` warnings, which are harmless).

| Element | Size | Weight | Colour |
|---|---|---|---|
| Title | 26pt | Bold | `#111111` |
| Subtitle | 14pt | Regular | `#222222` |
| Axis labels | 13-14pt | Regular | `#222222` |
| Tick labels | 13pt | Regular (bold for the `0` / `0%` tick) | `#222222` |
| Legend text | 14pt | Regular | `#222222` |
| End-of-series pill labels | 16pt | Bold | white or black (see section 7) |
| Value labels on bars | 13-14pt | Bold | `#222222` or white if inside a dark bar |
| Footnote | 10.5pt | Regular | `#222222` |

### Title and subtitle conventions

- Title states the comparison or message in one line, Title Case: `Reliance: Buy & Hold vs Monthly Covered Call`.
- Subtitle carries period, parameters and metric, separated by two spaces, a pipe, and two spaces: `Jan 2022 – Aug 2026  |  5% OTM Call  |  Cumulative P&L`.
- Use an en dash (`–`) for ranges and for negative numbers on axes, never a plain hyphen.
- Keep the title to a single line at 26pt (about 50 characters maximum on a 16-inch canvas). Shorten the wording rather than shrinking the font.

---

## 4. Colour palette

Soft, muted, legible on white. Assign colours by **meaning** and keep them consistent across charts in the same document.

| Role | Name | Hex |
|---|---|---|
| Primary / benchmark / baseline | Blue | `#6b8fd1` |
| Hedge / overlay / income / negative | Orange | `#e8856a` |
| Combined / outcome / positive | Green | `#3d9a7f` |
| Highlight band (events, regimes, breaches) | Soft yellow | `#f5dd8f` |
| Extra series 4 | Lavender | `#9b7fc7` |
| Extra series 5 | Mustard | `#d9a441` |
| Extra series 6 | Teal-blue | `#5aa9c9` |
| Titles | Near-black | `#111111` |
| Text and labels | Charcoal | `#222222` |
| Axis spines | Mid grey | `#888888` |
| Gridlines | Light grey | `#d0d0d0` |

Rules:

- **Maximum 3-4 colours per chart.** With more categories, group the small ones into "Other", or use small multiples.
- **Gains are green, losses are orange.** Do not use a red/green pair (poor for colour-blind readers and visually harsh).
- **Area fills** use the series colour at alpha ~0.22 with no outline.
- **Highlight bands** use `#f5dd8f` at alpha ~0.9, spanning the full plot height, drawn behind all data (`zorder=1`). Where a band overlaps an area fill it naturally blends to a warmer orange, which is the intended look.
- **Sequential scales** (heatmaps): single-hue blue ramp from white. **Diverging scales**: blue to white to orange, centred on zero.
- Neutral/other categories: light grey `#c9c9c9`.

---

## 5. Axes, grid and ticks

- **Spines:** show left and bottom only. Colour `#888888`, linewidth 1.5. Hide top and right.
- **Grid:**
  - Line, area and scatter charts: dashed grid on **both** axes.
  - Bar, column and waterfall charts: dashed grid on the **value axis only**.
  - Heatmaps and pie/donut charts: **no grid**.
  - Style: `color="#d0d0d0", linewidth=1, linestyle="--"`, behind the data.
- **Y ticks:** no tick marks (`length=0`), `pad=10`. Choose a clean step (10%, 5%, 1, 100 and so on) and pad the range by one step above the data. Start the range at the nearest step below the minimum.
- **X ticks:** short marks (`length=5`, `width=1.2`, grey), `pad=8`.
- **Percent axes:** format as `20%`, with an en dash for negatives (`–10%`).
- **Zero line:** if the data crosses zero, draw a solid black 1.8px line at 0 (`zorder=3`) and make the `0` / `0%` tick label **bold**.
- **Date axes:** tick only at meaningful intervals (for example January and July, formatted `Jan 22`). Start the x-range just before the first data point (shift plotted dates back by about 15 days) so the series begins on the y-axis and the first tick label sits slightly to its right. Add roughly 12 days of right padding after the last point so the end-dot is not on the plot edge.
- **Axis labels:** include units, for example `Cumulative P&L (%)`. Use generous `labelpad` (14 for x, 16 for y). The x-label for monthly series can simply be `Month`.

---

## 6. Marks (lines, bars, areas, points)

- **Lines:** 2.6px, solid, no markers, round-ish joins. Plot the headline series last so it sits on top (`zorder=5`; other series `zorder=4`).
- **Area fills:** between the series and zero (or baseline), series colour, alpha 0.22, no edge, `zorder=2`.
- **Bars/columns:** flat fill, no edge, width about 0.6, comfortable gaps, `zorder=3` so they sit above the grid. Positive green, negative orange unless colours encode categories. Single-series bars may use the blue.
- **Stacked bars:** same palette in order blue, orange, green, then extras. White 1.5px separators between segments.
- **Scatter points:** size 70-90, alpha 0.8, white 1px edge. Optional dashed trend line in orange (2px).
- **Donut slices:** `wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2)`, start angle 90, counter-clockwise off. No exploded slices.
- **Heatmap cells:** no gridlines, white 1px cell separators, annotation text bold 12pt, white text on dark cells and charcoal on light cells.
- **Waterfall:** floating bars; increases green, decreases orange, totals blue; thin grey dashed connectors between bars.

---

## 7. Signature element: end-of-series pill labels

The most recognisable part of the style. For each series in a line or area chart:

1. Put a **dot** on the final data point: size 90, series colour, white edge 1.5px, `zorder=6`, `clip_on=False`.
2. Put a **fully rounded pill** label to the right of the dot: offset 26pt right, vertically centred, filled with the series colour, no edge, bold 16pt, signed value rounded to a whole number (`+70%`, `–12%`).
3. Text colour: **white on blue and green, black on orange**.
4. `bbox=dict(boxstyle="round,pad=0.5,rounding_size=1.0", fc=color, ec="none")` and `annotation_clip=False`.
5. Reserve enough right margin that the pill is never clipped (plot right edge at about 0.84 of the figure width).
6. If two end-values are within about 5 percentage points, nudge the labels vertically so pills do not overlap (offset one by +/-14pt).

With end-labels present, the legend is optional. When used, keep it below the plot.

---

## 8. Legend

- Position: below the plot, centred (`loc="upper center", bbox_to_anchor=(0.5, -0.14)`).
- Layout: single horizontal row (`ncol` equals number of series, up to about 4).
- No frame, 14pt text, `handlelength=1.6`, `columnspacing=3`.
- Line swatches thickened to 4px.
- Omit when there is only one series, or when direct labels already identify everything.

---

## 9. Highlighting events and regimes

Use vertical yellow bands for breaches, crisis periods, rate-hike windows, earnings, regime changes and similar.

- Colour `#f5dd8f`, alpha 0.9, `lw=0`, `zorder=1`.
- Width about one period (for monthly data, from 27 days before to 3 days after the period date, so the band aligns with the plotted point after the 15-day shift).
- Keep to a handful of bands (roughly 3-6). If everything is highlighted nothing stands out.
- Explain the highlight in the subtitle or a short note if it is not obvious.

---

## 10. Chart-type playbook

### Line / multi-line (time series, P&L, cumulative returns)
Dashed grid on both axes, 2.6px lines, optional area fill under the key series, zero line if data crosses zero, pill end-labels, highlight bands for events, legend below.

### Bar / column
Dashed grid on y only. Flat bars, width 0.6. Value labels centred above (or below for negatives) each bar, bold. Sort descending unless the order is chronological or categorical. Bold zero line when values are negative.

### Stacked bar / stacked area
Muted palette in fixed order, white separators, total label above each stack in bold, legend below.

### Scatter
Dashed grid both axes, points size 80 alpha 0.8 with white edges, optional dashed orange trend line, label only the notable points (callout text 12pt with no arrow, or a thin grey leader line).

### Donut / pie
Prefer a donut. Muted palette, white 2px separators, labels outside with category and value, large bold total in the centre of the donut, no legend. No grid, no axes.

### Heatmap
No grid. Diverging blue-white-orange centred on zero for returns and correlations; sequential blue for magnitudes. Annotate every cell. Thin colour bar on the right, no outline, tick labels 12pt.

### Waterfall
Starting and ending totals in blue, increases green, decreases orange, dashed grey connectors, value labels above/below bars, y grid only.

### Multi-panel (small multiples)
Same palette and spacing in every panel. Share axes where the scale is comparable. Panel titles left-aligned, bold 14pt. One overall centred title and one footnote for the whole figure.

---

## 11. Reusable helper module

Save this as `chart_style.py` next to your script (or paste it at the top of a notebook). Everything in this guide is implemented here.

```python
"""Smith's chart style helpers (matplotlib)."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Palette --------------------------------------------------------------
BG     = "#ffffff"   # background is ALWAYS white
INK    = "#111111"   # titles
TEXT   = "#222222"   # labels, ticks, subtitle, footnote
AXIS   = "#888888"   # left and bottom spines
GRID   = "#d0d0d0"
BLUE   = "#6b8fd1"
ORANGE = "#e8856a"
GREEN  = "#3d9a7f"
YELLOW = "#f5dd8f"   # highlight bands
LAV, MUSTARD, TEAL = "#9b7fc7", "#d9a441", "#5aa9c9"
NEUTRAL = "#c9c9c9"
SERIES = [BLUE, ORANGE, GREEN, LAV, MUSTARD, TEAL]
POS, NEG = GREEN, ORANGE   # semantic colours: gains / losses

DIVERGING = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, "#ffffff", ORANGE])
SEQUENTIAL = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", BLUE])

DISCLAIMER = "The securities are quoted as an example and not as a recommendation."


def setup(figsize=(16, 9), dpi=150):
    """Create fig, ax with the base look (white background, font stack)."""
    plt.rcParams["font.family"] = ["Inter", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def style_axes(ax, xlabel=None, ylabel=None, grid="both", y_pct=False):
    """grid: 'both' (line/scatter), 'y' (bar/column/waterfall), 'x', or None."""
    if grid:
        ax.grid(True, axis=grid if grid in ("x", "y") else "both",
                color=GRID, lw=1, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.5)
    ax.tick_params(axis="y", colors=TEXT, labelsize=13, length=0, pad=10)
    ax.tick_params(axis="x", colors=TEXT, labelsize=13, length=5, width=1.2,
                   color=AXIS, pad=8)
    if y_pct:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "\u2013")))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, color=TEXT, labelpad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, color=TEXT, labelpad=16)


def zero_line(ax):
    """Solid black zero line (use when data crosses zero)."""
    ax.axhline(0, color="black", lw=1.8, zorder=3)


def bold_zero_tick(fig, ax):
    """Make the 0 / 0% y tick label bold. Call after ticks are final."""
    fig.canvas.draw()
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in ("0", "0%"):
            lbl.set_fontweight("bold")


def highlight_band(ax, x0, x1):
    """Vertical yellow band behind the data."""
    ax.axvspan(x0, x1, color=YELLOW, alpha=0.9, lw=0, zorder=1)


def area_fill(ax, x, y, color, base=0):
    ax.fill_between(x, base, y, color=color, alpha=0.22, lw=0, zorder=2)


def line(ax, x, y, color, label=None, headline=False):
    ax.plot(x, y, color=color, lw=2.6, label=label, zorder=5 if headline else 4)


def end_label(ax, x, y, text, color, text_color="white", offset=26, dy=0):
    """Dot on the last point + fully rounded pill label to its right."""
    ax.scatter(x, y, s=90, color=color, edgecolor="white", lw=1.5,
               zorder=6, clip_on=False)
    ax.annotate(text, xy=(x, y), xytext=(offset, dy), textcoords="offset points",
                va="center", ha="left", fontsize=16, fontweight="bold",
                color=text_color, annotation_clip=False, zorder=7,
                bbox=dict(boxstyle="round,pad=0.5,rounding_size=1.0",
                          fc=color, ec="none"))


def pill_text(color):
    """White text on blue/green, black on orange."""
    return "black" if color == ORANGE else "white"


def titles(fig, title, subtitle=None):
    fig.text(0.5, 0.94, title, ha="center", fontsize=26, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha="center", fontsize=14, color=TEXT)


def bottom_legend(ax, ncol=3):
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol,
                    frameon=False, fontsize=14, handlelength=1.6, columnspacing=3)
    for ln in leg.get_lines():
        ln.set_linewidth(4)
    return leg


def footnote(fig, text=DISCLAIMER):
    fig.text(0.10, 0.06, text, fontsize=10.5, color=TEXT)


def save(fig, path="chart.png", pad_inches=0.6, layout=(0.10, 0.84, 0.80, 0.24)):
    """Apply layout (left, right, top, bottom) and save with equal padding on all sides."""
    left, right, top, bottom = layout
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=pad_inches)
```

---

## 12. Worked examples

All examples assume `from chart_style import *` plus the imports shown.

### 12.1 Line chart with area fill, pill labels and highlight bands

```python
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from chart_style import *

# Hypothetical data
rng = np.random.RandomState(2694)
n = 56
dates = pd.date_range("2022-01-01", periods=n, freq="MS")
ret = rng.normal(0.006, 0.06, n); ret[0] = 0
prem = 0.0125 * np.abs(rng.normal(1, 0.25, n))
call_pnl = prem - np.maximum(ret - 0.05, 0); call_pnl[0] = 0
stock = np.cumsum(ret) * 100
call = np.cumsum(call_pnl) * 100
combo = stock + call
breached = (ret - 0.05) > 0.035
x = dates - pd.Timedelta(days=15)           # first point sits on the y-axis

fig, ax = setup()
for d in dates[breached]:
    highlight_band(ax, d - pd.Timedelta(days=27), d + pd.Timedelta(days=3))
area_fill(ax, x, call, ORANGE)
line(ax, x, stock, BLUE, "Stock (Buy & Hold)")
line(ax, x, call, ORANGE, "Covered Call")
line(ax, x, combo, GREEN, "Stock + Covered Call", headline=True)
zero_line(ax)

for series, color in [(combo, GREEN), (call, ORANGE), (stock, BLUE)]:
    end_label(ax, x[-1], series[-1], f"{series[-1]:+.0f}%".replace("-", "\u2013"),
              color, pill_text(color))

ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=12))
lo = np.floor(min(stock.min(), call.min(), combo.min()) / 10) * 10
hi = np.ceil(max(stock.max(), call.max(), combo.max()) / 10) * 10 + 10
ax.set_ylim(lo, hi); ax.set_yticks(np.arange(lo, hi + 1, 10))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

style_axes(ax, "Month", "Cumulative P&L (%)", grid="both", y_pct=True)
bold_zero_tick(fig, ax)
titles(fig, "Reliance: Buy & Hold vs Monthly Covered Call",
       "Jan 2022 – Aug 2026  |  5% OTM Call  |  Cumulative P&L")
bottom_legend(ax); footnote(fig)
save(fig, "line_chart.png")
```

### 12.2 Bar chart with value labels

```python
from chart_style import *

cats = ["Equities", "Credit", "Rates", "FX", "Commodities Markets"]
vals = [12.4, -5.1, 8.7, 3.2, -2.6]

fig, ax = setup()
bars = ax.bar(cats, vals, width=0.6, zorder=3,
              color=[POS if v >= 0 else NEG for v in vals])
for b, v in zip(bars, vals):
    ax.annotate(f"{v:+.1f}%".replace("-", "\u2013"),
                xy=(b.get_x() + b.get_width() / 2, v),
                xytext=(0, 8 if v >= 0 else -8), textcoords="offset points",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=14, fontweight="bold", color=TEXT)
ax.set_ylim(-10, 16)
style_axes(ax, None, "Return (%)", grid="y", y_pct=True)
zero_line(ax); bold_zero_tick(fig, ax)
titles(fig, "Asset Class Returns", "FY2025  |  Total return  |  USD")
footnote(fig, "Illustrative data only.")
save(fig, "bar_chart.png", layout=(0.10, 0.92, 0.80, 0.24))
```

### 12.3 Donut chart with centre total

```python
from chart_style import *

labels = ["Equities", "Fixed Income", "Alternatives", "Cash"]
vals = [48, 27, 17, 8]
colors = [BLUE, ORANGE, GREEN, NEUTRAL]

fig, ax = setup()
wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
import numpy as np
for w, lab, v in zip(wedges, labels, vals):
    ang = np.deg2rad((w.theta1 + w.theta2) / 2)
    xy = (np.cos(ang), np.sin(ang))
    ax.annotate(f"{lab}\n{v}%", xy=(xy[0] * 0.81, xy[1] * 0.81),
                xytext=(xy[0] * 1.25, xy[1] * 1.25), ha="center", va="center",
                fontsize=14, fontweight="bold", color=TEXT,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
ax.text(0, 0, "100%", ha="center", va="center", fontsize=30,
        fontweight="bold", color=INK)
ax.set_aspect("equal")
titles(fig, "Portfolio Allocation", "As of Dec 2025  |  % of total assets")
footnote(fig, "Illustrative data only.")
save(fig, "donut.png", layout=(0.10, 0.90, 0.80, 0.12))
```

### 12.4 Heatmap (correlation matrix)

```python
import numpy as np
from chart_style import *

names = ["SPX", "NDX", "Gold", "UST10Y", "USD"]
rng = np.random.RandomState(3)
a = rng.randn(5, 5); m = np.corrcoef(a @ a.T + np.eye(5) * 3)
m = (m + m.T) / 2; np.fill_diagonal(m, 1)

fig, ax = setup()
im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(5), names); ax.set_yticks(range(5), names)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
ax.grid(which="minor", color="white", lw=1); ax.grid(which="major", visible=False)
ax.tick_params(which="both", length=0, labelsize=13, colors=TEXT)
for s in ax.spines.values(): s.set_visible(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{m[i, j]:.2f}".replace("-", "\u2013"), ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if abs(m[i, j]) > 0.65 else TEXT)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=12, length=0, colors=TEXT)
titles(fig, "Cross-Asset Correlation", "Daily returns  |  Trailing 1Y")
footnote(fig, "Illustrative data only.")
save(fig, "heatmap.png", layout=(0.12, 0.90, 0.82, 0.12))
```

### 12.5 Scatter with trend line

```python
import numpy as np
from chart_style import *

rng = np.random.RandomState(1)
x = rng.uniform(10, 40, 40)
y = 0.8 * x + rng.normal(0, 4, 40)
m, c = np.polyfit(x, y, 1)

fig, ax = setup()
ax.scatter(x, y, s=80, color=BLUE, alpha=0.8, edgecolor="white", lw=1, zorder=4)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, m * xs + c, color=ORANGE, lw=2, linestyle="--", zorder=5)
style_axes(ax, "Implied volatility (%)", "Realised return (%)", grid="both")
titles(fig, "Volatility vs Return", "40 observations  |  Dashed line = linear fit")
footnote(fig, "Illustrative data only.")
save(fig, "scatter.png", layout=(0.10, 0.92, 0.80, 0.24))
```

---

## 13. Using real data

- Replace the simulated arrays with `pandas` data (CSV, Excel, `yfinance`, Bloomberg export). Keep dates as `datetime64`.
- Convert returns to cumulative percentages (`np.cumsum(ret) * 100` for simple additive P&L, or `(np.cumprod(1 + ret) - 1) * 100` for compounded returns). State which one in the subtitle.
- Do not change styling because data is real; only the title, subtitle and footnote text change.
- Keep the disclaimer footnote whenever a named security appears.
- When data is hypothetical, say so in the footnote (for example `Hypothetical data for illustration only.`) in addition to or instead of the standard disclaimer.

---

## 14. Workflow for producing a chart

1. Identify the chart type and pick the matching playbook in section 10.
2. Build the figure with `setup()`, plot data using the palette and mark rules.
3. Add overlays: `zero_line`, `highlight_band`, `area_fill`, `end_label` as appropriate.
4. Apply `style_axes` (choose the correct `grid` mode), then `bold_zero_tick` once ticks are final.
5. Add `titles`, `bottom_legend` (if needed) and `footnote`.
6. Save with `save(...)` so padding is uniform.
7. **View the rendered image** and run the checklist below. Fix and re-render rather than shipping a first draft.

---

## 15. Pre-delivery checklist

- [ ] Background is pure white `#ffffff`, including the exported file
- [ ] Gridlines are dashed light grey and sit behind the data
- [ ] Grid on the correct axes for the chart type (both / value-only / none)
- [ ] Top and right spines hidden; left and bottom grey
- [ ] Title bold, centred, 26pt, one line; subtitle centred 14pt with `  |  ` separators
- [ ] Muted palette only, at most 4 colours, consistent meaning (green gain, orange loss)
- [ ] Zero line black and tick label `0`/`0%` bold when data crosses zero
- [ ] Negative numbers use an en dash (`–`), percent axes formatted `20%`
- [ ] Pill end-labels on the last point of each series, colours and text colours correct, none overlapping or clipped
- [ ] Legend (if shown) below the plot, horizontal, frameless
- [ ] Footnote present, bottom-left, aligned with plot's left edge
- [ ] Equal padding on all four sides; nothing touches the image edge
- [ ] No chart junk: no borders, shadows, gradients, 3D, default matplotlib colours

---

## 16. Common mistakes to avoid

- Forgetting `facecolor=BG` in `savefig` (exports with a grey or transparent background).
- Using `plt.tight_layout()` together with manual `subplots_adjust`; use `save()` only.
- Solid gridlines or dark gridlines; they must be dashed and light.
- Putting the legend inside the plot over the data.
- Left-aligned or oversized titles that wrap onto two lines.
- Pill labels clipped by the right edge (right margin too small) or overlapping each other.
- Red/green colour pairs, or more than four colours in one chart.
- Leaving the footnote off a chart that names a security.
- Calling `bold_zero_tick` before the y-limits/ticks are set (the label will not exist or will be wrong).
- Plotting dates on exact month starts without the 15-day shift, which leaves a gap between the first point and the y-axis.
