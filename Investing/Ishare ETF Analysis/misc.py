"""Plot rolling one-year return against annualised volatility for several ETFs."""
from pathlib import Path
import sys
from tempfile import gettempdir

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

# Configuration: None for FUNDS_TO_PLOT uses every ticker in etf_loader.FUNDS.
FUNDS_TO_PLOT = ["IWMO"]     # e.g. ["IWMO", "IWQU", "IWVL"]
START_DATE = None        # e.g. "2015-01-01"; filters window end dates
END_DATE = None
REFRESH = False

BG = "#ffffff"
INK = "#000000"
AXIS = "#7f7f7f"
GRID = "#b9b9b9"
FOOT = "#8c8c8c"
PALETTE = (
    "#6b8fd1", "#e8856a", "#55a868", "#9b8ac4", "#fcb149",
    "#4c9aa6", "#c26d8f", "#8c8c5a", "#5b6fa8", "#d1a36b",
)
FONT_STACK = ("Inter", "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans")

CANVAS_W, CANVAS_H = 1920, 1080
LEFT_PX, RIGHT_PX = 60, 1860
TITLE_SIZE, SUBTITLE_SIZE, FOOT_SIZE = 28.8, 14.4, 9.6
TICK_SIZE, LABEL_SIZE, LEGEND_SIZE = 7.8, 9.0, 11
TITLE_TOP_PX, SUBTITLE_TOP_PX = 40, 103
SOURCE_TOP_PX, NOTE_TOP_PX = 978, 1002
CHART_TOP_PX, CHART_BOTTOM_PX = 173, 956
AXES_LEFT_PX = 150       # leaves room for y tick labels + title inside the 60px margin
TITLE_BASELINE_PX = TITLE_TOP_PX + 43
SUBTITLE_BASELINES_PX = (SUBTITLE_TOP_PX + 25, SUBTITLE_TOP_PX + 54)
SOURCE_BASELINE_PX = SOURCE_TOP_PX + 15
NOTE_BASELINES_PX = (NOTE_TOP_PX + 15, NOTE_TOP_PX + 35)


def load_navs(tickers) -> dict[str, pd.Series]:
    """Load saved/downloaded NAV history for each requested fund."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from etf_loader import FUNDS, load_all

    tickers = list(FUNDS) if tickers is None else [t.strip().upper() for t in tickers]
    unknown = [t for t in tickers if t not in FUNDS]
    if unknown:
        raise ValueError(f"Unknown funds {unknown}. Available funds: {', '.join(FUNDS)}")
    data = load_all(refresh=REFRESH)
    navs = {}
    for ticker in tickers:
        if ticker not in data or data[ticker]["nav"].empty:
            print(f"Skipping {ticker}: no NAV history.")
            continue
        nav = data[ticker]["nav"].copy()
        nav["Date"] = pd.to_datetime(nav["Date"])
        navs[ticker] = (
            nav.set_index("Date")["NAV"].sort_index()
            .loc[lambda s: ~s.index.duplicated(keep="last")]
        )
    if not navs:
        raise ValueError("No NAV history available for the selected funds.")
    return navs


def rolling_one_year_stats(nav: pd.Series) -> pd.DataFrame:
    """Return and annualised daily-return std dev over each calendar one-year window."""
    nav = nav.dropna().sort_index()
    if len(nav) < 2:
        return pd.DataFrame()
    targets = nav.index - pd.DateOffset(years=1)
    positions = nav.index.searchsorted(targets, side="right") - 1
    records = []
    for endpoint, start_target, start_pos in zip(nav.index, targets, positions):
        if start_pos < 0 or start_target < nav.index[0]:
            continue
        window = nav.iloc[start_pos:]
        window = window.loc[window.index <= endpoint]
        daily_returns = window.pct_change().dropna()
        if daily_returns.empty:
            continue
        records.append({
            "Date": endpoint,
            "Return": nav.loc[endpoint] / nav.iloc[start_pos] - 1,
            "Std Dev": daily_returns.std(ddof=1) * np.sqrt(252),
        })
    return pd.DataFrame(records).set_index("Date") if records else pd.DataFrame()


def build_stats(navs: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack every fund's windows into one long frame with a Fund column."""
    frames = []
    for ticker, nav in navs.items():
        stats = rolling_one_year_stats(nav)
        if stats.empty:
            print(f"Skipping {ticker}: less than one year of NAV history.")
            continue
        start = pd.Timestamp(START_DATE) if START_DATE else nav.index[0] + pd.DateOffset(years=1)
        end = pd.Timestamp(END_DATE) if END_DATE else nav.index[-1]
        stats = stats.loc[(stats.index >= start) & (stats.index <= end)].copy()
        if stats.empty:
            continue
        stats["Fund"] = ticker
        frames.append(stats)
    if not frames:
        raise ValueError("No complete one-year windows fall within the selected date range.")
    return pd.concat(frames)


def build_commentary(stats: pd.DataFrame) -> str:
    """One data-driven analyst paragraph; wording and signs follow the numbers."""
    by_fund = stats.groupby("Fund").agg(
        med_ret=("Return", "median"), med_vol=("Std Dev", "median"),
    )
    by_fund["ratio"] = by_fund["med_ret"] / by_fund["med_vol"]
    top_ret = by_fund["med_ret"].idxmax()
    top_ratio = by_fund["ratio"].idxmax()
    top_vol = by_fund["med_vol"].idxmax()
    worst_pos = int(stats["Return"].to_numpy().argmin())
    worst = stats.iloc[worst_pos]
    worst_date = stats.index[worst_pos]
    positive = (stats["Return"] > 0).mean() * 100
    n_funds = stats["Fund"].nunique()

    s1 = (
        f"{n_funds} ETFs, {len(stats):,} windows, {stats.index.min():%b %Y}–"
        f"{stats.index.max():%b %Y}: median return {stats['Return'].median():.1%} at "
        f"{stats['Std Dev'].median():.1%} volatility; {positive:.0f}% of windows positive."
    )
    s2 = (
        f"Best median return: {top_ret} ({by_fund.loc[top_ret, 'med_ret']:.1%}); best "
        f"return/vol: {top_ratio} ({by_fund.loc[top_ratio, 'ratio']:.2f}x); highest vol: "
        f"{top_vol} ({by_fund.loc[top_vol, 'med_vol']:.1%}); worst window: {worst['Fund']} "
        f"{worst['Return']:.1%} ({worst_date:%b %Y})."
    )
    return f"{s1} {s2}"


def main() -> None:
    stats = build_stats(load_navs(FUNDS_TO_PLOT))
    funds = list(stats["Fund"].unique())

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(FONT_STACK)
    inter_fonts = Path.home() / "Library" / "Fonts"
    inter_regular = inter_fonts / "Inter-VariableFont_opsz,wght.ttf"
    inter_italic = inter_fonts / "Inter-Italic-VariableFont_opsz,wght.ttf"
    inter_bold = Path(gettempdir()) / "volsmith-inter-bold.ttf"
    if inter_regular.exists():
        font_manager.fontManager.addfont(str(inter_regular))
        static_font = instantiateVariableFont(
            TTFont(str(inter_regular)), {"opsz": 32, "wght": 700}, inplace=False
        )
        for name_id, value in {
            1: "Inter Bold", 2: "Regular", 4: "Inter Bold",
            6: "Inter-Bold", 16: "Inter Bold", 17: "Regular",
        }.items():
            for name_record in static_font["name"].names:
                if name_record.nameID == name_id:
                    name_record.string = value.encode(name_record.getEncoding())
        static_font.save(str(inter_bold))
        font_manager.fontManager.addfont(str(inter_bold))
    if inter_italic.exists():
        font_manager.fontManager.addfont(str(inter_italic))

    def px_x(value: float) -> float:
        return value / CANVAS_W

    def px_y(value: float) -> float:
        return 1 - value / CANVAS_H

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)

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
        """Break to a new line only when the next word would pass the right margin."""
        lines, current = [], ""
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

    # Title
    title_font = FontProperties(fname=str(inter_bold)) if inter_bold.exists() else None
    fig.text(
        px_x(LEFT_PX), px_y(TITLE_BASELINE_PX),
        "ETF Rolling One-Year Returns vs Volatility",
        fontsize=TITLE_SIZE, fontproperties=title_font,
        fontweight="bold" if title_font is None else "normal",
        color=INK, ha="left", va="baseline",
    )

    # Dynamic analyst subtitle: flows across two lines, wrapping only at the right margin
    subtitle_lines = wrap_to_margins(
        build_commentary(stats), SUBTITLE_SIZE, RIGHT_PX - LEFT_PX,
    )
    if len(subtitle_lines) > len(SUBTITLE_BASELINES_PX):
        raise ValueError(
            "The analyst subtitle exceeds the two-line layout; shorten its commentary."
        )
    for line, baseline in zip(subtitle_lines, SUBTITLE_BASELINES_PX):
        fig.text(px_x(LEFT_PX), px_y(baseline), line, fontsize=SUBTITLE_SIZE,
                 color=INK, ha="left", va="baseline")

    # Chart (axes start inside the margin so tick labels and the y title stay within it)
    axes_top, axes_bottom = CHART_TOP_PX + 20, CHART_BOTTOM_PX - 110
    ax = fig.add_axes([
        px_x(AXES_LEFT_PX), px_y(axes_bottom), px_x(RIGHT_PX - AXES_LEFT_PX),
        (axes_bottom - axes_top) / CANVAS_H,
    ], facecolor="none")
    colours = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(funds)}
    for fund in funds:
        sub = stats[stats["Fund"] == fund]
        ax.scatter(sub["Std Dev"], sub["Return"], s=15, alpha=0.55,
                   color=colours[fund], edgecolors="none", zorder=3)
    ax.set_xlabel("Annualised standard deviation", fontsize=LABEL_SIZE, color=INK, labelpad=8)
    ax.set_ylabel("One-year return", fontsize=LABEL_SIZE, color=INK, labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", linewidth=0.8, color=GRID, zorder=0)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors=INK, length=0, pad=6)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=colours[f], markeredgecolor="none", label=f)
        for f in funds
    ]
    legend = ax.legend(
        handles=handles, loc="center", ncol=min(len(funds), 8), frameon=False,
        fontsize=LEGEND_SIZE, columnspacing=1.8, handletextpad=0.5,
    )
    legend.set_bbox_to_anchor(
        (0.5, px_y(CHART_BOTTOM_PX - 17)), transform=fig.transFigure,
    )

    # Footer
    source = "Source: iShares historical NAVs"
    note = (
        "Note: Each point is the NAV return over the calendar year ending on that date; "
        "the starting NAV is the last observation on or before the one-year lookback date. "
        "Volatility is the sample standard deviation of daily NAV returns in that same "
        "window, annualised by √252. Return/vol is the ratio of each fund's median return "
        "to its median volatility."
    )
    fig.text(px_x(LEFT_PX), px_y(SOURCE_BASELINE_PX), source, fontsize=FOOT_SIZE,
             style="italic", color=FOOT, ha="left", va="baseline")
    note_lines = wrap_to_margins(note, FOOT_SIZE, RIGHT_PX - LEFT_PX, "italic")
    if len(note_lines) > len(NOTE_BASELINES_PX):
        raise ValueError("The chart note exceeds the two-line footer layout; shorten the note.")
    for line, baseline in zip(note_lines, NOTE_BASELINES_PX):
        fig.text(px_x(LEFT_PX), px_y(baseline), line, fontsize=FOOT_SIZE,
                 style="italic", color=FOOT, ha="left", va="baseline")
    plt.show()


if __name__ == "__main__":
    main()