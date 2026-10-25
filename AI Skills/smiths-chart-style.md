---
name: smiths-chart-style
description: Smith's house style for report-ready charts, implemented by the institutional-exhibit Python package (pip install "institutional-exhibit>=0.5.0"). Use whenever creating, restyling or reviewing a chart, graph, plot or figure made with matplotlib, especially finance, reserves, returns, rates, FX, derivatives or market charts for reports and slides. Produces a fixed 1920x1080 canvas with a left-aligned bold title, subtitle, a themed chart area with legend at the bottom, and a Source and Note footer, all in Inter with locked sizes, spacing and colours. Charts are always displayed with f.show() and never saved to a file.
---

# Smith's Chart Style (institutional-exhibit)

Every chart is built with the `institutional-exhibit` package. It fixes the layout, fonts, colours and spacing, so the output is identical for everyone. Do not re-create the style by hand with raw matplotlib sizes and colours; use the package, and when something is missing, extend it from the package's own theme values (section 7).

The look in one sentence: **a white 1920x1080 canvas, a bold left-aligned title with a regular subtitle, a chart area with dashed light-grey gridlines, soft muted colours and the legend at the bottom, then an italic grey Source line and Note, with fixed margins and gaps.**

**Output rule: always display the chart with `f.show()`. Never save it.** Do not call `f.save()`, `savefig`, or write any image or data file for the chart.

---

## 1. Setup

```bash
pip install "institutional-exhibit>=0.5.0"      # Python 3.9+, Inter is bundled
pip install --upgrade institutional-exhibit
```

- `f.show()` requires version 0.5.0 or newer. If `AttributeError: 'Frame' object has no attribute 'show'` appears, upgrade the package.
- Import name has an underscore: `from institutional_exhibit import Frame, usd_bn, inr, THEME, SPEC`.
- Do not keep a local folder named `institutional_exhibit` next to the script. It shadows the installed package. Check with `python -c "import institutional_exhibit as m; print(m.__file__)"`; the path should be inside `site-packages`.
- The package registers the bundled Inter files and generates an ExtraBold title weight on import (needs `fonttools`, installed automatically). No system fonts are needed.

---

## 2. Non-negotiables

1. **Use `Frame`.** Title, subtitle, source and note go through `Frame(...)`. Never place them with `fig.text`.
2. **Do not change locked values.** Font sizes, margins, gaps, colours, line weights and legend style live in `SPEC` and `THEME` (frozen). Do not override them, monkey-patch them or hardcode different ones.
3. **Always call `f.fit()` after plotting and axis formatting, then `f.show()`.** Never save the figure: do not call `f.save()`, `savefig`, or write any image file. Display only. Do not use `plt.tight_layout()` or `bbox_inches="tight"`; the canvas size is exact.
4. **Set axis limits and ticks yourself.** `fit()` sizes the outside of the chart only. Choose clean `set_ylim`, tick steps and formatters so the top gridline reaches the top of the plot area.
5. **Source and note are mandatory.** Every chart has a `source` line. A `note` explains what is plotted, units and any caveats.
6. **Disclaimer.** Whenever a specific security, company, fund or index is named, add to the note: `The securities are quoted as an example and not as a recommendation.`
7. **Legend at the bottom, inside the plot area.** It is built automatically from `stacked_bars`, `line` and `add_legend_item`. Do not place legends elsewhere.
8. **Muted palette only.** Use `THEME.series` and `THEME.line`. No default matplotlib colours, no neon, no pure red or green.
9. **No chart junk.** No borders, 3D, shadows, gradients or top/right spines (the themed axes already follow this; the twin axis shows only its right spine).

---

## 3. Canvas and layout (fixed by the package)

| Property | Value |
|---|---|
| Canvas | 1920 x 1080 px (change only `width`/`height` in `Frame`) |
| DPI | 100 (figure size = px / 100) |
| Side margins | 60 px |
| Top margin | 40 px (title starts here) |
| Bottom margin | 40 px, measured from the last line of the note to the image edge. Constant however many lines the note wraps to |
| Gap, subtitle to chart area | 30 px, measured from the bottom of the last subtitle line's descenders |
| Gap, chart area to source | 30 px, measured to the top of the source line's capital letters |
| Source to note | 2 px |
| Chart area | Everything between those gaps. It includes the axes, tick labels, axis titles **and the legend** |

Behaviour to rely on:

- Text wraps automatically within the 1800 px text width.
- If the subtitle, source or note wrap to more lines, the chart area shrinks to make room. The gaps and the 40 px bottom margin never change.
- If the text is too long for the canvas height, `Frame` raises `ValueError("Text is too long ...")`. Shorten the text or increase `height`.
- `Frame(debug=True)` draws the grey plot area and orange margin lines, for checking the gaps. Never leave it on in a delivered chart.
- `f.plot_rect_px` returns `(left, top, width, height)` of the chart area.
- `f.show()` may scale the 1920x1080 frame to fit the window. Zoom in to inspect fine detail.

---

## 4. Typography

Font: **Inter** (bundled). Title uses the generated `Inter ExtraBold` at weight 800. All text inside the chart (ticks, axis titles, legend) uses Inter too; `Frame` sets this itself.

| Element | Size (px) | Weight / style | Colour |
|---|---|---|---|
| Title | 47.56 | ExtraBold | `#000000` |
| Subtitle | 20.23 (24 px line height) | Regular | `#000000` |
| Tick labels | 17.33 | Regular | `#000000` |
| Axis titles | 17.33 | Regular | `#000000` |
| Legend text | 18.33 | Regular | `#000000` |
| Source and note | 16 (20 px line height) | Italic | `#8c8c8c` |

### Text conventions

- **Title:** one line, Title Case, states the subject: `India's Foreign Exchange Reserves Composition`.
- **Subtitle:** one to three sentences giving the finding, with numbers: the period change, the latest move, and the latest split. It is commentary, not a label.
- **Source:** `Source: Reserve Bank of India via DBIE; USD/INR: Federal Reserve via FRED`. Name every data provider.
- **Note:** what the bars and lines show, units, how periods are defined, whether the latest period is partial, and how to read the direction (for example, `A higher USD/INR rate means a weaker rupee.`).
- Use `US$` for dollars, written `US$100bn`. The package escapes `$` for you; do not add backslashes yourself.
- Use an en dash (`–`) for ranges and negative numbers in text.

---

## 5. Colours (from `THEME`)

| Role | Hex |
|---|---|
| Series 1 (blue), primary or largest component | `#6b8fd1` |
| Series 2 (orange) | `#e8856a` |
| Series 3 (green) | `#55a868` |
| Series 4 (purple) | `#9b8ac4` |
| Line overlay (amber) | `#fcb149` |
| Text and titles | `#000000` |
| Source and note | `#8c8c8c` |
| Axis spines | `#7f7f7f` |
| Gridlines | `#b9b9b9` |

Rules:

- Stacked series take colours in the order above automatically. Put the largest component first.
- The overlay line (rate, price, index on the right-hand axis) is always amber with white-faced markers.
- At most four stacked series plus one line. Group small categories into "Other".
- A partial or provisional final period is shown with `partial_last=True`, which fades it to 60% opacity. Say so in the note.
- Bars have thin white edges so stacked segments are separated.

---

## 6. Using the package

### The pattern

```python
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from institutional_exhibit import Frame, usd_bn, inr

x = mdates.date2num(dates)             # dates as datetime.date / datetime

f = Frame(
    width=1920, height=1080,
    title="India's Foreign Exchange Reserves Composition",
    subtitle="From Q1 2023 to Q3 2026, reserves rose US$XXXbn to US$XXXbn ...",
    note="Note: Stacked bars show ... in US$ billions. ...",
    source="Source: Reserve Bank of India via DBIE; USD/INR: Federal Reserve via FRED",
)

ax, ax2 = f.add_axes(twin=True)        # twin=False returns a single axes

f.stacked_bars(ax, x, [fca, gold, sdr, rtp],
               labels=["Foreign currency assets (FCA)", "Gold reserves",
                       "SDRs", "IMF Reserve Tranche Position (RTP)"],
               width=34, partial_last=True)       # width in days on a date axis
f.line(ax2, x, usdinr, label="USD/INR (quarterly average)")

ax.set_ylim(0, 800);  ax2.set_ylim(82, 96)
ax.set_xlim(x[0] - 70, x[-1] + 70)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
ax.yaxis.set_major_formatter(usd_bn)               # US$100bn
ax2.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax2.yaxis.set_major_formatter(inr)                 # ₹82
f.ylabels(ax, "Reserves (US$ billions)", ax2, "Indian rupees per US dollar")

f.fit()
f.show()        # displays via plt.show(); never save
```

### API summary

- `Frame(width, height, title, subtitle, source, note, dpi=100, background="#FFFFFF", debug=False)`
- `f.add_axes(twin=False)` returns `ax`, or `(ax, ax2)` with a right-hand axis.
- `f.stacked_bars(ax, x, series, labels, width=0.7, partial_last=False)`
- `f.line(ax, x, y, label)`
- `f.add_legend_item(handle, label)` for custom legend entries.
- `f.ylabels(ax, left, ax2=None, right=None)`
- `f.fit()` places the legend and sizes the axes to fill the chart area.
- `f.show()` renders the frame and displays it with `plt.show()`. This is the only output step. Never save files.
- Formatters: `usd_bn` (`US$100bn`) and `inr` (`₹82`).

The package also has a `save()` method for other users. This skill never uses it.

### Axis conventions

- **Y ticks:** no tick marks, clean steps (100, 5, 1, 10%), range padded so the top gridline sits at the top of the plot area.
- **X ticks:** short grey marks. For quarterly or monthly bars, label only the years (`YearLocator`, `%Y`) or use `Q1 '23` labels when there are few bars. Do not label every bar if it crowds.
- **Dual axes:** left for the stacked quantity, right for the overlay line. Give both axis titles with units.
- **Bar width:** thin enough to show clear white space. On a date axis about 34 days for quarterly data, about 145 for annual.
- **Axis range:** start the x-range slightly before the first bar and end slightly after the last (about two bar-widths) so end bars are not on the spine.

---

## 7. Chart types and extending the package

The package natively covers **stacked bars with an optional overlay line on a second axis**. For other types, still start with `Frame` and `f.add_axes()`, so the canvas, text, margins, gaps, legend placement and `fit()` stay identical, and plot with theme values:

```python
from matplotlib.lines import Line2D
from institutional_exhibit import Frame, THEME as T

ax = f.add_axes()                       # themed axes: grid, spines, ticks already applied
ax.plot(x, y, color=T.series[0], linewidth=T.line_w * 72 / 100)   # px -> points at dpi 100
f.add_legend_item(Line2D([0], [0], color=T.series[0], lw=3), "Series name")
f.fit(); f.show()
```

Rules for those cases:

- Convert any theme size in px to points with `px * 72 / dpi` (dpi is 100).
- Use `T.series` for categories, `T.line` for an overlay, `T.grid` for any extra reference lines. Add no new colours.
- Single-series bars use `T.series[0]`; gains and losses may use green (`T.series[2]`) and orange (`T.series[1]`).
- Heatmaps and donut charts have no grid and no axes. Hide the axes spines and grid on that axes, but keep the `Frame` text and `fit()`.
- Put end-of-series values in the subtitle or as simple text labels. The package has no pill-label helper; do not recreate one without being asked.
- If a chart type is needed repeatedly, add a helper to the package (a new method on `Frame` using `THEME`) and release a new version, rather than copying code between scripts.

---

## 8. Data and wording

- Use real data when available; label hypothetical data in the note (`Hypothetical data for illustration only.`).
- State the basis of any change in the subtitle (period start to end, quarter-on-quarter, year-on-year, simple or compounded).
- Placeholders such as `XXX` or `DD Mon` are only for templates; replace them with computed values before delivery.
- Keep numbers consistent between the subtitle, the chart and the note.

---

## 9. Workflow

1. Prepare data and compute the figures for the subtitle and note.
2. Create `Frame(...)` with all four text fields.
3. `add_axes`, plot with `stacked_bars` and `line` (or themed plotting for other types).
4. Set limits, ticks, formatters and axis titles.
5. `f.fit()`, then `f.show()`. Never save the figure.
6. **Look at the displayed chart** and run the checklist. If a gap looks wrong, rerun with `debug=True` to see the chart area against the text, then turn it off.

---

## 10. Pre-delivery checklist

- [ ] Built with `Frame` and displayed with `f.show()`; no image file was written
- [ ] Title bold, left-aligned, one line; subtitle gives the finding with real numbers
- [ ] Source line present and names every provider; note explains units and caveats
- [ ] Disclaimer included if a named security, company, fund or index appears
- [ ] Legend at the bottom of the chart area, below the x labels, not overlapping
- [ ] Top gridline reaches the top of the chart area; limits and ticks are clean
- [ ] Colours only from the theme; at most four stacked series plus one amber line
- [ ] Partial final period faded and mentioned in the note
- [ ] Dashed light-grey gridlines behind the data; only left, bottom (and right on the twin axis) spines
- [ ] 30 px gaps above and below the chart area, and 40 px under the note, look right (`debug=True` to check)
- [ ] No `findfont` warnings, no leftover placeholders, `debug` is off
- [ ] Negative numbers and ranges use an en dash in text

---

## 11. Common mistakes

- Forgetting `f.fit()`, or calling it before limits, ticks and formatters are final.
- Calling `f.save()` or `savefig`. Charts are displayed with `f.show()` only.
- Using `plt.tight_layout()` or `bbox_inches="tight"` with `Frame`; the canvas size is exact.
- Using a package older than 0.5.0, where `f.show()` does not exist.
- Leaving a local `institutional_exhibit` folder that shadows the installed package.
- Overriding fonts or sizes with `rcParams` or `fontsize=` arguments on ticks and legends.
- Asking for font weight 800 on a font other than the bundled Inter title family (causes `findfont` warnings).
- Reusing a version number when releasing package changes (PyPI rejects it; bump the version).
- Writing `\$` by hand in text (the package escapes `$` already).
- Leaving `debug=True` on, or placeholder `XXX` text in the subtitle.
- Letting the subtitle or note grow so long that the chart area becomes too small to read.