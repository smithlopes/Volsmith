from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib import font_manager

# =====================================================================
# IMPORT YAHOO FINANCE
# =====================================================================

try:
    import yfinance as yf
except ImportError as error:
    raise ImportError(
        "\nCould not import 'yfinance'.\n\n"
        "Install or upgrade the required packages using:\n\n"
        "    pip install --upgrade yfinance pandas\n\n"
        f"Original import error: {error}"
    ) from error

# =====================================================================
# USER SETTINGS
# =====================================================================

# Manual chart positioning controls
TITLE_Y = 0.955
SUBTITLE_Y = 0.915
SUBTITLE_LINE_SPACING = 0.020
HEAT_MAP_Y = 0.12
SOURCE_Y = 0.081
NOTE_Y = 0.060
NOTE_LINE_SPACING = 0.016

# Horizons shown in the heatmap (label -> look-back offset).
PERIODS = {
    "1D": pd.DateOffset(days=1),
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
}

PERIOD_COLUMNS = list(PERIODS.keys())

# Horizons of this many years or more are shown annualised (p.a.).
# Set to None to show cumulative returns for every horizon.
ANNUALISE_FROM_YEARS = None

# Column used to sort the rows
SORT_BY = "1Y"

# Volatility column: annualised SD of daily returns over this look-back
SD_LOOKBACK = pd.DateOffset(years=1)
SD_LABEL = "1Y SD"
MIN_SD_OBSERVATIONS = 20

FONT_FAMILY = "Helvetica"

# Extra calendar days requested before the longest look-back, so the
# observation on or before the 10Y target date is always captured.
DOWNLOAD_LOOKBACK_DAYS = 15

# =====================================================================
# OUTPUT LOCATION
# =====================================================================

OUTPUT_DIR = Path(__file__).resolve().parent

PNG_FILE = (
    OUTPUT_DIR
    / "Global_Currency_Returns_Heatmap.png"
)

# =====================================================================
# GLOBAL STYLE
# =====================================================================

plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["font.sans-serif"] = [
    FONT_FAMILY,
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]

# =====================================================================
# MAJOR GLOBAL CURRENCIES
# YAHOO FINANCE SPOT FX RATES VS THE USD
# =====================================================================

SECTORS = {
    "Euro": "EURUSD=X",
    "British Pound": "GBPUSD=X",
    "Japanese Yen": "JPY=X",
    "Swiss Franc": "CHF=X",
    "Canadian Dollar": "CAD=X",
    "Australian Dollar": "AUDUSD=X",
    "New Zealand Dollar": "NZDUSD=X",
    "Norwegian Krone": "NOK=X",
    "Swedish Krona": "SEK=X",
    "Chinese Yuan": "CNY=X",
    "Indian Rupee": "INR=X",
    "South Korean Won": "KRW=X",
    "Taiwan Dollar": "TWD=X",
    "Hong Kong Dollar": "HKD=X",
    "Singapore Dollar": "SGD=X",
    "Mexican Peso": "MXN=X",
    "Brazilian Real": "BRL=X",
    "South African Rand": "ZAR=X",
    "Turkish Lira": "TRY=X",
    "Polish Zloty": "PLN=X",
    "Israeli Shekel": "ILS=X",
}

# =====================================================================
# GLOBAL CURRENCY BENCHMARKS
# =====================================================================

BENCHMARKS = {
    "US Dollar Index (DXY)": "DX-Y.NYB",
    "EM Currencies (CEW)": "CEW",
}

ALL_INDICES = {
    **SECTORS,
    **BENCHMARKS,
}

BENCHMARK_NAMES = list(
    BENCHMARKS
)

# Yahoo quotes these pairs as units of foreign currency per 1 USD.
# Levels are inverted (1 / level) so that every series is expressed as
# USD per 1 unit of foreign currency; a positive return therefore always
# means the currency appreciated against the USD.
INVERTED_TICKERS = {
    "JPY=X",
    "CHF=X",
    "CAD=X",
    "NOK=X",
    "SEK=X",
    "CNY=X",
    "INR=X",
    "KRW=X",
    "TWD=X",
    "HKD=X",
    "SGD=X",
    "MXN=X",
    "BRL=X",
    "ZAR=X",
    "TRY=X",
    "PLN=X",
    "ILS=X",
}

# =====================================================================
# TICKER VALIDATION
# =====================================================================


def clean_index_code(
    index_code: object,
) -> str:
    """Convert a ticker symbol to a clean string."""

    if index_code is None:
        return ""

    return str(index_code).strip()


def validate_ticker_code(
    index_name: str,
    index_code: object,
) -> str:
    """Validate and return a Yahoo Finance ticker symbol."""

    cleaned_code = clean_index_code(
        index_code
    )

    if not cleaned_code:
        raise ValueError(
            f"No Yahoo Finance ticker was supplied for {index_name}."
        )

    if " " in cleaned_code:
        raise ValueError(
            f"Yahoo Finance ticker for {index_name} must not "
            f"contain spaces. Received: {cleaned_code!r}"
        )

    return cleaned_code


# =====================================================================
# DATE RANGE
# =====================================================================

today = pd.Timestamp.today().normalize()

download_start_date = (
    today
    - pd.DateOffset(years=10)
    - pd.Timedelta(days=14)
)

download_end_date = today

chart_title = (
    "Global Currency Multi-Period Returns vs USD"
)

# =====================================================================
# PREPARE VALID TICKERS
# =====================================================================

valid_indices: dict[str, str] = {}

for index_name, index_code in ALL_INDICES.items():

    cleaned_index_code = validate_ticker_code(
        index_name=index_name,
        index_code=index_code,
    )

    valid_indices[
        index_name
    ] = cleaned_index_code

if not valid_indices:
    raise ValueError(
        "No valid Yahoo Finance tickers were supplied."
    )

# Request each code only once.
unique_index_codes = list(
    dict.fromkeys(
        valid_indices.values()
    )
)

index_code_to_names: dict[str, list[str]] = {}

for index_name, index_code in valid_indices.items():

    index_code_to_names.setdefault(
        index_code,
        [],
    ).append(
        index_name
    )

# =====================================================================
# DOWNLOAD YAHOO FINANCE FX DATA
# =====================================================================

print()

print(
    "Downloading Yahoo Finance FX data..."
)

print(
    f"Requested period: "
    f"{download_start_date:%Y-%m-%d} to "
    f"{download_end_date:%Y-%m-%d}"
)

print(
    f"Requesting {len(unique_index_codes)} unique "
    f"ticker(s) in one batched request."
)

download_started = time.perf_counter()

try:

    # yfinance treats the end date as exclusive, so add one day.
    raw_download = yf.download(
        unique_index_codes,
        start=download_start_date.strftime(
            "%Y-%m-%d"
        ),
        end=(
            download_end_date
            + pd.Timedelta(days=1)
        ).strftime(
            "%Y-%m-%d"
        ),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

except Exception as error:

    raise RuntimeError(
        f"The batched Yahoo Finance request failed. "
        f"Original error: {error}"
    ) from error

download_seconds = (
    time.perf_counter()
    - download_started
)

if raw_download is None or raw_download.empty:
    raise ValueError(
        "No data returned from Yahoo Finance."
    )

close_prices = raw_download["Close"]

if isinstance(close_prices, pd.Series):

    close_prices = close_prices.to_frame(
        name=unique_index_codes[0]
    )

if getattr(close_prices.index, "tz", None) is not None:

    close_prices.index = (
        close_prices.index.tz_localize(None)
    )

close_prices = close_prices.rename_axis(
    "DATE"
)

downloaded_data = (
    close_prices
    .reset_index()
    .melt(
        id_vars="DATE",
        var_name="INDEX_CODE",
        value_name="LEVEL",
    )
    .dropna(
        subset=[
            "LEVEL",
        ]
    )
)

# Express every pair as USD per 1 unit of foreign currency.
inverted_mask = (
    downloaded_data["INDEX_CODE"]
    .isin(INVERTED_TICKERS)
)

downloaded_data.loc[
    inverted_mask,
    "LEVEL",
] = (
    1.0
    / downloaded_data.loc[
        inverted_mask,
        "LEVEL",
    ]
)

hist = pd.DataFrame(
    downloaded_data
)

if hist.empty:
    raise ValueError(
        "Yahoo Finance returned an empty dataset."
    )

print(
    f"Yahoo Finance download completed in "
    f"{download_seconds:.2f} seconds."
)

print(
    f"Raw observations returned: "
    f"{len(hist):,}"
)

# =====================================================================
# NORMALISE YAHOO FINANCE DATA
# =====================================================================

hist.columns = [
    str(column).strip().upper()
    for column in hist.columns
]

required_columns = {
    "INDEX_CODE",
    "DATE",
    "LEVEL",
}

missing_columns = (
    required_columns.difference(
        hist.columns
    )
)

if missing_columns:
    raise ValueError(
        "Yahoo Finance response is missing required column(s): "
        + ", ".join(
            sorted(
                missing_columns
            )
        )
    )

hist["INDEX_CODE"] = (
    hist["INDEX_CODE"]
    .astype(str)
    .str.strip()
)

hist["DATE"] = pd.to_datetime(
    hist["DATE"],
    errors="coerce",
    utc=True,
).dt.tz_convert(None)

hist["LEVEL"] = pd.to_numeric(
    hist["LEVEL"],
    errors="coerce",
)

invalid_rows = (
    hist["DATE"].isna()
    | hist["LEVEL"].isna()
)

if invalid_rows.any():

    print(
        f"Removing {int(invalid_rows.sum())} row(s) "
        f"with invalid dates or index levels."
    )

    hist = hist.loc[
        ~invalid_rows
    ].copy()

if hist.empty:
    raise ValueError(
        "No valid Yahoo Finance observations remained after "
        "normalising the downloaded data."
    )

requested_code_set = set(
    unique_index_codes
)

unmatched_code_rows = (
    ~hist["INDEX_CODE"].isin(
        requested_code_set
    )
)

if unmatched_code_rows.any():

    unmatched_codes = sorted(
        hist.loc[
            unmatched_code_rows,
            "INDEX_CODE",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Removing observations for unrecognised "
        "ticker(s): "
        + ", ".join(
            unmatched_codes
        )
    )

hist = hist.loc[
    ~unmatched_code_rows
].copy()

if hist.empty:
    raise ValueError(
        "No observations matched the requested "
        "tickers."
    )

if "CURRENCY" not in hist.columns:
    hist["CURRENCY"] = "USD"

hist["REQUESTED_INDEX_CODE"] = hist["INDEX_CODE"]

hist = (
    hist.sort_values(
        [
            "INDEX_CODE",
            "DATE",
        ]
    )
    .drop_duplicates(
        subset=[
            "INDEX_CODE",
            "DATE",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)

hist["price"] = hist["LEVEL"]

hist["price_source"] = (
    "Yahoo Finance Daily Close (USD per unit of currency)"
)

returned_codes = set(
    hist["INDEX_CODE"]
    .unique()
    .tolist()
)

for missing_code in unique_index_codes:

    if missing_code not in returned_codes:

        print(
            f"No Yahoo Finance history returned for ticker "
            f"{missing_code}: "
            + ", ".join(
                index_code_to_names.get(
                    missing_code,
                    [],
                )
            )
        )


def get_price_on_or_before(
    price_data: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.Series | None:
    eligible_data = price_data.loc[
        price_data["date"] <= target_date
    ]

    if eligible_data.empty:
        return None

    return eligible_data.iloc[-1]


# =====================================================================
# CALCULATE MULTI-PERIOD RETURNS
# =====================================================================

# Common reference date: the latest observation in the dataset
as_of_date = hist["DATE"].max()

print(
    f"Latest observation date: {as_of_date:%Y-%m-%d}"
)

rows: list[dict[str, object]] = []

for index_name, index_code in valid_indices.items():

    try:

        temp = (
            hist.loc[
                hist["INDEX_CODE"] == index_code,
                [
                    "DATE",
                    "price",
                    "price_source",
                    "CURRENCY",
                ],
            ]
            .rename(columns={"DATE": "date"})
            .dropna(subset=["date", "price"])
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )

        if temp.empty:

            print(
                f"No Yahoo Finance history returned for "
                f"{index_name}: {index_code}"
            )

            continue

        end_observation = temp.iloc[-1]

        end_date = pd.Timestamp(
            end_observation["date"]
        )

        end_price = float(
            end_observation["price"]
        )

        row: dict[str, object] = {
            "Market": index_name,
        }

        for period_label, offset in PERIODS.items():

            target_start_date = end_date - offset

            start_observation = get_price_on_or_before(
                temp,
                target_start_date,
            )

            if start_observation is None:

                row[period_label] = np.nan

                continue

            start_price = float(
                start_observation["price"]
            )

            if start_price == 0:

                period_return = np.nan

            else:

                period_return = (
                    (end_price / start_price) - 1
                ) * 100

            row[period_label] = period_return

        rows.append(
            row
        )

    except Exception as exc:

        print(
            f"Failed for {index_name}: {exc}"
        )

# =====================================================================
# BUILD DATAFRAME
# =====================================================================

combined_df = pd.DataFrame(
    rows
)

if combined_df.empty:
    raise ValueError(
        "No valid sector data found."
    )

combined_df = combined_df.set_index(
    "Market"
)

combined_df = combined_df.reindex(
    columns=PERIOD_COLUMNS
)

combined_df = combined_df.sort_values(
    "1Y",
    ascending=False,
    na_position="last",
)

# =====================================================================
# DYNAMIC INSIGHT-LED SUBTITLE
# =====================================================================

subtitle = (
    "Cross-currency performance against the US dollar across "
    "short-, medium- and long-term horizons"
)

insight_df = combined_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

one_year_ranking = (
    insight_df["1Y"]
    .dropna()
    .sort_values(ascending=False)
)

ten_year_ranking = (
    insight_df["10Y"]
    .dropna()
    .sort_values(ascending=False)
)

one_month_ranking = (
    insight_df["1M"]
    .dropna()
    .sort_values(ascending=False)
)

if (
    len(one_year_ranking) >= 2
    and len(ten_year_ranking) >= 2
    and len(one_month_ranking) >= 2
):

    subtitle = (
        f"{str(one_year_ranking.index[0])} led one-year returns "
        f"against the US dollar at "
        f"{float(one_year_ranking.iloc[0]):+.1f}%, while "
        f"{str(one_year_ranking.index[-1])} lagged at "
        f"{float(one_year_ranking.iloc[-1]):+.1f}%. "
        f"{str(ten_year_ranking.index[0])} delivered the strongest "
        f"ten-year return at "
        f"{float(ten_year_ranking.iloc[0]):+.1f}%, and "
        f"{str(one_month_ranking.index[0])} led the latest one-month "
        f"performance at "
        f"{float(one_month_ranking.iloc[0]):+.1f}%"
    )

# =====================================================================
# PREPARE HEATMAP DATA
# =====================================================================

heatmap_df = combined_df[
    PERIOD_COLUMNS
].copy()

annot_df = heatmap_df.map(
    lambda value: (
        f"{value:+.1f}%"
        if pd.notna(value)
        else ""
    )
)

# =====================================================================
# PLOT
# =====================================================================

number_of_rows = len(
    heatmap_df
)

total_number_of_columns = len(
    PERIOD_COLUMNS
)

figure_width = 23

figure_height = max(
    10,
    0.42 * number_of_rows + 4.2,
)

fig = plt.figure(
    figsize=(
        figure_width,
        figure_height,
    ),
    facecolor="white",
)

renderer = fig.canvas.get_renderer()

common_left_position = 0.03

common_right_position = 0.97

# ---------------------------------------------------------------------
# Measure the widest y-axis label so the first column fits its content
# ---------------------------------------------------------------------

maximum_label_width_pixels = 0.0

for label in [
    "Currencies",
    *heatmap_df.index,
]:

    probe = fig.text(
        0,
        0,
        label,
        fontsize=16,
        fontweight="bold",
    )

    maximum_label_width_pixels = max(
        maximum_label_width_pixels,
        probe.get_window_extent(
            renderer=renderer
        ).width,
    )

    probe.remove()

label_padding_left_pixels = 0.12 * fig.dpi

label_padding_right_pixels = 0.20 * fig.dpi

label_column_pixels = (
    maximum_label_width_pixels
    + label_padding_left_pixels
    + label_padding_right_pixels
)

table_width_pixels = (
    (
        common_right_position
        - common_left_position
    )
    * fig.bbox.width
)

pixels_per_heatmap_column = (
    (
        table_width_pixels
        - label_column_pixels
    )
    / total_number_of_columns
)

y_axis_label_column_width = (
    label_column_pixels
    / pixels_per_heatmap_column
)

label_text_offset = (
    label_padding_left_pixels
    / pixels_per_heatmap_column
)

ax = fig.add_axes(
    [
        common_left_position,
        HEAT_MAP_Y,
        common_right_position
        - common_left_position,
        0.74,
    ]
)

# ---------------------------------------------------------------------
# Heatmap: one independent colour scale per horizon column
# ---------------------------------------------------------------------

for column in PERIOD_COLUMNS:

    column_df = pd.DataFrame(
        np.nan,
        index=heatmap_df.index,
        columns=heatmap_df.columns,
    )

    column_df[column] = heatmap_df[column]

    column_annot = pd.DataFrame(
        "",
        index=heatmap_df.index,
        columns=heatmap_df.columns,
    )

    column_annot[column] = annot_df[column]

    sns.heatmap(
        column_df,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=column_annot.values,
        fmt="",
        annot_kws={
            "size": 14,
            "weight": "bold",
        },
        mask=column_df.isna(),
        cbar=False,
        linewidths=1,
        linecolor="white",
    )

ax.set_xlabel("")

ax.set_ylabel("")

ax.set_xticks([])

ax.set_xlim(
    -y_axis_label_column_width,
    total_number_of_columns,
)

ax.set_ylim(
    number_of_rows,
    -1,
)

# =====================================================================
# X-AXIS LABEL CELLS
# =====================================================================

x_axis_labels = PERIOD_COLUMNS

for column_position, column_label in enumerate(
    x_axis_labels
):

    ax.add_patch(
        Rectangle(
            (
                column_position,
                -1,
            ),
            1,
            1,
            facecolor="#F2F2F2",
            edgecolor="none",
            linewidth=0,
            zorder=10,
            clip_on=False,
        )
    )

    ax.text(
        column_position + 0.5,
        -0.5,
        column_label,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="black",
        zorder=11,
        clip_on=False,
    )

ax.add_patch(
    Rectangle(
        (
            0,
            -1,
        ),
        total_number_of_columns,
        1,
        facecolor="none",
        edgecolor="#1b263b",
        linewidth=2.5,
        zorder=15,
        clip_on=False,
    )
)

# =====================================================================
# Y-AXIS LABEL CELLS
# =====================================================================

ax.set_yticks([])

ax.tick_params(
    axis="y",
    which="both",
    length=0,
)

for row_position, row_label in enumerate(
    heatmap_df.index
):

    if row_label in BENCHMARK_NAMES:

        y_axis_label_fill = "#a9d6e5"

    else:

        y_axis_label_fill = "#F2F2F2"

    ax.add_patch(
        Rectangle(
            (
                -y_axis_label_column_width,
                row_position,
            ),
            y_axis_label_column_width,
            1,
            facecolor=y_axis_label_fill,
            edgecolor="none",
            linewidth=0,
            zorder=10,
            clip_on=False,
        )
    )

    ax.text(
        -y_axis_label_column_width
        + label_text_offset,
        row_position + 0.5,
        row_label,
        ha="left",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="black",
        zorder=11,
        clip_on=False,
    )

# =====================================================================
# TOP-LEFT CORNER CELL
# =====================================================================

ax.add_patch(
    Rectangle(
        (
            -y_axis_label_column_width,
            -1,
        ),
        y_axis_label_column_width,
        1,
        facecolor="#F2F2F2",
        edgecolor="#1b263b",
        linewidth=2.5,
        zorder=17,
        clip_on=False,
    )
)

ax.text(
    -y_axis_label_column_width
    + label_text_offset,
    -0.5,
    "Currencies",
    ha="left",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="black",
    zorder=18,
    clip_on=False,
)

# =====================================================================
# BORDERS AND ROW GUIDES
# =====================================================================

# Vertical border between the label column and the data columns
ax.plot(
    [
        0,
        0,
    ],
    [
        -1,
        number_of_rows,
    ],
    color="#1b263b",
    linewidth=2.5,
    zorder=26,
    clip_on=False,
)

# Row guides
for row_boundary in range(
    1,
    number_of_rows,
):

    ax.plot(
        [
            -y_axis_label_column_width,
            total_number_of_columns,
        ],
        [
            row_boundary,
            row_boundary,
        ],
        color="#1b263b",
        linewidth=1,
        linestyle=(
            0,
            (
                1,
                2.5,
            ),
        ),
        alpha=1,
        zorder=20,
        clip_on=False,
    )

# Outside border of the entire table
ax.add_patch(
    Rectangle(
        (
            -y_axis_label_column_width,
            -1,
        ),
        total_number_of_columns
        + y_axis_label_column_width,
        number_of_rows + 1,
        facecolor="none",
        edgecolor="#1b263b",
        linewidth=2.5,
        zorder=25,
        clip_on=False,
    )
)

# =====================================================================
# TITLE
# =====================================================================

fig.text(
    common_left_position,
    TITLE_Y,
    chart_title,
    ha="left",
    va="top",
    fontsize=36,
    fontweight="bold",
)

# =====================================================================
# WORD-WRAPPED SUBTITLE AND NOTES
# =====================================================================


def wrap_text_to_figure_width(
    text: str,
    left: float,
    right: float,
    **text_kwargs,
) -> list[str]:
    """Wrap text on word boundaries so each line fits left..right."""

    wrapped_lines: list[str] = []

    current_line = ""

    for word in text.split():

        candidate = (
            word
            if not current_line
            else f"{current_line} {word}"
        )

        probe = fig.text(
            left,
            0.5,
            candidate,
            **text_kwargs,
        )

        candidate_width = (
            probe.get_window_extent(
                renderer=renderer
            ).width
            / fig.bbox.width
        )

        probe.remove()

        if (
            left + candidate_width <= right
            or not current_line
        ):

            current_line = candidate

        else:

            wrapped_lines.append(
                current_line
            )

            current_line = word

    if current_line:

        wrapped_lines.append(
            current_line
        )

    return wrapped_lines


subtitle_right_limit = 0.97

subtitle_font_size = 18.0

subtitle_color = "#666666"

subtitle_lines = wrap_text_to_figure_width(
    subtitle,
    common_left_position,
    subtitle_right_limit,
    fontsize=subtitle_font_size,
)

if subtitle_lines:

    fig.text(
        common_left_position,
        SUBTITLE_Y,
        subtitle_lines[0],
        ha="left",
        va="top",
        fontsize=subtitle_font_size,
        color=subtitle_color,
    )

    if len(subtitle_lines) >= 2:

        fig.text(
            common_left_position,
            SUBTITLE_Y - SUBTITLE_LINE_SPACING,
            " ".join(
                subtitle_lines[1:]
            ),
            ha="left",
            va="top",
            fontsize=subtitle_font_size,
            color=subtitle_color,
        )

# =====================================================================
# SOURCE
# =====================================================================

fig.text(
    common_left_position,
    SOURCE_Y,
    "Source: Yahoo Finance",
    ha="left",
    va="bottom",
    fontsize=16,
    style="italic",
    color="#7d8597",
    alpha=1,
)

# =====================================================================
# DATA NOTE
# =====================================================================

data_note = (
    "Notes: Returns are cumulative holding-period returns calculated "
    "from Yahoo Finance daily closing spot exchange rates, expressed "
    "as USD per unit of foreign currency, and are not annualized; a "
    "positive return indicates the currency appreciated against the "
    "US dollar. Returns exclude carry and interest-rate differentials. "
    "DXY is shown as published (positive = US dollar strength); CEW is "
    "the WisdomTree Emerging Currency Strategy Fund share price. "
    "Observation dates may vary by market."
)

data_note_font_size = 16

data_note_color = "#7d8597"

data_note_alpha = 1

data_note_first_line_y_position = NOTE_Y

data_note_second_line_y_position = NOTE_Y - NOTE_LINE_SPACING

data_note_third_line_y_position = NOTE_Y - (2 * NOTE_LINE_SPACING)

data_note_lines = wrap_text_to_figure_width(
    data_note,
    common_left_position,
    common_right_position,
    fontsize=data_note_font_size,
    style="italic",
)

while len(data_note_lines) > 3 and data_note_font_size > 12:
    data_note_font_size -= 0.5
    data_note_lines = wrap_text_to_figure_width(
        data_note,
        common_left_position,
        common_right_position,
        fontsize=data_note_font_size,
        style="italic",
    )

data_note_y_positions = [
    data_note_first_line_y_position,
    data_note_second_line_y_position,
    data_note_third_line_y_position,
]

for line, y_position in zip(data_note_lines, data_note_y_positions):
    fig.text(
        common_left_position,
        y_position,
        line,
        ha="left",
        va="bottom",
        fontsize=data_note_font_size,
        style="italic",
        color=data_note_color,
        alpha=data_note_alpha,
    )

# =====================================================================
# Confirm Font Used
# =====================================================================

resolved = font_manager.findfont(
    font_manager.FontProperties(family="sans-serif", weight="bold")
)
print("Font used:", resolved)

# =====================================================================
# DISPLAY
# =====================================================================

plt.show()
