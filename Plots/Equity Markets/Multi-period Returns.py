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
# IMPORT MSCI-DATA
# =====================================================================

try:
    from mscidata import msci
except ImportError as error:
    raise ImportError(
        "\nCould not import 'msci' from the 'mscidata' package.\n\n"
        "Install or upgrade the required packages using:\n\n"
        "    pip install --upgrade msci-data pandas\n\n"
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

MSCI_RETURN_VARIANT = "NETR"

# Extra calendar days requested before the longest look-back, so the
# observation on or before the 10Y target date is always captured.
DOWNLOAD_LOOKBACK_DAYS = 15

# =====================================================================
# OUTPUT LOCATION
# =====================================================================

OUTPUT_DIR = Path(__file__).resolve().parent

PNG_FILE = (
    OUTPUT_DIR
    / "Global_Market_Returns_Heatmap.png"
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
# MAJOR GLOBAL EQUITY MARKETS
# MSCI NET TOTAL RETURN INDICES IN USD
# =====================================================================

SECTORS = {
    "United States": "984000",
    "Canada": "912400",
    "United Kingdom": "982600",
    "Eurozone": "106400",
    "Germany": "928000",
    "France": "925000",
    "Italy": "938000",
    "Spain": "972400",
    "Switzerland": "975600",
    "Japan": "939200",
    "China": "302400",
    "India": "935600",
    "South Korea": "941000",
    "Taiwan": "915800",
    "Hong Kong": "934400",
    "Australia": "903600",
    "Brazil": "907600",
    "Mexico": "848400",
    "South Africa": "971000",
    "UAE": "133717",
    "Singapore": "998100",
}

# =====================================================================
# GLOBAL EQUITY BENCHMARKS
# =====================================================================

BENCHMARKS = {
    "MSCI World": "990100",
    "MSCI Emerging Markets": "891800",
    "MSCI All Country": "892400",
    "MSCI Ex-AMER": "990300",
    "MSCI World Momentum": "703755",
}

ALL_INDICES = {
    **SECTORS,
    **BENCHMARKS,
}

BENCHMARK_NAMES = list(
    BENCHMARKS
)

# =====================================================================
# MSCI INDEX-CODE VALIDATION
# =====================================================================


def clean_index_code(
    index_code: object,
) -> str:
    """Convert an MSCI index code to a clean string."""

    if index_code is None:
        return ""

    return str(index_code).strip()


def validate_numeric_index_code(
    index_name: str,
    index_code: object,
) -> str:
    """Validate and return an MSCI numeric index code."""

    cleaned_code = clean_index_code(
        index_code
    )

    if not cleaned_code:
        raise ValueError(
            f"No MSCI index code was supplied for {index_name}."
        )

    if not cleaned_code.isdecimal():
        raise ValueError(
            f"MSCI index code for {index_name} must contain "
            f"digits only. Received: {cleaned_code!r}"
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
    "Global Equity Markets Multi-Period Returns"
)

# =====================================================================
# PREPARE VALID MSCI INDEX CODES
# =====================================================================

valid_indices: dict[str, str] = {}

for index_name, index_code in ALL_INDICES.items():

    cleaned_index_code = validate_numeric_index_code(
        index_name=index_name,
        index_code=index_code,
    )

    valid_indices[
        index_name
    ] = cleaned_index_code

if not valid_indices:
    raise ValueError(
        "No valid numeric MSCI index codes were supplied."
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
# DOWNLOAD MSCI NET TOTAL RETURN DATA
# =====================================================================

print()

print(
    "Downloading MSCI Net Total Return index data..."
)

print(
    f"Requested period: "
    f"{download_start_date:%Y-%m-%d} to "
    f"{download_end_date:%Y-%m-%d}"
)

print(
    f"Requesting {len(unique_index_codes)} unique "
    f"MSCI index code(s) in one batched request."
)

download_started = time.perf_counter()

try:

    downloaded_data = msci.get_levels(
        unique_index_codes,
        download_start_date.strftime(
            "%Y-%m-%d"
        ),
        download_end_date.strftime(
            "%Y-%m-%d"
        ),
        MSCI_RETURN_VARIANT,
    )

except Exception as error:

    raise RuntimeError(
        f"The batched MSCI data request failed. "
        f"Original error: {error}"
    ) from error

download_seconds = (
    time.perf_counter()
    - download_started
)

if downloaded_data is None:
    raise ValueError(
        "No data returned from MSCI."
    )

hist = pd.DataFrame(
    downloaded_data
)

if hist.empty:
    raise ValueError(
        "MSCI returned an empty dataset."
    )

print(
    f"MSCI download completed in "
    f"{download_seconds:.2f} seconds."
)

print(
    f"Raw observations returned: "
    f"{len(hist):,}"
)

# =====================================================================
# NORMALISE MSCI DATA
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
        "MSCI response is missing required column(s): "
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

# Protect against numeric codes being returned as 990100.0
hist["INDEX_CODE"] = (
    hist["INDEX_CODE"]
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
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
        "No valid MSCI observations remained after "
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
        "Removing observations for unrecognised MSCI "
        "index code(s): "
        + ", ".join(
            unmatched_codes
        )
    )

hist = hist.loc[
    ~unmatched_code_rows
].copy()

if hist.empty:
    raise ValueError(
        "No observations matched the requested MSCI "
        "index codes."
    )

if "CURRENCY" not in hist.columns:
    hist["CURRENCY"] = "USD"

hist["REQUESTED_INDEX_CODE"] = hist["INDEX_CODE"]
hist["REQUESTED_VARIANT"] = MSCI_RETURN_VARIANT

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

hist["price_source"] = "MSCI Net Total Return Index Level"

returned_codes = set(
    hist["INDEX_CODE"]
    .unique()
    .tolist()
)

for missing_code in unique_index_codes:

    if missing_code not in returned_codes:

        print(
            f"No MSCI history returned for code {missing_code}: "
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
                f"No MSCI history returned for "
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
    "Cross-market performance across short-, medium- and long-term horizons"
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
        f"{str(one_year_ranking.index[0])} led one-year returns at "
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
    "Economies",
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
    "Markets",
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
    "Source: MSCI",
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
    "from MSCI Net Total Return index levels in USD and are not "
    "annualized. NETR incorporates reinvested dividends after "
    "applicable withholding-tax assumptions; observation dates may "
    "vary by market."
)

data_note_font_size = 16

data_note_color = "#7d8597"

data_note_alpha = 1

data_note_lines = wrap_text_to_figure_width(
    data_note,
    common_left_position,
    common_right_position,
    fontsize=data_note_font_size,
    style="italic",
)

if data_note_lines:

    fig.text(
        common_left_position,
        NOTE_Y,
        data_note_lines[0],
        ha="left",
        va="bottom",
        fontsize=data_note_font_size,
        style="italic",
        color=data_note_color,
        alpha=data_note_alpha,
    )

    if len(data_note_lines) >= 2:

        fig.text(
            common_left_position,
            NOTE_Y - NOTE_LINE_SPACING,
            " ".join(
                data_note_lines[1:]
            ),
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