from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

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

# YEAR = 2025
# YEAR = 2024
# YEAR = None -> Rolling 12 Months
YEAR = None

# True includes the current partial month.
# False uses 12 completed months ending at the previous month-end.
INCLUDE_CURRENT_MONTH = True

FONT_FAMILY = "Helvetica"

# Extra calendar days requested before the first reporting month; the
# previous month-end observation is required to calculate the first
# monthly return.
DOWNLOAD_LOOKBACK_DAYS = 10

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

FONT_FAMILY = "Helvetica"

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


def is_placeholder_code(
    index_code: object,
) -> bool:
    """Return True when a ticker symbol is a placeholder."""

    cleaned_code = clean_index_code(
        index_code
    )

    if not cleaned_code:
        return True

    upper_code = cleaned_code.upper()

    exact_placeholders = {
        "NONE",
        "N/A",
        "NA",
        "TBC",
        "TBD",
        "PLACEHOLDER",
    }

    placeholder_prefixes = (
        "REPLACE",
        "REPLACE WITH",
        "REPLACE-WITH",
        "ADD",
        "INSERT",
        "ENTER",
        "PUT",
    )

    if upper_code in exact_placeholders:
        return True

    return upper_code.startswith(
        placeholder_prefixes
    )


def validate_ticker_code(
    index_name: str,
    index_code: object,
) -> str:
    """Validate and return a Yahoo Finance ticker symbol."""

    cleaned_code = clean_index_code(
        index_code
    )

    if not cleaned_code or " " in cleaned_code:
        raise ValueError(
            f"Yahoo Finance ticker for {index_name} must be a "
            f"non-empty symbol without spaces. "
            f"Received: {cleaned_code!r}"
        )

    return cleaned_code


# =====================================================================
# DATE RANGE
# =====================================================================

today = pd.Timestamp.today().normalize()

if YEAR is not None:

    start_date = pd.Timestamp(
        f"{YEAR}-01-01"
    )

    end_date = pd.Timestamp(
        f"{YEAR}-12-31"
    )

    chart_title = (
        f"Global Currency Returns vs USD {YEAR}"
    )

else:

    if INCLUDE_CURRENT_MONTH:

        end_date = today

        start_date = (
            today.replace(day=1)
            - pd.DateOffset(months=12)
        )

    else:

        end_date = (
            today.replace(day=1)
            - pd.Timedelta(days=1)
        )

        start_date = (
            end_date.to_period("M").start_time
            - pd.DateOffset(months=12)
        )

    chart_title = (
        "Global Currency Returns vs USD Over Last 12 Months"
    )

# Request only the dates needed for the calculation.
# The lookback captures the previous month-end observation
# needed to calculate the first displayed monthly return.
download_start_date = (
    start_date
    - pd.Timedelta(
        days=DOWNLOAD_LOOKBACK_DAYS
    )
)

# Do not request future observations when a current or future
# calendar year has been selected.
download_end_date = min(
    end_date,
    today,
)

if download_start_date >= download_end_date:
    raise ValueError(
        "The calculated download start date must be earlier "
        "than the calculated download end date."
    )

# =====================================================================
# PREPARE VALID TICKERS
# =====================================================================

valid_indices: dict[str, str] = {}

for index_name, index_code in ALL_INDICES.items():

    if is_placeholder_code(index_code):

        print(
            f"Skipping {index_name}: a confirmed Yahoo Finance "
            f"ticker has not been supplied."
        )

        continue

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

# Retain support for situations where more than one chart label
# is intentionally assigned the same ticker.
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

    invalid_row_count = int(
        invalid_rows.sum()
    )

    print(
        f"Removing {invalid_row_count} row(s) "
        f"with invalid dates or index levels."
    )

    hist = (
        hist.loc[
            ~invalid_rows
        ]
        .copy()
    )

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

hist = (
    hist.loc[
        ~unmatched_code_rows
    ]
    .copy()
)

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

missing_returned_codes = [
    index_code
    for index_code in unique_index_codes
    if index_code not in returned_codes
]

for missing_code in missing_returned_codes:

    affected_names = index_code_to_names.get(
        missing_code,
        [],
    )

    print(
        f"No Yahoo Finance history returned for ticker {missing_code}: "
        + ", ".join(
            affected_names
        )
    )

# =====================================================================
# CALCULATE MONTHLY RETURNS
# =====================================================================

rows: list[dict[str, object]] = []

requested_start_month = (
    start_date.to_period("M")
)

requested_end_month = (
    end_date.to_period("M")
)

previous_month = (
    requested_start_month - 1
)

for sector_name, index_code in ALL_INDICES.items():

    try:

        if is_placeholder_code(index_code):
            continue

        cleaned_index_code = validate_ticker_code(
            index_name=sector_name,
            index_code=index_code,
        )

        temp = (
            hist.loc[
                hist["INDEX_CODE"] == cleaned_index_code,
                [
                    "DATE",
                    "price",
                    "price_source",
                    "CURRENCY",
                ],
            ]
            .copy()
            .rename(
                columns={
                    "DATE": "date",
                }
            )
            .sort_values(
                "date"
            )
        )

        if temp.empty:

            print(
                f"No Yahoo Finance history returned for "
                f"{sector_name}: {cleaned_index_code}"
            )

            continue

        temp["observation_date"] = temp["date"]

        indexed_temp = temp.set_index(
            "date"
        )

        monthly_prices = (
            indexed_temp["price"]
            .resample("ME")
            .last()
            .dropna()
        )

        monthly_observation_dates = (
            indexed_temp["observation_date"]
            .resample("ME")
            .last()
            .reindex(
                monthly_prices.index
            )
        )

        monthly_price_sources = (
            indexed_temp["price_source"]
            .resample("ME")
            .last()
            .reindex(
                monthly_prices.index
            )
        )

        monthly_periods = (
            monthly_prices.index.to_period("M")
        )

        monthly_mask = (
            (monthly_periods >= previous_month)
            & (monthly_periods <= requested_end_month)
        )

        monthly_prices = (
            monthly_prices.loc[
                monthly_mask
            ]
        )

        monthly_observation_dates = (
            monthly_observation_dates.reindex(
                monthly_prices.index
            )
        )

        monthly_price_sources = (
            monthly_price_sources.reindex(
                monthly_prices.index
            )
        )

        if len(monthly_prices) < 2:

            print(
                f"Insufficient monthly history for "
                f"{sector_name}: {cleaned_index_code}"
            )

            continue

        monthly_returns = (
            monthly_prices
            .pct_change(
                fill_method=None
            )
            * 100
        )

        return_periods = (
            monthly_returns.index.to_period("M")
        )

        reporting_return_mask = (
            (return_periods >= requested_start_month)
            & (return_periods <= requested_end_month)
        )

        row: dict[str, object] = {
            "Sector": sector_name,
        }

        returned_currency = "USD"

        if (
            "CURRENCY" in temp.columns
            and not temp["CURRENCY"].dropna().empty
        ):

            returned_currency = str(
                temp["CURRENCY"].dropna().iloc[0]
            )

        for position, (dt, ret) in enumerate(
            monthly_returns.items()
        ):

            if not bool(reporting_return_mask[position]):
                continue

            if pd.isna(ret):
                continue

            if position == 0:
                continue

            month_label = dt.strftime("%b %y")

            row[month_label] = float(ret)

        reporting_prices_mask = (
            (
                monthly_prices.index.to_period("M")
                >= previous_month
            )
            & (
                monthly_prices.index.to_period("M")
                >= previous_month
            )
            & (
                monthly_prices.index.to_period("M")
                <= requested_end_month
            )
        )

        reporting_prices = (
            monthly_prices.loc[
                reporting_prices_mask
            ]
        )

        reporting_observation_dates = (
            monthly_observation_dates.reindex(
                reporting_prices.index
            )
        )

        reporting_price_sources = (
            monthly_price_sources.reindex(
                reporting_prices.index
            )
        )

        if len(reporting_prices) < 2:

            print(
                f"Insufficient reporting period data for "
                f"{sector_name}: {cleaned_index_code}"
            )

            continue

        total_return = (
            (
                float(
                    reporting_prices.iloc[-1]
                )
                / float(
                    reporting_prices.iloc[0]
                )
                - 1
            )
            * 100
        )

        row["Total"] = total_return

        total_start_observation_date = (
            reporting_observation_dates.iloc[0]
        )

        total_end_observation_date = (
            reporting_observation_dates.iloc[-1]
        )

        total_start_price = float(
            reporting_prices.iloc[0]
        )

        total_end_price = float(
            reporting_prices.iloc[-1]
        )

        total_start_price_source = (
            reporting_price_sources.iloc[0]
        )

        total_end_price_source = (
            reporting_price_sources.iloc[-1]
        )

        volatility_prices = (
            temp.loc[
                (
                    temp["date"]
                    >= total_start_observation_date
                )
                & (
                    temp["date"]
                    <= total_end_observation_date
                ),
                [
                    "date",
                    "price",
                ],
            ]
            .dropna(
                subset=[
                    "price",
                ]
            )
            .drop_duplicates(
                subset=[
                    "date",
                ],
                keep="last",
            )
            .sort_values(
                "date"
            )
            .set_index(
                "date"
            )["price"]
        )

        daily_returns = (
            volatility_prices
            .pct_change(
                fill_method=None
            )
            .dropna()
        )

        if len(daily_returns) >= 2:

            daily_sd = float(
                daily_returns.std(
                    ddof=1
                )
            )

            annualized_sd = (
                daily_sd
                * np.sqrt(252)
                * 100
            )

        else:

            daily_sd = np.nan

            annualized_sd = np.nan

        row[
            "Annualized SD"
        ] = annualized_sd

        rows.append(
            row
        )

    except Exception as exc:

        print(
            f"Failed for {sector_name}: {exc}"
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
    "Sector"
)

month_cols = [
    column
    for column in combined_df.columns
    if column not in [
        "Total",
        "Annualized SD",
    ]
]

combined_df = combined_df[
    month_cols
    + [
        "Total",
        "Annualized SD",
    ]
]

combined_df = combined_df.sort_values(
    "Total",
    ascending=False,
)

# =====================================================================
# DYNAMIC INSTITUTIONAL INSIGHT-LED SUBTITLE
# =====================================================================

subtitle = (
    "Monthly currency returns against the US dollar across major "
    "economies, with cumulative period performance and annualised "
    "volatility"
)

insight_df = (
    combined_df[
        [
            "Total",
            "Annualized SD",
        ]
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .dropna(
        subset=[
            "Total",
            "Annualized SD",
        ]
    )
    .copy()
)

if not insight_df.empty:

    return_ranking = (
        insight_df["Total"]
        .sort_values(
            ascending=False
        )
    )

    volatility_ranking = (
        insight_df["Annualized SD"]
        .sort_values(
            ascending=False
        )
    )

    highest_return_name = str(
        return_ranking.index[0]
    )

    highest_return_value = float(
        return_ranking.iloc[0]
    )

    lowest_return_name = str(
        return_ranking.index[-1]
    )

    lowest_return_value = float(
        return_ranking.iloc[-1]
    )

    if len(return_ranking) >= 2:

        second_return_name = str(
            return_ranking.index[1]
        )

        second_return_value = float(
            return_ranking.iloc[1]
        )

    else:

        second_return_name = ""

        second_return_value = np.nan

    most_volatile_name = str(
        volatility_ranking.index[0]
    )

    most_volatile_value = float(
        volatility_ranking.iloc[0]
    )

    least_volatile_name = str(
        volatility_ranking.index[-1]
    )

    least_volatile_value = float(
        volatility_ranking.iloc[-1]
    )

    efficiency_df = insight_df.loc[
        insight_df["Annualized SD"] > 0
    ].copy()

    efficiency_df[
        "Return-to-Volatility Ratio"
    ] = (
        efficiency_df["Total"]
        / efficiency_df["Annualized SD"]
    )

    efficiency_df = (
        efficiency_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=[
                "Return-to-Volatility Ratio",
            ]
        )
    )

    if not efficiency_df.empty:

        efficiency_name = str(
            efficiency_df[
                "Return-to-Volatility Ratio"
            ]
            .idxmax()
        )

        efficiency_value = float(
            efficiency_df.loc[
                efficiency_name,
                "Return-to-Volatility Ratio",
            ]
        )

        if second_return_name:

            subtitle = (
                f"{highest_return_name} led currency "
                f"returns against the US dollar at "
                f"{highest_return_value:+.1f}%, "
                f"followed by {second_return_name} at "
                f"{second_return_value:+.1f}%, while "
                f"{lowest_return_name} lagged at "
                f"{lowest_return_value:+.1f}%. "
                f"{efficiency_name} delivered the strongest "
                f"return relative to volatility at "
                f"{efficiency_value:.2f}x; "
                f"{least_volatile_name} recorded the lowest "
                f"volatility at {least_volatile_value:.1f}%, "
                f"compared with {most_volatile_name} at "
                f"{most_volatile_value:.1f}%"
            )

        else:

            subtitle = (
                f"{highest_return_name} led currency "
                f"returns against the US dollar at "
                f"{highest_return_value:+.1f}%, "
                f"while {lowest_return_name} lagged at "
                f"{lowest_return_value:+.1f}%. "
                f"{efficiency_name} delivered the strongest "
                f"return relative to volatility at "
                f"{efficiency_value:.2f}x; "
                f"{least_volatile_name} recorded the lowest "
                f"volatility at {least_volatile_value:.1f}%, "
                f"compared with {most_volatile_name} at "
                f"{most_volatile_value:.1f}%"
            )

# =====================================================================
# PREPARE HEATMAP DATA
# =====================================================================

heatmap_df = combined_df[
    month_cols
    + [
        "Total",
    ]
].copy()

annualized_sd_series = (
    combined_df["Annualized SD"]
    .copy()
)

annot_df = heatmap_df.map(
    lambda value: (
        f"{value:+.1f}%"
        if pd.notna(value)
        else ""
    )
)

# =====================================================================
# INDEPENDENT MONTHLY AND TOTAL HEATMAP SCALES
# =====================================================================

monthly_heatmap_df = (
    heatmap_df.copy()
)

monthly_heatmap_df["Total"] = np.nan

monthly_annot_df = (
    annot_df.copy()
)

monthly_annot_df["Total"] = ""

total_heatmap_df = (
    heatmap_df.copy()
)

total_heatmap_df.loc[
    :,
    month_cols,
] = np.nan

total_annot_df = pd.DataFrame(
    "",
    index=annot_df.index,
    columns=annot_df.columns,
)

total_annot_df["Total"] = (
    annot_df["Total"]
)

# =====================================================================
# PLOT
# =====================================================================

number_of_rows = len(
    heatmap_df
)

total_idx = len(
    month_cols
)

sd_column_index = (
    total_idx + 1
)

total_number_of_columns = (
    sd_column_index + 1
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

ax = fig.add_axes(
    [
        common_left_position,
        HEAT_MAP_Y,
        common_right_position
        - common_left_position,
        0.74,
    ]
)

fig.canvas.draw()

y_axis_label_font = FontProperties(
    family=FONT_FAMILY,
    size=16,
    weight="bold",
)

maximum_label_width_pixels = max(
    renderer.get_text_width_height_descent(
        str(row_label),
        y_axis_label_font,
        ismath=False,
    )[0]
    for row_label in heatmap_df.index
)

heatmap_axis_width_pixels = (
    ax.get_window_extent(
        renderer=renderer
    ).width
)

pixels_per_heatmap_column = (
    heatmap_axis_width_pixels
    / total_number_of_columns
)

y_axis_label_horizontal_padding = 0.6

y_axis_label_column_width = (
    maximum_label_width_pixels
    / pixels_per_heatmap_column
    + y_axis_label_horizontal_padding
)

# ---------------------------------------------------------------------
# Heatmaps (independent scales)
# ---------------------------------------------------------------------

sns.heatmap(
    monthly_heatmap_df,
    cmap="RdYlGn",
    center=0,
    annot=monthly_annot_df,
    fmt="",
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold",
    },
    mask=monthly_heatmap_df.isna(),
    cbar=False,
    linewidths=1,
    linecolor="white",
)

sns.heatmap(
    total_heatmap_df,
    cmap="RdYlGn",
    center=0,
    annot=total_annot_df,
    fmt="",
    annot_kws={
        "fontsize": 15,
        "fontweight": "bold",
    },
    mask=total_heatmap_df.isna(),
    cbar=False,
    linewidths=1,
    linecolor="white",
)

ax.set_xlabel("")

ax.set_ylabel("")

ax.set_xticks([])

# =====================================================================
# ANNUALIZED SD DATA BARS (LEFT TO RIGHT GRADIENT)
# =====================================================================

finite_sd_values = (
    annualized_sd_series
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .dropna()
)

if finite_sd_values.empty:

    maximum_sd = 1.0

else:

    maximum_sd = float(
        finite_sd_values.max()
    )

if maximum_sd <= 0:

    maximum_sd = 1.0

sd_gradient_cmap = (
    LinearSegmentedColormap.from_list(
        "AnnualizedSDGradient",
        [
            "#F2F2F2",
            "#FFE8E8",
            "#FFCACA",
            "#FF9D9D",
            "#FF7070",
            "#FF4D4D",
        ],
    )
)

for row_position, sd_value in enumerate(
    annualized_sd_series
):

    cell_x = sd_column_index

    cell_y = row_position

    # Cell background
    ax.add_patch(
        Rectangle(
            (
                cell_x,
                cell_y,
            ),
            1,
            1,
            facecolor="white",
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
    )

    if pd.notna(sd_value) and np.isfinite(sd_value):

        normalized_sd = float(
            np.clip(
                sd_value / maximum_sd,
                0,
                1,
            )
        )

        if normalized_sd > 0:

            gradient = np.linspace(
                0,
                normalized_sd,
                512,
            ).reshape(
                1,
                -1,
            )

            ax.imshow(
                gradient,
                aspect="auto",
                cmap=sd_gradient_cmap,
                interpolation="bicubic",
                extent=[
                    cell_x,
                    cell_x + normalized_sd,
                    cell_y + 1,
                    cell_y,
                ],
                vmin=0,
                vmax=1,
                origin="upper",
                zorder=3,
            )

        ax.text(
            cell_x + 0.5,
            cell_y + 0.5,
            f"{sd_value:.1f}%",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="black",
            zorder=5,
        )

    ax.add_patch(
        Rectangle(
            (
                cell_x,
                cell_y,
            ),
            1,
            1,
            facecolor="none",
            edgecolor="white",
            linewidth=1,
            zorder=6,
        )
    )

# imshow resets the axis limits, so apply the table limits afterwards
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

x_axis_labels = (
    list(month_cols)
    + [
        "Total",
        "Std Dev",
    ]
)

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

        y_axis_label_fill = "#a8dadc"

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
        + 0.15,
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
    + 0.15,
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

# Bottom border of the label column
ax.plot(
    [
        -y_axis_label_column_width,
        0,
    ],
    [
        number_of_rows,
        number_of_rows,
    ],
    color="#1b263b",
    linewidth=2.5,
    zorder=24,
    clip_on=False,
)

# Vertical border between the label column and the data columns
ax.axvline(
    0,
    ymin=0,
    ymax=1,
    color="#1b263b",
    linewidth=2.5,
    zorder=14,
    clip_on=False,
)

ax.plot(
    [
        0,
        0,
    ],
    [
        -1,
        0,
    ],
    color="#1b263b",
    linewidth=2.5,
    zorder=14,
    clip_on=False,
)

# Total column borders
ax.axvline(
    total_idx,
    color="#1b263b",
    linewidth=2.5,
    zorder=8,
)

ax.axvline(
    sd_column_index,
    color="#1b263b",
    linewidth=2.5,
    zorder=8,
)

ax.plot(
    [
        total_idx,
        total_idx,
    ],
    [
        -1,
        0,
    ],
    color="#1b263b",
    linewidth=2.5,
    zorder=16,
    clip_on=False,
)

ax.plot(
    [
        sd_column_index,
        sd_column_index,
    ],
    [
        -1,
        0,
    ],
    color="#1b263b",
    linewidth=2.5,
    zorder=16,
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
        zorder=23,
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
# DYNAMIC WORD-WRAPPED SUBTITLE
#
# The subtitle begins at the same position as the outside
# left border of the table.
#
# It wraps at the final complete word that fits before
# figure position 0.97.
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

subtitle_first_line_y_position = SUBTITLE_Y

subtitle_second_line_y_position = SUBTITLE_Y - SUBTITLE_LINE_SPACING

subtitle_font_size = 18.0

subtitle_color = "#666666"

subtitle_lines = wrap_text_to_figure_width(
    subtitle,
    common_left_position,
    subtitle_right_limit,
    fontsize=subtitle_font_size,
)

# =====================================================================
# DRAW SUBTITLE
# =====================================================================

if subtitle_lines:

    first_subtitle_line = (
        subtitle_lines[0]
    )

    fig.text(
        common_left_position,
        subtitle_first_line_y_position,
        first_subtitle_line,
        ha="left",
        va="top",
        fontsize=subtitle_font_size,
        color=subtitle_color,
    )

    if len(subtitle_lines) >= 2:

        second_subtitle_line = (
            " ".join(
                subtitle_lines[1:]
            )
        )

        fig.text(
            common_left_position,
            subtitle_second_line_y_position,
            second_subtitle_line,
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
    "Notes: Returns are calculated from Yahoo Finance daily closing "
    "spot exchange rates, expressed as USD per unit of foreign "
    "currency; a positive return indicates the currency appreciated "
    "against the US dollar. Returns exclude carry and interest-rate "
    "differentials. The latest month may be incomplete and observation "
    "dates may vary by market. DXY is shown as published, so a "
    "positive value indicates US dollar strength against its basket; "
    "CEW is the WisdomTree Emerging Currency Strategy Fund share "
    "price. Annualized volatility is calculated from daily "
    "closing-rate returns using a 252-trading-day convention."
)

data_note_font_size = 16

data_note_color = "#7d8597"

data_note_alpha = 1

data_note_first_line_y_position = NOTE_Y

data_note_second_line_y_position = NOTE_Y - NOTE_LINE_SPACING

data_note_third_line_y_position = NOTE_Y - (2 * NOTE_LINE_SPACING)

fig.canvas.draw()

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
