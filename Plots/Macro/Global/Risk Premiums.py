from __future__ import annotations

"""Regional implied ERP dashboard.

The UK/EU historical consensus-growth and aggregate payout inputs are not in
public, point-in-time downloadable datasets. This script therefore never
fabricates them. It creates a clearly-labelled input workbook/CSV template and
computes results only for rows with complete, sourced inputs. Populate that
template from the named vendor/public sources before treating regional ERPs as
estimates. Damodaran's US published ERP is fetched separately as a reference.
"""

from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
import io

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Configuration and input contract
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "Risk Premiums Regional Inputs.csv"
OUTPUT_TABLE = HERE / "Risk Premiums Calculation Table.csv"
OUTPUT_QOQ = HERE / "Risk Premiums QoQ Changes.csv"
OUTPUT_SENSITIVITY = HERE / "Risk Premiums Sensitivity.csv"
OUTPUT_SOURCES = HERE / "Risk Premiums Sources.csv"
OUTPUT_CHART = HERE / "Risk Premiums.png"

AS_OF = pd.Timestamp.today().normalize()
N_QUARTERS = 12
PERIODS = list(pd.date_range(end=AS_OF, periods=N_QUARTERS + 1, freq="QE"))
PERIODS[-1] = AS_OF
US_DATA_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls"
US_PAGE = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/implpr.html"

REGIONS = {
    "UK": {"index": "FTSE All-Share; FTSE 100 fallback", "currency": "GBP",
           "rf": "Bank of England IUDMNZC, 10y nominal zero-coupon gilt"},
    "Europe": {"index": "STOXX Europe 600; Euro Stoxx 50 fallback", "currency": "EUR",
               "rf": "Bundesbank BBSSY.D.REN.EUR.A630.000000WT1010.A, 10y current Bund"},
}
SOURCE_REGISTER = [
    ("Bank of England", "IUDMNZC — 10-year nominal zero-coupon gilt yield; daily", "https://www.bankofengland.co.uk/statistics/yield-curves", "Official daily curve; no API; archive may be revised."),
    ("Deutsche Bundesbank", "BBSSY.D.REN.EUR.A630.000000WT1010.A — daily current 10-year federal bond yield", "https://www.bundesbank.de/en/statistics/money-and-capital-markets/interest-rates-and-yields/daily-yields-of-current-federal-securities-772220", "Use last available observation on or before the valuation date; current-bond series, not a constant-maturity fitted curve."),
    ("FTSE Russell / London Stock Exchange", "FTSE All-Share price index level (fallback FTSE 100); source field: official closing index value", "https://www.londonstockexchange.com/indices/ftse-all-share", "Historical index close may require licensed historical data."),
    ("STOXX", "SXXP — STOXX Europe 600 price index; SXXPD — STOXX Europe 600 Dividend Points", "https://stoxx.com/index/sxxpd/", "Dividend points are cumulative gross cash dividends; historical files may require data access/license."),
    ("Computershare", "UK Dividend Monitor — aggregate dividend payment totals", "https://www.computershare.com/uk/insights/share-registry/dividend-monitor", "Broad UK-market proxy only; not an index-constituent point-in-time dividend yield."),
    ("UK Financial Conduct Authority", "Share buybacks in UK-listed equities — Bloomberg-based FTSE 350 annual repurchases", "https://www.fca.org.uk/publication/multi-firm-reviews/share-buybacks-uk-listed-equities.pdf", "Annual FTSE 350 totals do not provide quarter-level FTSE All-Share input; use only as explicitly estimated proxy."),
    ("Bloomberg / LSEG I/B/E/S", "Index-level 5-year forward EPS growth consensus, as-of vintage for each valuation date", "https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes", "Requires licensed point-in-time consensus history; current consensus is not an acceptable substitute."),
    ("Bloomberg / FactSet / LSEG", "Aggregate index constituent repurchases / buybacks, cash value, trailing 12 months", "https://www.factset.com/", "Requires licensed constituent-level or index aggregate history; flag all estimates."),
    ("Aswath Damodaran, NYU Stern", "histimpl.xls — historical US implied premium (DDM) monthly series", "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/implpr.html", "US reference only; monthly observation carried to each quarter-end without forward filling."),
]
INPUT_COLUMNS = [
    "region", "as_of", "index_used", "index_level", "dividend_yield_pct",
    "buyback_yield_pct", "estimated_buyback_yield_pct", "growth_5y_pct", "risk_free_pct", "input_status",
    "index_source_field", "dividend_source_field", "buyback_source_field",
    "growth_source_field", "risk_free_source_field", "retrieved_on", "notes",
]


def make_input_template() -> None:
    """Create rows for all required dates; blank inputs remain visibly blank."""
    rows = []
    for region, spec in REGIONS.items():
        for period in PERIODS:
            rows.append({
                "region": region, "as_of": period.strftime("%Y-%m-%d"),
                "index_used": spec["index"], "index_level": np.nan,
                "dividend_yield_pct": np.nan, "buyback_yield_pct": np.nan,
                "estimated_buyback_yield_pct": np.nan,
                "growth_5y_pct": np.nan, "risk_free_pct": np.nan,
                "input_status": "MISSING — populate from sources below",
                "index_source_field": "", "dividend_source_field": "",
                "buyback_source_field": "", "growth_source_field": "",
                "risk_free_source_field": "", "retrieved_on": "",
                "notes": "Do not substitute current consensus for historical point-in-time consensus.",
            })
    pd.DataFrame(rows, columns=INPUT_COLUMNS).to_csv(INPUT_FILE, index=False)


def solve_irr(index_level: float, div_yield: float, buyback_yield: float,
              growth: float, rf: float) -> float:
    """Two-stage FCFE DDM with terminal growth equal to local risk-free rate.

    The starting TTM cash flow is normalized to index points as P0 times the
    sourced dividend/buyback yields. Years 1–5 grow at the point-in-time
    consensus rate. Thereafter cash flow grows at rf. Solve PV = index level.
    """
    if index_level <= 0 or growth <= -1 or rf <= -1:
        return np.nan
    c0 = index_level * (div_yield + buyback_yield)

    def residual(r: float) -> float:
        if r <= rf:
            return np.inf
        cashflows = [c0 * (1 + growth) ** year for year in range(1, 6)]
        pv_explicit = sum(cf / (1 + r) ** year for year, cf in enumerate(cashflows, 1))
        terminal = cashflows[-1] * (1 + rf) / (r - rf)
        return pv_explicit + terminal / (1 + r) ** 5 - index_level

    lo = max(rf + 1e-8, -0.999999)
    hi = 5.0
    try:
        return brentq(residual, lo, hi, maxiter=1000)
    except (ValueError, RuntimeError, OverflowError):
        return np.nan


def calculate(row: pd.Series, buyback_case: str, growth_shift_pp: float = 0) -> dict:
    values = [row.get("index_level"), row.get("dividend_yield_pct"), row.get("growth_5y_pct"), row.get("risk_free_pct")]
    if any(pd.isna(v) for v in values):
        return {"irr_pct": np.nan, "erp_pct": np.nan, "quality": "MISSING REQUIRED INPUT"}
    bb = row.get("buyback_yield_pct")
    if buyback_case == "dividends only":
        bb_yield, quality = 0.0, "Buybacks excluded by scenario"
    else:
        estimated = row.get("estimated_buyback_yield_pct")
        if pd.notna(bb):
            bb_yield, quality = float(bb) / 100, str(row.get("input_status", "Sourced buyback"))
        elif pd.notna(estimated):
            bb_yield, quality = float(estimated) / 100, "ESTIMATED BUYBACK — see row notes"
        else:
            return {"irr_pct": np.nan, "erp_pct": np.nan, "quality": "MISSING BUYBACK INPUT"}
    rf = float(row["risk_free_pct"]) / 100
    irr = solve_irr(float(row["index_level"]), float(row["dividend_yield_pct"]) / 100,
                    bb_yield, float(row["growth_5y_pct"]) / 100 + growth_shift_pp / 100, rf)
    return {"irr_pct": irr * 100 if np.isfinite(irr) else np.nan,
            "erp_pct": (irr - rf) * 100 if np.isfinite(irr) else np.nan,
            "quality": quality if np.isfinite(irr) else "NO VALID IRR ROOT"}


def fetch_us_reference() -> pd.DataFrame:
    req = Request(US_DATA_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; macro-chart/1.0)"})
    with urlopen(req, timeout=30) as response:
        raw = response.read()
    # The Damodaran workbook is an Excel 97-2003 file. Header names have changed
    # over time; locate the date and implied-premium columns semantically.
    book = pd.ExcelFile(io.BytesIO(raw))
    frame = pd.read_excel(book, sheet_name=0, header=None)
    header_row = None
    for i in range(min(12, len(frame))):
        joined = " ".join(str(v).lower() for v in frame.iloc[i].dropna().tolist())
        if "date" in joined and ("implied" in joined or "premium" in joined):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not identify columns in Damodaran histimpl.xls")
    data = frame.iloc[header_row + 1:].copy()
    data.columns = [str(v).strip() for v in frame.iloc[header_row].tolist()]
    date_col = next(c for c in data if "date" in c.lower())
    erp_col = next(c for c in data if "implied" in c.lower() and ("premium" in c.lower() or "erp" in c.lower()))
    dates = pd.to_datetime(data[date_col], errors="coerce")
    erp = pd.to_numeric(data[erp_col].astype(str).str.replace("%", "", regex=False), errors="coerce")
    # Excel values may be stored as decimal fractions or percentage points.
    if erp.dropna().abs().median() < 0.2:
        erp *= 100
    out = pd.DataFrame({"date": dates, "us_erp_pct": erp}).dropna()
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out


def main() -> None:
    if not INPUT_FILE.exists():
        make_input_template()
        print(f"Created regional input template: {INPUT_FILE}")
        print("UK and Europe ERPs remain blank until the point-in-time inputs are sourced and entered.")

    inputs = pd.read_csv(INPUT_FILE)
    inputs["as_of"] = pd.to_datetime(inputs["as_of"], errors="coerce")
    inputs = inputs[inputs["region"].isin(REGIONS)].copy()
    pd.DataFrame(SOURCE_REGISTER, columns=["publisher", "series_or_field", "source_url", "limitations"]).assign(
        retrieved_on=date.today().isoformat()
    ).to_csv(OUTPUT_SOURCES, index=False)
    result_rows = []
    for _, row in inputs.iterrows():
        for case in ("dividends only", "dividends + buybacks"):
            result = calculate(row, case)
            result_rows.append({**row.to_dict(), "scenario": case, **result,
                                "formula": "P0 = Σ(CF_t/(1+r)^t), t=1..5 + [CF5(1+rf)/(r-rf)]/(1+r)^5; ERP=r-rf"})
    results = pd.DataFrame(result_rows).sort_values(["region", "as_of", "scenario"])
    results.to_csv(OUTPUT_TABLE, index=False)

    qoq_rows = []
    for region, group in results.groupby("region"):
        for scenario, sg in group.groupby("scenario"):
            sg = sg.sort_values("as_of")
            for _, current in sg.iterrows():
                previous = sg[sg["as_of"] < current["as_of"]].tail(1)
                if previous.empty:
                    continue
                prev = previous.iloc[0]
                qoq_rows.append({"region": region, "as_of": current["as_of"], "scenario": scenario,
                                 "risk_free_change_bp": (current["risk_free_pct"] - prev["risk_free_pct"])*100,
                                 "erp_change_bp": (current["erp_pct"] - prev["erp_pct"])*100,
                                 "cost_of_equity_change_bp": (current["irr_pct"] - prev["irr_pct"])*100})
    qoq = pd.DataFrame(qoq_rows)
    qoq.to_csv(OUTPUT_QOQ, index=False)

    headline = qoq[qoq.scenario.eq("dividends + buybacks")].dropna(subset=["cost_of_equity_change_bp"])
    if not headline.empty:
        biggest = headline.reindex(headline.cost_of_equity_change_bp.abs().sort_values(ascending=False).index).head(4)
        for _, move in biggest.iterrows():
            driver = "risk-free rates" if abs(move.risk_free_change_bp) > abs(move.erp_change_bp) else "implied ERP"
            print(f"Largest QoQ move: {move.region} {pd.Timestamp(move.as_of).date()}: "
                  f"cost of equity {move.cost_of_equity_change_bp:+.0f} bp; mostly {driver} "
                  f"(rates {move.risk_free_change_bp:+.0f} bp, ERP {move.erp_change_bp:+.0f} bp). "
                  "Separating price, growth-revision and payout effects requires sourced point-in-time inputs.")

    sens_rows = []
    for _, row in inputs.iterrows():
        for case in ("dividends only", "dividends + buybacks"):
            for shift in (-1, 0, 1):
                val = calculate(row, case, shift)
                sens_rows.append({"region": row["region"], "as_of": row["as_of"], "scenario": case,
                                  "growth_shift_pp": shift, **val})
    pd.DataFrame(sens_rows).to_csv(OUTPUT_SENSITIVITY, index=False)

    try:
        us = fetch_us_reference()
    except Exception as exc:
        print(f"US Damodaran reference unavailable: {exc}")
        us = pd.DataFrame(columns=["date", "us_erp_pct"])
    if not us.empty:
        us = us.sort_values("date").drop_duplicates("date", keep="last")
        reference_dates = sorted(set(PERIODS))
        picked = []
        for target in reference_dates:
            available = us[us.date <= target]
            if not available.empty:
                observation = available.iloc[-1]
                picked.append({"date": observation["date"],
                               "reference_date": target,
                               "us_erp_pct": observation["us_erp_pct"]})
        latest_by_date = pd.DataFrame(picked)
    else:
        latest_by_date = us

    # Slide styling follows the established macro chart system.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"]})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = {"UK": "#e8856a", "Europe": "#6b8fd1"}
    xdates = sorted(inputs["as_of"].dropna().unique())
    offsets = {"UK": -0.12, "Europe": 0.12}
    width = 0.20
    for region in REGIONS:
        subset = results[(results.region == region) & (results.scenario == "dividends + buybacks")].sort_values("as_of")
        xs = np.array([pd.Timestamp(d).toordinal() for d in subset.as_of])
        rfs = subset.risk_free_pct.to_numpy(dtype=float)
        erps = subset.erp_pct.to_numpy(dtype=float)
        xpos = xs + offsets[region] * 90
        ax.bar(xpos, rfs, width=width*90, color=colors[region], edgecolor="white", linewidth=.5, label=f"{region} risk-free rate")
        ax.bar(xpos, erps, width=width*90, bottom=rfs, color=colors[region], alpha=.52, edgecolor="white", linewidth=.5, hatch="///", label=f"{region} implied ERP")
        for x, a, b in zip(xpos, rfs, erps):
            if np.isfinite(a) and np.isfinite(b):
                ax.text(x, a/2, f"{a:.1f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
                ax.text(x, a+b/2, f"{b:.1f}%", ha="center", va="center", fontsize=7, color="#222222")
                ax.text(x, a+b+.12, f"{a+b:.1f}%", ha="center", va="bottom", fontsize=7, color="#222222")

    if not latest_by_date.empty:
        ax.plot(latest_by_date.reference_date.map(pd.Timestamp.toordinal), latest_by_date.us_erp_pct,
                color="#fcb149", linewidth=2.5, marker="o", markersize=3,
                label="US Damodaran published ERP (reference only)")
    ax.set_ylabel("Cost of equity / implied ERP (%)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xticks([pd.Timestamp(d).toordinal() for d in xdates])
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %Y") for d in xdates], rotation=45, ha="right")
    ax.grid(axis="y", color="#b9b9b9", linestyle="--", linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#7f7f7f")
    ax.spines["bottom"].set_color("#7f7f7f")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Regional Implied Equity Risk Premiums", x=.04, y=.96, ha="left", fontsize=23, fontweight="bold")
    fig.text(.04, .91, "Quarter-end build-up: local 10-year risk-free rate + region-specific implied ERP; US ERP is reference only.", fontsize=11)
    # Compact latest QoQ table under the chart; the CSV contains the full
    # quarter-by-quarter series for every region and both payout cases.
    latest_qoq = qoq[qoq.scenario.eq("dividends + buybacks")].dropna(subset=["cost_of_equity_change_bp"])
    if not latest_qoq.empty:
        rows = []
        for region in REGIONS:
            rg = latest_qoq[latest_qoq.region.eq(region)].sort_values("as_of")
            if not rg.empty:
                v = rg.iloc[-1]
                rows.append([region, f"{v.risk_free_change_bp:+.0f} bp",
                             f"{v.erp_change_bp:+.0f} bp", f"{v.cost_of_equity_change_bp:+.0f} bp"])
        if rows:
            table_ax = fig.add_axes([.32, .115, .52, .105])
            table_ax.axis("off")
            table = table_ax.table(cellText=rows,
                                   colLabels=["Latest QoQ", "Risk-free", "ERP", "Total cost of equity"],
                                   cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.35)
    fig.text(.04, .04,
             "Sources: BoE IUDMNZC; Bundesbank BBSSY.D.REN.EUR.A630.000000WT1010.A; FTSE Russell and Computershare UK Dividend Monitor; STOXX SXXP/SXXPD; I/B/E/S or Bloomberg BEst vintage growth and Bloomberg/FactSet/LSEG repurchases; Damodaran histimpl.xls. "
             f"Retrieved {date.today().isoformat()}. GBP, EUR and USD costs of equity are local-currency rates and are not directly comparable without inflation adjustment.",
             fontsize=7, color="#8c8c8c", style="italic", wrap=True)
    fig.subplots_adjust(left=.09, right=.97, top=.86, bottom=.28 if not latest_qoq.empty else .17)
    fig.savefig(OUTPUT_CHART, facecolor="white")
    plt.close(fig)

    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Wrote {OUTPUT_QOQ}")
    print(f"Wrote {OUTPUT_SENSITIVITY}")
    print(f"Wrote {OUTPUT_SOURCES}")
    print(f"Wrote {OUTPUT_CHART}")
    print(f"US reference source: {US_PAGE} (workbook URL: {US_DATA_URL}; retrieved {date.today().isoformat()})")


if __name__ == "__main__":
    main()
