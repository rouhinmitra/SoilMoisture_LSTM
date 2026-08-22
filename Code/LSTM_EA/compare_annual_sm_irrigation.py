"""
Compare annual soil moisture values with annual irrigation amounts at each site.

- RZSM averages are computed only for May–October each year.
- Compares irrigation with: RZSM 25cm, RZSM 50cm, RZSM 100cm, and (RZSM 50cm − 25cm).

Requires: pandas, openpyxl (for Excel). Plots require matplotlib.
Default data dir: /Users/rouhinmitra/SM_work/Data/Base/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# Default paths (script dir = LSTM_EA)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/Users/rouhinmitra/SM_work/Data/Base/")
DEFAULT_IRRIGATION_FILE = SCRIPT_DIR / "Irrigation-data.xlsx"

# May–October only (inclusive)
MONTH_START, MONTH_END = 5, 10

# RZSM columns: 25cm, 50cm, 100cm; (50−25) is computed
RZSM_25 = "RZSM_25_avg"
RZSM_50 = "RZSM_50_avg"
RZSM_100 = "RZSM_100_avg"
RZSM_COLS = [RZSM_25, RZSM_50, RZSM_100]
RZSM_50_25_DIFF = "rzsm_50_25_diff"  # computed column name

# Daily precipitation column (first available used for accumulated precip)
PRECIP_COLS = ["P_PI_F_1_1_1", "P_PI_F_2_2_1"]


def load_irrigation_data(irrigation_file_path=None):
    """Load irrigation data from Excel; return site -> {year: value}."""
    path = Path(irrigation_file_path or DEFAULT_IRRIGATION_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Irrigation file not found: {path}")
    irrigation_df = pd.read_excel(path)
    irrigation_map = {}
    if "Year" in irrigation_df.columns:
        if "US-Ne1" in irrigation_df.columns:
            irrigation_map["ne1"] = dict(zip(irrigation_df["Year"], irrigation_df["US-Ne1"]))
        if "US-Ne2" in irrigation_df.columns:
            irrigation_map["ne2"] = dict(zip(irrigation_df["Year"], irrigation_df["US-Ne2"]))
    return irrigation_map


def get_site_from_df(df, filepath):
    """Identify site (ne1, ne2, ne3) from filename or Name column."""
    name = Path(filepath).name.lower()
    if "ne1" in name:
        return "ne1"
    if "ne2" in name:
        return "ne2"
    if "ne3" in name:
        return "ne3"
    if "Name" in df.columns and len(df) > 0:
        val = str(df["Name"].iloc[0]).lower()
        if "ne1" in val:
            return "ne1"
        if "ne2" in val:
            return "ne2"
        if "ne3" in val:
            return "ne3"
    return None


def load_site_csvs(data_dir):
    """Load all site CSVs and return list of (site_name, dataframe)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    out = []
    for f in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        # Need at least one RZSM column
        has_rzsm = any(c in df.columns for c in RZSM_COLS)
        if not has_rzsm:
            continue
        site = get_site_from_df(df, str(f))
        if site is None:
            continue
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        out.append((site, df))
    return out


def annual_precipitation(site_dfs):
    """
    Accumulated precipitation per (site, year):
    - precip_summer: sum of daily precip May 1–Oct 31
    - precip_jan_apr: sum of daily precip Jan 1–Apr 30
    """
    precip_col = None
    for c in PRECIP_COLS:
        for _, df in site_dfs:
            if c in df.columns:
                precip_col = c
                break
        if precip_col is not None:
            break
    if precip_col is None:
        return pd.DataFrame()

    rows = []
    for site, df in site_dfs:
        if precip_col not in df.columns:
            continue
        df = df.copy()
        # Summer: May (5) through October (10)
        summer_mask = (df["Month"] >= MONTH_START) & (df["Month"] <= MONTH_END)
        # Jan 1 – Apr 30: month in (1,2,3) or (month==4 and day<=30)
        jan_apr_mask = (df["Month"] < 4) | ((df["Month"] == 4) & (df["Day"] <= 30))
        for year in df["Year"].unique():
            sy = df[(df["Year"] == year) & summer_mask]
            jy = df[(df["Year"] == year) & jan_apr_mask]
            rows.append({
                "site": site,
                "Year": year,
                "precip_summer": sy[precip_col].sum(),
                "precip_jan_apr": jy[precip_col].sum(),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def annual_soil_moisture(site_dfs):
    """
    Aggregate soil moisture by site and year: mean (and std, count) for May–October only.
    Metrics: RZSM 25cm, 50cm, 100cm, and (50cm − 25cm). One row per (site, year).
    """
    METRICS = [
        (RZSM_25, "rzsm_25"),
        (RZSM_50, "rzsm_50"),
        (RZSM_100, "rzsm_100"),
        (RZSM_50_25_DIFF, "rzsm_50_25_diff"),
    ]
    rows = []
    for site, df in site_dfs:
        mask = (df["Month"] >= MONTH_START) & (df["Month"] <= MONTH_END)
        df_sub = df.loc[mask].copy()
        if df_sub.empty:
            continue
        if RZSM_25 in df_sub.columns and RZSM_50 in df_sub.columns:
            df_sub[RZSM_50_25_DIFF] = df_sub[RZSM_50] - df_sub[RZSM_25]
        for year in df_sub["Year"].unique():
            row = {"site": site, "Year": year}
            for col, prefix in METRICS:
                if col not in df_sub.columns:
                    continue
                g = df_sub[df_sub["Year"] == year][col]
                row[f"{prefix}_mean"] = g.mean()
                row[f"{prefix}_std"] = g.std()
                row[f"{prefix}_n"] = g.count()
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def annual_irrigation_table(irrigation_map):
    """Convert irrigation map to long table: site, year, irrigation."""
    rows = []
    for site, year_val in irrigation_map.items():
        for year, val in year_val.items():
            rows.append({"site": site, "Year": year, "irrigation": val})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def merge_annual_sm_irrigation(sm_annual, irr_annual):
    """Merge annual soil moisture and irrigation by site and year."""
    return pd.merge(
        sm_annual,
        irr_annual,
        on=["site", "Year"],
        how="outer",
    )


def run_comparison(data_dir=None, irrigation_file=None, out_dir=None, save_plots=True):
    """Load data, merge annual SM and irrigation, plot and print stats."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    irrigation_file = Path(irrigation_file or DEFAULT_IRRIGATION_FILE)
    out_dir = Path(out_dir or SCRIPT_DIR / "outputs" / "sm_irrigation_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load irrigation (ne1, ne2 only in Excel; ne3 = 0)
    irrigation_map = load_irrigation_data(irrigation_file)
    irr_annual = annual_irrigation_table(irrigation_map)

    # Load site CSVs and aggregate annual soil moisture
    site_dfs = load_site_csvs(data_dir)
    if not site_dfs:
        print("No site CSVs found with RZSM column.")
        return None

    sm_annual = annual_soil_moisture(site_dfs)
    merged = merge_annual_sm_irrigation(sm_annual, irr_annual)
    precip_annual = annual_precipitation(site_dfs)
    if not precip_annual.empty:
        merged = pd.merge(merged, precip_annual, on=["site", "Year"], how="left")
    merged = merged.sort_values(["site", "Year"]).reset_index(drop=True)

    mean_cols = [c for c in merged.columns if c.endswith("_mean")]
    if mean_cols:
        merged = merged.loc[merged[mean_cols].notna().any(axis=1)]  # keep row if any metric present
    merged["irrigation"] = merged["irrigation"].fillna(0.0)

    # Summary table (May–October RZSM only)
    print("Annual soil moisture (May–Oct) vs irrigation by site")
    print("=" * 60)
    print(merged.to_string())
    csv_path = out_dir / "annual_sm_irrigation_by_site.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Correlations: each metric vs irrigation, by site
    for metric in ["rzsm_25_mean", "rzsm_50_mean", "rzsm_100_mean", "rzsm_50_25_diff_mean"]:
        if metric not in merged.columns:
            continue
        print(f"\nCorrelation ({metric} vs irrigation) by site:")
        for site in merged["site"].unique():
            sub = merged[merged["site"] == site]
            sub = sub.dropna(subset=[metric, "irrigation"])
            if sub["irrigation"].nunique() < 2:
                print(f"  {site}: N/A (irrigation constant)")
                continue
            r = sub[metric].corr(sub["irrigation"])
            print(f"  {site}: r = {r:.4f}  (n = {len(sub)})")

    if save_plots:
        _plot_comparison(merged, out_dir)

    return merged


# Metrics to plot: label, column name
PLOT_METRICS = [
    ("RZSM 25cm (May–Oct mean)", "rzsm_25_mean"),
    ("RZSM 50cm (May–Oct mean)", "rzsm_50_mean"),
    ("RZSM 100cm (May–Oct mean)", "rzsm_100_mean"),
    ("RZSM 50−25cm (May–Oct mean)", "rzsm_50_25_diff_mean"),
]


def _plot_comparison(merged, out_dir):
    """Generate scatter and time-series figures for each RZSM metric vs irrigation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return
    sites = sorted(merged["site"].unique())
    n_sites = len(sites)

    for label, col in PLOT_METRICS:
        if col not in merged.columns:
            continue
        # Scatter: irrigation vs this metric, one subplot per site
        fig, axes = plt.subplots(1, n_sites, figsize=(5 * n_sites, 5))
        if n_sites == 1:
            axes = [axes]
        for ax, site in zip(axes, sites):
            sub = merged[merged["site"] == site].dropna(subset=[col, "irrigation"])
            if sub.empty:
                ax.set_title(f"Site {site} (no data)")
                continue
            ax.scatter(sub["irrigation"], sub[col], alpha=0.8, edgecolors="k", s=60)
            ax.set_xlabel("Annual irrigation")
            ax.set_ylabel(label)
            ax.set_title(f"Site {site}")
            if sub["irrigation"].nunique() >= 2:
                r = sub[col].corr(sub["irrigation"])
                ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=11, va="top")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Annual {label} vs irrigation by site", fontsize=12)
        fig.tight_layout()
        safe_name = col.replace("_mean", "")
        fig.savefig(out_dir / f"scatter_{safe_name}_vs_irrigation_by_site.png", dpi=150)
        plt.close(fig)

    # Time series: one figure per site, all metrics + irrigation + precip bar charts + combined bar
    n_precip = 2 if ("precip_summer" in merged.columns and "precip_jan_apr" in merged.columns) else 0
    has_combined = n_precip == 2  # one extra panel with all three in one chart
    for site in sites:
        sub = merged[merged["site"] == site].sort_values("Year")
        if sub.empty:
            continue
        n_metrics = sum(1 for _, c in PLOT_METRICS if c in sub.columns)
        if n_metrics == 0:
            continue
        n_rows = n_metrics + 1 + n_precip + (1 if has_combined else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        idx = 0
        for label, col in PLOT_METRICS:
            if col not in sub.columns:
                continue
            ax = axes[idx]
            ax.plot(sub["Year"], sub[col], "o-", color="C0", label=label)
            std_col = col.replace("_mean", "_std")
            if std_col in sub.columns and sub[std_col].notna().any():
                ax.fill_between(sub["Year"], sub[col] - sub[std_col], sub[col] + sub[std_col], alpha=0.2, color="C0")
            ax.set_ylabel(label)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            idx += 1
        ax_irr = axes[idx]
        ax_irr.bar(sub["Year"], sub["irrigation"], color="C1", alpha=0.8, label="Irrigation")
        ax_irr.set_ylabel("Annual irrigation")
        ax_irr.legend(loc="best")
        ax_irr.grid(True, alpha=0.3)
        idx += 1
        if "precip_summer" in sub.columns:
            ax_summer = axes[idx]
            ax_summer.bar(sub["Year"], sub["precip_summer"], color="C2", alpha=0.8, label="Accum. summer precip (May–Oct)")
            ax_summer.set_ylabel("Precip (May–Oct)")
            ax_summer.legend(loc="best")
            ax_summer.grid(True, alpha=0.3)
            idx += 1
        if "precip_jan_apr" in sub.columns:
            ax_janapr = axes[idx]
            ax_janapr.bar(sub["Year"], sub["precip_jan_apr"], color="C3", alpha=0.8, label="Accum. precip (Jan 1–Apr 30)")
            ax_janapr.set_ylabel("Precip (Jan–Apr 30)")
            ax_janapr.legend(loc="best")
            ax_janapr.grid(True, alpha=0.3)
            idx += 1
        # One bar per year = sum of irrigation + precip_summer + precip_jan_apr
        if has_combined:
            ax_comb = axes[idx]
            total = sub["irrigation"].fillna(0) + sub["precip_summer"].fillna(0) + sub["precip_jan_apr"].fillna(0)
            ax_comb.bar(sub["Year"], total, color="C4", alpha=0.8, label="Irrigation + Precip May–Oct + Precip Jan–Apr 30")
            ax_comb.set_ylabel("Total (sum)")
            ax_comb.legend(loc="best")
            ax_comb.grid(True, alpha=0.3)
            idx += 1
        axes[-1].set_xlabel("Year")
        fig.suptitle(f"Site {site}: May–Oct RZSM, irrigation, and precipitation", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_site_{site}.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare annual soil moisture with annual irrigation per site.")
    parser.add_argument("--data-dir", type=str, default=None, help=f"Directory of site CSVs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--irrigation-file", type=str, default=None, help="Path to Irrigation-data.xlsx")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for CSV and plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    run_comparison(
        data_dir=args.data_dir,
        irrigation_file=args.irrigation_file,
        out_dir=args.out_dir,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
