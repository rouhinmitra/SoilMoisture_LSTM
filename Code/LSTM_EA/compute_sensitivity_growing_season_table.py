#!/usr/bin/env python
"""
Compute growing-season (May–October) R² per site for sensitivity experiments.

Reads the per-site prediction CSVs produced by `run_sensitivity.py` and
aggregates performance over the growing season only, returning a table with:

- rows: experiments (models)
- columns: one column per site + a `mean_R2` column

This is intended to support manuscript/figure tables comparing the impact of
removing different feature groups on out-of-sample R² during the growing season.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Ordered list of experiments to include in the table.
# These names must match the keys in `build_experiments()` in run_sensitivity.py.
EXPERIMENTS: List[str] = [
    "baseline",
    "baseline_no_doy",
    "no_ssm",
    "no_precip_irrig",
    "no_meteo",
    "no_sentinel2",
    "no_alpha_earth",
    "no_presto",
    "no_ssm_s2_presto",
    "no_presto_ae",
    "no_presto_ae_s2",
    "no_ssm_no_presto",
    "no_ssm_no_presto_no_s2",
    "no_ssm_no_s2",
]

# Year range to match spatial transferability scripts (2017–2023).
YEAR_RANGE = (2017, 2023)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R² (and a few ancillary metrics) with NaN handling.

    This mirrors the implementation in `plot_sensitivity_timeseries_by_year.py`
    so R² values are consistent across scripts.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {
            "r2": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "pbias": np.nan,
            "nrmse": np.nan,
            "corr": np.nan,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = np.mean(y_pred - y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan
    obs_range = y_true.max() - y_true.min()
    nrmse = rmse / obs_range if obs_range > 0 else np.nan
    corr = np.corrcoef(y_true, y_pred)[0, 1] if n >= 2 else np.nan

    return {
        "r2": r2,
        "bias": bias,
        "rmse": rmse,
        "pbias": pbias,
        "nrmse": nrmse,
        "corr": corr,
    }


def infer_sites(seq_root: Path) -> List[str]:
    """Infer site names from baseline prediction files (predictions_<site>.csv)."""
    baseline_dir = seq_root / "baseline"
    if not baseline_dir.exists():
        return []
    sites: List[str] = []
    for csv_path in baseline_dir.glob("predictions_*.csv"):
        name = csv_path.stem  # predictions_<site>
        if "_" not in name:
            continue
        site = name.split("_", 1)[1]
        sites.append(site)
    return sorted(set(sites))


def compute_growing_season_r2_for_experiment(
    seq_root: Path,
    experiment: str,
    sites: List[str],
) -> Dict[str, Optional[float]]:
    """Compute May–October (2017–2023) R² per site for a single experiment."""
    exp_dir = seq_root / experiment
    r2_by_site: Dict[str, Optional[float]] = {}

    for site in sites:
        csv_path = exp_dir / f"predictions_{site}.csv"
        if not csv_path.exists():
            r2_by_site[site] = np.nan
            continue

        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
        except Exception:
            r2_by_site[site] = np.nan
            continue

        if "Date" not in df.columns or "RZSM_obs" not in df.columns or "RZSM_pred" not in df.columns:
            r2_by_site[site] = np.nan
            continue

        # Restrict to growing-season months (May–October) and years 2017–2023
        df["Date"] = pd.to_datetime(df["Date"])
        df["month"] = df["Date"].dt.month
        df["year"] = df["Date"].dt.year
        growing = df[
            df["month"].isin([5, 6, 7, 8, 9, 10])
            & (df["year"] >= YEAR_RANGE[0])
            & (df["year"] <= YEAR_RANGE[1])
        ]
        if len(growing) < 2:
            r2_by_site[site] = np.nan
            continue

        y_true = growing["RZSM_obs"].values
        y_pred = growing["RZSM_pred"].values
        metrics = compute_metrics(y_true, y_pred)
        r2_by_site[site] = metrics["r2"]

    return r2_by_site


def build_growing_season_table(
    sensitivity_root: Path,
    seq_length: int,
    sites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assemble the May–October R² table for all experiments."""
    seq_root = sensitivity_root / f"seq{seq_length}"
    if sites is None or len(sites) == 0:
        sites = infer_sites(seq_root)

    if not sites:
        raise RuntimeError(f"No sites found under {seq_root}/baseline (no predictions_*.csv files).")

    rows = []
    for exp in EXPERIMENTS:
        r2_by_site = compute_growing_season_r2_for_experiment(seq_root, exp, sites)
        row = {"experiment": exp}
        row.update(r2_by_site)
        # Compute mean across available sites (ignoring NaNs)
        site_values = np.array([r2_by_site[s] for s in sites], dtype=float)
        row["mean_R2"] = float(np.nanmean(site_values)) if np.isfinite(site_values).any() else np.nan
        rows.append(row)

    columns = ["experiment"] + sites + ["mean_R2"]
    df = pd.DataFrame(rows, columns=columns)
    return df


def format_table_for_print(df: pd.DataFrame, sites: List[str]) -> str:
    """Return a nicely formatted text table for the console."""
    # Header
    headers = ["experiment"] + sites + ["mean_R2"]
    col_widths = {h: max(len(h), 10) for h in headers}

    for _, row in df.iterrows():
        col_widths["experiment"] = max(col_widths["experiment"], len(str(row["experiment"])))
        for s in sites + ["mean_R2"]:
            val = row.get(s)
            txt = "nan" if pd.isna(val) else f"{val:.4f}"
            col_widths[s] = max(col_widths[s], len(txt))

    def fmt_row(values: Dict[str, str]) -> str:
        return "  ".join(f"{values[h]:<{col_widths[h]}}" for h in headers)

    lines = []
    # Header line
    header_vals = {h: h for h in headers}
    lines.append(fmt_row(header_vals))
    # Separator
    lines.append("-" * (sum(col_widths.values()) + 2 * (len(headers) - 1)))

    # Data rows
    for _, row in df.iterrows():
        vals: Dict[str, str] = {"experiment": str(row["experiment"])}
        for s in sites + ["mean_R2"]:
            v = row[s]
            vals[s] = "nan" if pd.isna(v) else f"{v:.4f}"
        lines.append(fmt_row(vals))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute growing-season (May–October) R² per site for sensitivity experiments.\n"
            "Reads prediction CSVs from run_sensitivity.py outputs and writes a table."
        )
    )
    parser.add_argument(
        "--sensitivity-root",
        type=str,
        default="outputs/sensitivity",
        help="Root directory containing seq<N>/<experiment>/predictions_<site>.csv (default: outputs/sensitivity).",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=15,
        help="Sequence length used for the sensitivity runs (seq<N> subdirectory name).",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        metavar="SITE",
        help="Optional list of sites to include (default: infer from baseline predictions).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity_root = Path(args.sensitivity_root)

    df = build_growing_season_table(
        sensitivity_root=sensitivity_root,
        seq_length=args.seq_length,
        sites=args.sites,
    )

    seq_root = sensitivity_root / f"seq{args.seq_length}"
    out_path = seq_root / "sensitivity_growing_season_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("GROWING-SEASON (MAY–OCTOBER) R² PER SITE – SENSITIVITY EXPERIMENTS")
    print("=" * 80)
    sites = [c for c in df.columns if c not in ("experiment", "mean_R2")]
    print(format_table_for_print(df, sites))
    print("\nSaved table to:", out_path)


if __name__ == "__main__":
    main()

