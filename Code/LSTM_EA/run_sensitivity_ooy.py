#!/usr/bin/env python
"""
Feature-group sensitivity (ablation) for the out-of-year (temporal transfer) LSTM
at single stations — same experiment definitions as run_sensitivity.py (subset),
trained/evaluated with the loop in run_lstm_out_of_year_single_station.py.

Baseline matches current OOY defaults: no Sentinel-2, no Presto; static precip_jan_apr
only. If the baseline has no S2 columns, the ``no_sentinel2`` experiment is identical
to baseline and is skipped unless ``--include-degenerate``.

Experiments:
  baseline, no_ssm, no_sentinel2, no_meteo, no_precip_irrig

Outputs under ``outputs/sensitivity_ooy/seq{N}/`` (or ``--output-root``):
  - sensitivity_ooy_results.csv
  - sensitivity_ooy_summary.csv (pooled mean/std by experiment)
  - sensitivity_ooy_summary_by_window.csv
  - sensitivity_bar_chart.png, sensitivity_delta_r2.png (unless --no-plots)
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from run_lstm_out_of_year_single_station import (
    OOYCVConfig,
    STATIONS,
    generate_end_anchored_year_windows,
    run_ooy_for_station,
)
from run_sensitivity import (
    METEO_FEATURES,
    PRECIP_FEATURES_DYN,
    S2_FEATURES,
    SSM_FEATURES,
    compute_extra_metrics,
    create_delta_chart,
    create_sensitivity_bar_chart,
)
from utils.logging_utils import setup_logging

# Order preserved in summaries and plots
OOY_SENSITIVITY_EXPERIMENT_NAMES: List[str] = [
    "baseline",
    "no_ssm",
    "no_sentinel2",
    "no_meteo",
    "no_precip_irrig",
]


def baseline_dynamic_cols() -> List[str]:
    """Same uncommented dynamics as OOYCVConfig.__post_init__ (no Sentinel-2)."""
    return [
        "SSM",
        "SSM_avg",
        "SWC_PI_F_2_1_1",
        "SWC_PI_F_3_1_1",
        "P_PI_F_1_1_1",
        "P_PI_F_2_2_1",
        "I",
        "TA_1_1_1",
        "RH_1_1_1",
        "LE_1_1_1",
        "NETRAD_1_1_1",
    ]


def build_ooy_sensitivity_experiments() -> Dict[str, dict]:
    """Experiment name -> overrides applied on top of OOYCVConfig (via dataclasses.replace)."""
    base = baseline_dynamic_cols()
    experiments: Dict[str, dict] = {}

    experiments["baseline"] = dict(
        description="OOY baseline (all default dynamics + precip_jan_apr static)",
        dynamic_cols=list(base),
        static_cols=["precip_jan_apr"],
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    experiments["no_ssm"] = dict(
        description="Remove SSM-related dynamics",
        dynamic_cols=[c for c in base if c not in SSM_FEATURES],
        static_cols=["precip_jan_apr"],
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    experiments["no_sentinel2"] = dict(
        description="Remove Sentinel-2 dynamics (no-op if baseline has no S2)",
        dynamic_cols=[c for c in base if c not in S2_FEATURES],
        static_cols=["precip_jan_apr"],
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    experiments["no_meteo"] = dict(
        description="Remove meteorological dynamics; omit static precip (mirrors run_sensitivity no_meteo)",
        dynamic_cols=[c for c in base if c not in METEO_FEATURES],
        static_cols=[],
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    experiments["no_precip_irrig"] = dict(
        description="Remove precip/irrigation dynamics and static precip; exclude Excel irrigation static",
        dynamic_cols=[c for c in base if c not in PRECIP_FEATURES_DYN],
        static_cols=[],
        use_presto_static=False,
        exclude_irrigation_static=True,
    )

    return experiments


def _validate_experiments(experiments: Dict[str, dict]) -> None:
    for name, cfg in experiments.items():
        if not cfg.get("dynamic_cols"):
            raise ValueError(f"Experiment {name!r} has empty dynamic_cols.")


def _filter_no_sentinel2_if_degenerate(
    experiments: Dict[str, dict],
    include_degenerate: bool,
    logger: logging.Logger,
) -> Dict[str, dict]:
    base = baseline_dynamic_cols()
    has_s2 = any(c in S2_FEATURES for c in base)
    if has_s2 or include_degenerate or "no_sentinel2" not in experiments:
        return experiments
    logger.info(
        "Skipping experiment 'no_sentinel2': baseline has no Sentinel-2 columns "
        "(same as baseline). Use --include-degenerate to run it anyway."
    )
    return {k: v for k, v in experiments.items() if k != "no_sentinel2"}


def _attach_extra_metrics_from_preds(
    fold_row: pd.Series,
    preds_df: pd.DataFrame,
) -> Tuple[float, float]:
    if preds_df is None or preds_df.empty:
        return float("nan"), float("nan")
    mask = (preds_df["station"] == fold_row["station"]) & (
        preds_df["fold"] == fold_row["fold"]
    )
    sub = preds_df.loc[mask]
    if len(sub) < 2:
        return float("nan"), float("nan")
    ex = compute_extra_metrics(
        sub["RZSM_pred"].to_numpy(dtype=float),
        sub["RZSM_true"].to_numpy(dtype=float),
    )
    return ex["nrmse"], ex["pbias"]


def parse_args() -> argparse.Namespace:
    all_names = list(build_ooy_sensitivity_experiments().keys())
    parser = argparse.ArgumentParser(
        description="OOY temporal-transfer LSTM feature sensitivity (ablations).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        choices=all_names,
        default=None,
        metavar="EXP",
        help="Run only these experiments. Default: all (no_sentinel2 may be auto-skipped).",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        metavar="N",
        help="LSTM sequence length (default: OOYCVConfig default, usually 20).",
    )
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Single inclusive year window only, e.g. --year-range 2014 2023. "
        "Default: same end-anchored windows as run_lstm_out_of_year_single_station.main.",
    )
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Run no_sentinel2 even when baseline has no S2 columns (duplicate of baseline).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/sensitivity_ooy",
        help="Directory for results (seq{N} subfolder is added).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write bar chart or delta R² figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = OOYCVConfig()
    seq_len = args.seq_length if args.seq_length is not None else base_config.seq_length

    output_root = Path(args.output_root) / f"seq{seq_len}"
    output_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_root), log_level="INFO")
    log = logging.getLogger(__name__)

    all_experiments = build_ooy_sensitivity_experiments()
    _validate_experiments(all_experiments)

    if args.experiments:
        experiments = {k: all_experiments[k] for k in args.experiments}
    else:
        experiments = dict(all_experiments)

    experiments = _filter_no_sentinel2_if_degenerate(
        experiments, args.include_degenerate, log
    )
    _validate_experiments(experiments)

    if args.year_range is not None:
        ws, we = args.year_range
        windows: List[Tuple[int, int]] = [(ws, we)]
    else:
        ys, ye = base_config.year_range
        windows = generate_end_anchored_year_windows(ys, ye, min_years=3)

    log.info("=" * 70)
    log.info("OOY FEATURE SENSITIVITY (TEMPORAL TRANSFER LSTM)")
    log.info("=" * 70)
    log.info("seq_length=%s", seq_len)
    log.info("experiments=%s", list(experiments.keys()))
    log.info("windows=%s", windows)

    all_rows: List[dict] = []
    t0 = time.time()

    for exp_name, exp_cfg in experiments.items():
        log.info("\n%s", "#" * 70)
        log.info("# EXPERIMENT: %s — %s", exp_name, exp_cfg["description"])
        log.info(
            "#   dynamic_cols (%d): %s",
            len(exp_cfg["dynamic_cols"]),
            exp_cfg["dynamic_cols"],
        )
        log.info("#   static_cols: %s", exp_cfg["static_cols"])
        log.info(
            "#   use_presto_static=%s exclude_irrigation_static=%s",
            exp_cfg["use_presto_static"],
            exp_cfg["exclude_irrigation_static"],
        )
        log.info("%s\n", "#" * 70)

        torch.manual_seed(base_config.seed)
        np.random.seed(base_config.seed)

        for window_start, window_end in windows:
            window_tag = f"years_{window_start}_{window_end}"
            window_output_dir = output_root / exp_name / window_tag
            window_output_dir.mkdir(parents=True, exist_ok=True)

            wc = replace(
                base_config,
                year_range=(window_start, window_end),
                output_dir=str(window_output_dir),
                dynamic_cols=list(exp_cfg["dynamic_cols"]),
                static_cols=list(exp_cfg["static_cols"]),
                use_presto_static=exp_cfg["use_presto_static"],
                exclude_irrigation_static=exp_cfg["exclude_irrigation_static"],
                seq_length=seq_len,
            )

            for station_name, data_file in STATIONS.items():
                data_path = Path(wc.data_dir) / data_file
                if not data_path.exists():
                    log.warning(
                        "Skipping %s — file not found: %s", station_name, data_path
                    )
                    continue
                try:
                    metrics_df, preds_df = run_ooy_for_station(
                        station_name, data_file, wc, log
                    )
                except Exception:
                    log.exception(
                        "Failed station %s window %s experiment %s",
                        station_name,
                        window_tag,
                        exp_name,
                    )
                    continue

                if metrics_df.empty:
                    continue

                fold_only = metrics_df[metrics_df["fold"] != "OVERALL"]
                for _, row in fold_only.iterrows():
                    nrmse, pbias = _attach_extra_metrics_from_preds(row, preds_df)
                    rec = {
                        "experiment": exp_name,
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_tag": window_tag,
                        "station": row["station"],
                        "fold": row["fold"],
                        "test_year": row["test_year"],
                        "r2": row["r2"],
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "bias": row["bias"],
                        "correlation": row["correlation"],
                        "n_samples": row["n_samples"],
                        "n_train": row["n_train"],
                        "n_val": row["n_val"],
                        "n_test": row["n_test"],
                        "nrmse": nrmse,
                        "pbias": pbias,
                    }
                    all_rows.append(rec)

    elapsed = time.time() - t0
    log.info("All experiment loops finished in %.1f min", elapsed / 60.0)

    if not all_rows:
        log.warning("No result rows collected; CSVs not written.")
        return

    results_df = pd.DataFrame(all_rows)
    results_path = output_root / "sensitivity_ooy_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Wrote %s", results_path)

    # Station-balanced summary:
    # 1) average folds (and windows, if multiple) within each station
    # 2) average those station means per experiment
    station_means = (
        results_df.groupby(["experiment", "station"], sort=False)
        .agg(
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            nrmse_mean=("nrmse", "mean"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
            pbias_mean=("pbias", "mean"),
            n_rows=("r2", "size"),
        )
        .reset_index()
    )
    station_means_path = output_root / "sensitivity_ooy_station_means.csv"
    station_means.to_csv(station_means_path, index=False)
    log.info("Wrote %s", station_means_path)

    summary = (
        station_means.groupby("experiment", sort=False)
        .agg(
            r2_mean=("r2_mean", "mean"),
            r2_std=("r2_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            nrmse_mean=("nrmse_mean", "mean"),
            nrmse_std=("nrmse_mean", "std"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            bias_mean=("bias_mean", "mean"),
            bias_std=("bias_mean", "std"),
            pbias_mean=("pbias_mean", "mean"),
            pbias_std=("pbias_mean", "std"),
            n_stations=("station", "nunique"),
        )
        .reset_index()
    )
    order = [x for x in OOY_SENSITIVITY_EXPERIMENT_NAMES if x in summary["experiment"].values]
    summary["experiment"] = pd.Categorical(
        summary["experiment"], categories=order, ordered=True
    )
    summary = summary.sort_values("experiment").reset_index(drop=True)
    summary_path = output_root / "sensitivity_ooy_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)

    station_window_means = (
        results_df.groupby(
            ["experiment", "window_start", "window_end", "station"], sort=False
        )
        .agg(
            r2_mean=("r2", "mean"),
            rmse_mean=("rmse", "mean"),
            nrmse_mean=("nrmse", "mean"),
            mae_mean=("mae", "mean"),
            pbias_mean=("pbias", "mean"),
        )
        .reset_index()
    )
    by_win = (
        station_window_means.groupby(
            ["experiment", "window_start", "window_end"], sort=False
        )
        .agg(
            r2_mean=("r2_mean", "mean"),
            r2_std=("r2_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            nrmse_mean=("nrmse_mean", "mean"),
            mae_mean=("mae_mean", "mean"),
            pbias_mean=("pbias_mean", "mean"),
            n_stations=("station", "nunique"),
        )
        .reset_index()
    )
    by_win_path = output_root / "sensitivity_ooy_summary_by_window.csv"
    by_win.to_csv(by_win_path, index=False)
    log.info("Wrote %s", by_win_path)

    print("\n" + "=" * 100)
    print("OOY SENSITIVITY SUMMARY (station-averaged over windows/folds)")
    print("=" * 100)
    hdr = (
        f"{'Experiment':<18} {'R²':>8} {'RMSE':>8} {'NRMSE':>8} "
        f"{'MAE':>8} {'Bias':>8} {'PBIAS%':>8}"
    )
    print(hdr)
    print("-" * 100)
    for _, row in summary.iterrows():
        print(
            f"{row['experiment']:<18} "
            f"{row['r2_mean']:>7.4f}  {row['rmse_mean']:>7.4f}  {row['nrmse_mean']:>7.4f}  "
            f"{row['mae_mean']:>7.4f}  {row['bias_mean']:>7.4f}  {row['pbias_mean']:>7.2f}"
        )
    print("=" * 100)

    # Per-station CV R² by experiment (station means over folds/windows).
    station_order = sorted(station_means["station"].dropna().unique().tolist())
    print("\n" + "=" * 100)
    print("OOY SENSITIVITY STATION-WISE CV R² (mean over folds/windows)")
    print("=" * 100)
    if station_order:
        hdr_cols = " ".join([f"{s:>10}" for s in station_order])
        print(f"{'Experiment':<18} {hdr_cols}")
        print("-" * 100)
        for exp in summary["experiment"].tolist():
            row = station_means[station_means["experiment"] == exp]
            vals = []
            for s in station_order:
                srow = row[row["station"] == s]
                if srow.empty:
                    vals.append(f"{'nan':>10}")
                else:
                    vals.append(f"{float(srow['r2_mean'].iloc[0]):>10.4f}")
            print(f"{exp:<18} {' '.join(vals)}")
    else:
        print("No station rows available.")
    print("=" * 100)

    expected_default = set(OOY_SENSITIVITY_EXPERIMENT_NAMES)
    if (
        not any(c in S2_FEATURES for c in baseline_dynamic_cols())
        and not args.include_degenerate
    ):
        expected_default.discard("no_sentinel2")
    full_default_run = args.experiments is None and set(experiments.keys()) == expected_default

    if (
        not args.no_plots
        and full_default_run
        and "baseline" in summary["experiment"].values
        and len(summary) >= 2
    ):
        ooy_title = (
            "Feature Sensitivity — Out-of-Year Temporal Transfer LSTM (single station)"
        )
        create_sensitivity_bar_chart(summary, output_root, suptitle=ooy_title)
        create_delta_chart(
            summary, output_root, context_label="Out-of-year temporal transfer"
        )
    elif not args.no_plots and not full_default_run:
        log.info("Skipping global plots (run all default experiments for bar/delta charts).")

    log.info("OOY sensitivity analysis complete.")


if __name__ == "__main__":
    main()
