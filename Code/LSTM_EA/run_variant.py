#!/usr/bin/env python
"""
Driver for the concept-shift work plan: run the spatial LOSO experiment as a named
variant, into its own output directory, and record per-site x year metrics.

Deliberately thin.  It calls run_cv.run_single_fold unmodified and seeds exactly the
way run_cv.main does (run_cv.py:415-416), so `--variant baseline` reproduces the
current run_cv.py behaviour and later variants differ only by the one thing changed.

Usage:
    python run_variant.py --variant baseline
    python run_variant.py --variant no_irrigation_static --compare-to baseline

Writes to outputs/model_variants/<variant>/ only.  The published outputs/ tree is
never touched.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import plot_style as ps
import variant_metrics as vm
from run_cv import CVConfig, CV_FOLDS, run_single_fold
from utils.logging_utils import setup_logging

BASE = Path(__file__).resolve().parent

# Each variant is exactly one delta from CVConfig()'s published defaults.
# 'baseline' is empty by construction, so it reproduces run_cv.py.
VARIANTS = {
    "baseline": {},
    "no_irrigation_static": {"exclude_irrigation_static": True},

    # The Mar-04 configuration that produced Table 3 (Ne1 0.5192 / Ne2 0.6729 /
    # Ne3 0.6790).  Not recoverable from git - the Mar-04 tree had uncommitted
    # edits - so every value below is read off
    # outputs/sensitivity/seq20/experiment_20260304_135503.log, except
    # weight_decay, which was never logged (see WEIGHT_DECAY note below).
    "paper_config": {
        "hidden_dim": 32,               # log: "Created StandardLSTM: ... hidden_dim=32"
        "num_layers": 1,                # log: "... num_layers=1"
        "dropout": 0.5,                 # log: "... dropout=0.5"
        "batch_size": 32,               # log: "Batch size: 32"
        "learning_rate": 0.0005,        # log: "Learning rate: 0.0005"
        "early_stopping_patience": 20,  # log: "Early stopping patience: 20"
        "epochs": 150,                  # log: "Epochs: 150"
        "weight_decay": 0.001,          # NOT logged; committed default at 960b330 (Mar-02)
    },

    # paper_config + the two data-path correctness fixes.  This is the anchor for
    # all architecture work; paper_config itself stays as the as-published reference.
    "paper_config_fixed": {
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.5,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "early_stopping_patience": 20,
        "epochs": 150,
        "weight_decay": 0.001,
        "require_contiguous_windows": True,    # A4
        "impute_statics_after_scaling": True,  # A3
        # NOTE: we deliberately did NOT split A4-only vs A3-only attribution runs.
        # Combined delta vs paper_config was within seed noise at every site
        # (Ne1 -0.026, Ne2 -0.014, Ne3 -0.003; all |t| < 1.1), so the attribution
        # would not have been resolvable at 5 seeds.  Decision recorded 2026-08-23.
    },

    # B4: forget-gate bias initialised to 1.0, everything else = paper_config_fixed.
    "b4_forget_bias": {
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.5,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "early_stopping_patience": 20,
        "epochs": 150,
        "weight_decay": 0.001,
        "require_contiguous_windows": True,
        "impute_statics_after_scaling": True,
        "forget_gate_bias_init": 1.0,
    },

    # D1: SWI + API channels at T = 5 / 20 / 60 days, on top of paper_config_fixed.
    # Adds 6 dynamic channels (swi_5/20/60, api_5/20/60); no other change.
    "d1_swi_api": {
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.5,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "early_stopping_patience": 20,
        "epochs": 150,
        "weight_decay": 0.001,
        "require_contiguous_windows": True,
        "impute_statics_after_scaling": True,
        "swi_api_taus": (5, 20, 60),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS),
                        help="Variant name; also the output subdirectory.")
    parser.add_argument("--compare-to", default="baseline",
                        help="Variant to diff against in the printed table (default: baseline).")
    parser.add_argument("--pipeline", default="run_cv", choices=["run_cv", "sensitivity"],
                        help="Which published pipeline to reproduce. 'sensitivity' builds the "
                             "config via run_sensitivity.build_experiments()['baseline_no_doy'] "
                             "and runs folds through its _run_fold_with_overrides - the path the "
                             "paper's Table 3 actually used.")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds; each is a full 3-fold CV run.")
    parser.add_argument("--season", default="may_oct", choices=["all", "may_oct"],
                        help="Season used for the printed comparison table.")
    args = parser.parse_args()

    run_tag = args.variant if args.pipeline == "run_cv" else f"{args.variant}_{args.pipeline}"
    out_dir = vm.VARIANTS_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = VARIANTS[args.variant]

    if args.pipeline == "sensitivity":
        # Reproduce run_sensitivity.main's construction exactly (run_sensitivity.py:588-600).
        import run_sensitivity as rs
        exp = rs.build_experiments()["baseline_no_doy"]
        config = CVConfig()
        config.seq_length = 20
        config.dynamic_cols = exp["dynamic_cols"]
        config.static_cols = exp["static_cols"]
        config.use_presto_static = exp["use_presto_static"]
        config.add_temporal = False           # baseline_no_doy
        # The sensitivity runner reads the flag off this private attribute, not the field.
        config._exclude_irrigation_static = bool(
            overrides.get("exclude_irrigation_static", exp["exclude_irrigation_static"]))
        # Apply the variant's hyperparameter overrides on top of the sensitivity config.
        for _k, _v in overrides.items():
            if _k != "exclude_irrigation_static":
                setattr(config, _k, _v)
        fold_runner = rs._run_fold_with_overrides
    else:
        config = CVConfig(**overrides)
        fold_runner = run_single_fold

    config.output_dir = str(out_dir)

    logger = setup_logging(str(out_dir), log_level="INFO")
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("VARIANT: %s", args.variant)
    log.info("=" * 70)
    log.info("Output dir:      %s", out_dir)
    log.info("Overrides:       %s", overrides or "(none - reproduces run_cv.py)")
    log.info("Model:           %s (hidden=%d, layers=%d, dropout=%.2f)",
             config.model_type, config.hidden_dim, config.num_layers, config.dropout)
    log.info("Train:           bs=%d lr=%g wd=%g patience=%d",
             config.batch_size, config.learning_rate, config.weight_decay,
             config.early_stopping_patience)
    log.info("seq_length:      %d", config.seq_length)
    log.info("year_range:      %s", (config.year_range,))
    log.info("val_years:       %s", (config.val_years,))
    log.info("target:          %s", config.target_col)
    log.info("dynamic cols:    %d", len(config.dynamic_cols))
    log.info("static cols:     %d (+1 irrigation unless excluded)", len(config.static_cols))
    log.info("pipeline:        %s", args.pipeline)
    log.info("use_presto:      %s", config.use_presto_static)
    log.info("excl irrigation: %s", getattr(config, "_exclude_irrigation_static",
                                            config.exclude_irrigation_static))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    log.info("seeds:           %s", seeds)

    all_rows = []
    results = []
    for seed in seeds:
        # Seed once before the fold loop, as both run_cv.main (run_cv.py:415-416) and
        # run_sensitivity.main (run_sensitivity.py:604-605) do.
        config.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        label = args.variant if len(seeds) == 1 else f"{args.variant}_seed{seed}"
        log.info("=" * 70)
        log.info("SEED %d  ->  variant label %r", seed, label)
        log.info("=" * 70)

        seed_results = []
        for fold in CV_FOLDS:
            r = fold_runner(fold, config, log)
            seed_results.append(r)
            all_rows.extend(vm.rows_for_fold(
                variant=label,
                experiment=f"spatial_loso_{args.pipeline}",
                test_station=r["test_station"],
                dates=r["dates_test"],
                y_true=r["y_true"],
                y_pred=r["y_pred"],
            ))
        if seed == seeds[0]:
            results = seed_results

    # --- metrics -----------------------------------------------------------
    rows = all_rows
    station_data = {}
    for r in results:
        station = r["test_station"]
        station_data[station] = {
            "y_true": r["y_true"],
            "y_pred": r["y_pred"],
            "r2": r["r2_may_oct"] if r["r2_may_oct"] == r["r2_may_oct"] else r["metrics"]["r2"],
        }

    summary_path = vm.append_rows(rows)
    log.info("Summary metrics -> %s", summary_path)

    fig = ps.scatter_panels(station_data)
    fig_path = ps.save_figure(fig, out_dir / f"{args.variant}_scatter.png")
    log.info("Figure -> %s", fig_path)

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"VARIANT: {args.variant}   (season={args.season})")
    print("=" * 70)

    table = vm.comparison_table(args.variant, baseline=args.compare_to, season=args.season)
    print(vm.render(table))

    print("\nPooled per-site (year=ALL):")
    if table is not None:
        print(vm.render(table[table["year"].astype(str) == "ALL"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
