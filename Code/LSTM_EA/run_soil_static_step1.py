"""
STEP 1 - soil hydraulic properties as static features.  15 seeds vs paper_config_fixed.

Adds ONLY field_capacity_60cm and wilting_point_60cm to static_cols (193 -> 195,
plus the auto-appended irrigation = 196).  All 14 soil columns in the station
CSVs are CONSTANT PER SITE, so at n=3 sites the full set is an exact site
one-hot; two columns is the smallest version of the test.

PRE-REGISTERED PREDICTION (recorded before running): this should FAIL.  Under
LOSO the model sees two distinct FC/WP values in training and must extrapolate
to an unseen third.  Two points do not define a relationship.  Same
over-conditioning mechanism as D2 and the Presto ablation.

A DATA REPAIR IS REQUIRED FIRST, AND IT IS REPORTED AS A RESULT IN ITS OWN RIGHT.
field_capacity_60cm / wilting_point_60cm are populated for every year at Ne1 but
ONLY IN ODD YEARS at Ne2 and Ne3 (2018/2020/2022/2024 are entirely blank) - a
join artifact in whatever built the station CSVs, since a static soil property
cannot vary by year.  Each site has exactly ONE distinct non-null value, so the
repair is a constant fill, not an imputation of an unknown:
    Ne1 FC=0.3473 WP=0.2013 | Ne2 FC=0.3186 WP=0.1641 | Ne3 FC=0.3232 WP=0.1675
Left unrepaired, 44% of Ne2/Ne3 windows have the feature blanked to the training
mean and a null would be uninterpretable - indistinguishable from the feature
simply being half-absent.

NOT LEAKAGE: FC/WP is a static soil property of the field, known before any
prediction is made, exactly like the AlphaEarth A00-A63 statics that are already
used at the held-out site.  No target information enters.

Both arms are run so the repair itself is auditable:
    filled  (PRIMARY)   - the constant fill above; tests the stated hypothesis
    asis    (SECONDARY) - the CSVs untouched; shows what the raw columns give

Usage:  python run_soil_static_step1.py --arm filled --seeds 1,2,...
Writes outputs/model_variants/soil_static_fc_wp_<arm>/ and appends
summary_metrics.csv with experiment='soil_static_step1'.
"""
import argparse, logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
import variant_metrics as vm
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS
from src import data_loader as dl

BASE = Path(__file__).resolve().parent
SOIL = ["field_capacity_60cm", "wilting_point_60cm"]


def install_fill():
    """Wrap DataProcessor.load_and_clean to fill the site-constant soil columns."""
    orig = dl.DataProcessor.load_and_clean

    def patched(self, filepath):
        df = orig(self, filepath)
        if df is None:
            return df
        for c in SOIL:
            if c in df.columns:
                u = df[c].dropna().unique()
                if len(u) == 1:
                    n_before = int(df[c].isna().sum())
                    df[c] = u[0]
                    if n_before:
                        logging.getLogger(__name__).info(
                            f"  soil fill: {c} filled {n_before} NaN rows with {u[0]:.4f} "
                            f"({Path(filepath).name})")
                elif len(u) > 1:
                    raise RuntimeError(f"{c} is not site-constant in {filepath}: {u}")
        return df

    dl.DataProcessor.load_and_clean = patched
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="filled", choices=["filled", "asis"])
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
    a = ap.parse_args()

    if a.arm == "filled":
        install_fill()
        print("soil columns: CONSTANT-FILLED per site\n")
    else:
        print("soil columns: AS-IS (odd years only at Ne2/Ne3)\n")

    ov = VARIANTS["soil_static_fc_wp"]
    exp = rs.build_experiments()["baseline_no_doy"]
    cfg = CVConfig()
    cfg.seq_length = 20
    cfg.dynamic_cols = exp["dynamic_cols"]
    cfg.static_cols = exp["static_cols"]
    cfg.use_presto_static = exp["use_presto_static"]
    cfg.add_temporal = False
    cfg._exclude_irrigation_static = bool(exp["exclude_irrigation_static"])
    for k, v in ov.items():
        setattr(cfg, k, v)

    out_dir = BASE / "outputs" / "model_variants" / f"soil_static_fc_wp_{a.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)
    logging.disable(logging.INFO)

    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"variant soil_static_fc_wp[{a.arm}]  static_cols={len(cfg.static_cols)} (+1 irrigation)")
    print(f"seeds: {seeds}\n")

    rows = []
    for seed in seeds:
        cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg.output_dir = str(out_dir / f"seed{seed}")
        for fold in CV_FOLDS:
            r = rs._run_fold_with_overrides(fold, cfg, log)
            rows.extend(vm.rows_for_fold(
                variant=f"soil_static_fc_wp_{a.arm}",
                experiment="soil_static_step1",
                test_station=r["test_station"], dates=r["dates_test"],
                y_true=r["y_true"], y_pred=r["y_pred"]))
            print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}",
                  flush=True)
    vm.append_rows(rows)
    print(f"\nrows -> summary_metrics.csv  variant='soil_static_fc_wp_{a.arm}', "
          f"experiment='soil_static_step1'")


if __name__ == "__main__":
    main()
