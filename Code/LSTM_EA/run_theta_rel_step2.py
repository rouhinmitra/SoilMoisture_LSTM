"""
STEP 2 - normalized target: relative available water.

    theta_rel = (RZSM_25_avg - WP*100) / (FC*100 - WP*100)

UNITS: RZSM_25_avg is percent (~32); FC/WP are fractions (0.347/0.201).  FC and WP
are multiplied by 100 before use.  Verified pre-run: theta_rel means land at
0.80 / 1.08 / 1.07 for Ne1/Ne2/Ne3 - plausible range, no clipping applied.

NOT CLIPPED.  theta_rel > 1 in 28% / 67% / 65% of days at Ne1/Ne2/Ne3, and dips
below 0 at Ne1.  That is real: the pedotransfer FC/WP are not calibrated to these
sensors.  Clipping would hide it.

ODD-YEAR FC/WP REPAIR from step 1 is applied (Ne2/Ne3 have FC/WP only in odd
years; each site has one distinct value, so the fill is a constant).

Unlike step 1 this is a DETERMINISTIC transform, not a learned relationship, so
it never asks the network to extrapolate across soil space.  Inversion for
scoring uses the TEST site's own FC/WP, so reported R2 is in volumetric units and
directly comparable to paper_config_fixed.

PRE-REGISTERED PREDICTION (recorded before the run): FC-WP spans are nearly
identical across sites (14.60 / 15.45 / 15.57), so this is essentially a
site-specific OFFSET of ~3.7 points at Ne1 relative to Ne2/Ne3, not a rescaling.
It should move the site-level offset and barely touch within-site shape.  R2
improving WITHOUT the offset moving would mean something else is going on.
Pre-run descriptive check found corr(WP, mean RZSM) = -0.950 across the 3 sites,
so the transform is expected to ADD ~4.2 volumetric points of between-site
separation where only 1.66 exists.

Variants:
  A  target normalized only
  B  A plus the same normalization applied to the four surface SM inputs
     (SSM, SSM_avg, SWC_PI_F_2_1_1, SWC_PI_F_3_1_1).  NOTE: those are SURFACE
     probes but the only FC/WP available is the 60cm pair, so B applies a
     root-zone scaling to surface inputs.  Recorded as a known approximation.
"""
import argparse, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
import variant_metrics as vm
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS
from src import data_loader as dl

BASE = Path(__file__).resolve().parent
FC = {"ne1": 0.3473, "ne2": 0.3186, "ne3": 0.3232}
WP = {"ne1": 0.2013, "ne2": 0.1641, "ne3": 0.1675}
SSM_COLS = ["SSM", "SSM_avg", "SWC_PI_F_2_1_1", "SWC_PI_F_3_1_1"]
SOIL = ["field_capacity_60cm", "wilting_point_60cm"]


def fcwp(site):
    return FC[site] * 100.0, WP[site] * 100.0


def install(norm_inputs: bool):
    orig = dl.DataProcessor.load_and_clean

    def patched(self, filepath):
        # load_and_clean validates that target_col exists in the CSV, and theta_rel
        # does not yet.  Validate against the raw target, then derive theta_rel here;
        # the windowing that follows reads feature_config.target_col, which stays
        # 'theta_rel'.
        tc = self.feature_config.target_col
        self.feature_config.target_col = "RZSM_25_avg"
        try:
            df = orig(self, filepath)
        finally:
            self.feature_config.target_col = tc
        if df is None:
            return df
        site = dl._site_from_filepath(str(filepath))
        for c in SOIL:                      # step-1 repair
            if c in df.columns:
                u = df[c].dropna().unique()
                if len(u) == 1:
                    df[c] = u[0]
        fc, wp = fcwp(site)
        span = fc - wp
        df["theta_rel"] = (df["RZSM_25_avg"] - wp) / span      # NOT clipped
        if norm_inputs:
            for c in SSM_COLS:
                if c in df.columns:
                    df[c] = (df[c] - wp) / span
        return df

    dl.DataProcessor.load_and_clean = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B"])
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
    a = ap.parse_args()
    install(norm_inputs=(a.variant == "B"))

    exp = rs.build_experiments()["baseline_no_doy"]
    cfg = CVConfig(); cfg.seq_length = 20
    cfg.dynamic_cols = exp["dynamic_cols"]; cfg.static_cols = exp["static_cols"]
    cfg.use_presto_static = exp["use_presto_static"]; cfg.add_temporal = False
    cfg._exclude_irrigation_static = bool(exp["exclude_irrigation_static"])
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(cfg, k, v)
    cfg.target_col = "theta_rel"                     # the only structural change

    out = BASE / "outputs" / "model_variants" / f"theta_rel_{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__); logging.disable(logging.INFO)
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"theta_rel variant {a.variant}  (normalize inputs={a.variant=='B'})  "
          f"target={cfg.target_col}\nseeds: {seeds}\n")

    for seed in seeds:
        cfg.seed = seed
        torch.manual_seed(seed); np.random.seed(seed)
        cfg.output_dir = str(out / f"seed{seed}")
        for fold in CV_FOLDS:
            r = rs._run_fold_with_overrides(fold, cfg, log)
            print(f"  seed{seed:<3} {r['test_station']}  R2(theta_rel space)="
                  f"{r['r2_may_oct']:+.4f}", flush=True)
    print("\ntraining done; score with score_theta_rel_step2.py")


if __name__ == "__main__":
    main()
