"""
S1 BACKSCATTER AS DYNAMIC FEATURES.  15 seeds vs paper_config_fixed.

The ONLY change from the published baseline is +3 dynamic channels:
    VV, VH, VV_VH_diff          (dynamic_cols 22 -> 25)
Everything else - architecture, hyperparameters, folds, seeds, statics, the
original Data/Base station CSVs and the original Presto embeddings - is
untouched, so any delta is attributable to S1 alone.

VV_VH_diff is the cross-ratio.  VV and VH are already in dB, so their DIFFERENCE
is the linear ratio 10*log10(sigma_VV/sigma_VH); dividing dB values would be
meaningless.

Two arms, to separate "does S1 help" from "does S1 spatial support matter":
    --source point   tower-pixel S1 (the existing s1_presto_inputs.csv)
    --source field   field-averaged S1 (6.3k pixels, speckle-suppressed)
Both are linearly interpolated to daily and extended through 2024 so the window
set is IDENTICAL to baseline (year_range is (2017,2024) and April-2024 windows
exist; leaving 2024 as NaN would drop those windows and break comparability).

Because interpolation leaves no NaNs, nan_threshold never drops a window on
account of S1 - but in the growing season the median day is 3 d from a real
acquisition and the worst is 18 d, so these channels are far smoother than true
backscatter dynamics.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sensitivity as rs
import variant_metrics as vm
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS
from src import data_loader as dl

BASE = Path(__file__).resolve().parent
S1 = {"point": "/Users/rouhinmitra/SM_work/Code/Data/field_avg/s1_daily_point.csv",
      "field": "/Users/rouhinmitra/SM_work/Code/Data/field_avg/s1_daily_field.csv"}
COLS = ["VV", "VH", "VV_VH_diff"]
SITEMAP = {"ne1": "US-Ne1", "ne2": "US-Ne2", "ne3": "US-Ne3"}


def install(path):
    s1 = pd.read_csv(path, parse_dates=["Date"])
    orig = dl.DataProcessor.load_and_clean

    def patched(self, filepath):
        # load_and_clean validates dynamic_cols against the CSV, and VV/VH are merged
        # in only afterwards.  Hide them for the validation, restore immediately; the
        # windowing that follows reads the full 25-column list.
        fc = self.feature_config
        keep = list(fc.dynamic_cols)
        fc.dynamic_cols = [c for c in keep if c not in COLS]
        try:
            df = orig(self, filepath)
        finally:
            fc.dynamic_cols = keep
        if df is None:
            return df
        site = dl._site_from_filepath(str(filepath))
        sub = s1[s1.Site == SITEMAP[site]][["Date"] + COLS]
        d = df["Date"] if np.issubdtype(df["Date"].dtype, np.datetime64) \
            else pd.to_datetime(df["Date"])
        out = df.merge(sub, left_on=d.dt.normalize(), right_on="Date",
                       how="left", suffixes=("", "_s1"))
        out = out.drop(columns=[c for c in ("key_0", "Date_s1") if c in out.columns])
        return out

    dl.DataProcessor.load_and_clean = patched


ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["point", "field"])
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()
install(S1[a.source])

exp = rs.build_experiments()["baseline_no_doy"]
cfg = CVConfig(); cfg.seq_length = 20
cfg.dynamic_cols = list(exp["dynamic_cols"]) + COLS          # 22 -> 25
cfg.static_cols = exp["static_cols"]
cfg.use_presto_static = exp["use_presto_static"]; cfg.add_temporal = False
cfg._exclude_irrigation_static = bool(exp["exclude_irrigation_static"])
for k, v in VARIANTS["paper_config_fixed"].items():
    setattr(cfg, k, v)

out = BASE / "outputs" / "model_variants" / f"s1_feat_{a.source}"
out.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__); logging.disable(logging.INFO)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"S1-as-features [{a.source}] | dynamic_cols={len(cfg.dynamic_cols)} "
      f"(+{len(COLS)}: {COLS})\nseeds: {seeds}\n")

rows = []
for seed in seeds:
    cfg.seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.output_dir = str(out / f"seed{seed}")
    for fold in CV_FOLDS:
        r = rs._run_fold_with_overrides(fold, cfg, log)
        rows.extend(vm.rows_for_fold(variant=f"s1_feat_{a.source}",
                                     experiment="s1_dynamic_features",
                                     test_station=r["test_station"], dates=r["dates_test"],
                                     y_true=r["y_true"], y_pred=r["y_pred"]))
        print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}", flush=True)
vm.append_rows(rows)
print(f"\ndone -> {out}")
