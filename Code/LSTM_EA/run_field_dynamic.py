"""
Field-averaged DYNAMIC Sentinel-2 in the LSTM, optionally with S1 alongside.

Replaces the 11 Sentinel-2 dynamic columns in the station CSVs
(ndvi, b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12) with the field-averaged
extraction, linearly interpolated to daily and extended through 2024 so the
window set matches baseline.

  --arm s2      baseline with the 11 S2 columns swapped        (22 dynamic cols)
  --arm s2_s1   the above plus VV, VH, VV_VH_diff              (25 dynamic cols)

A KNOWN CONFOUND, STATED UP FRONT.  The station S2 columns are populated on all
2483 days of 2017-2023 while only 430 are real overpasses, and they correlate
only 0.82-0.88 with s2_presto_inputs.csv with per-band ratios spanning
10.6k-15.5k - so they came from a different extraction AND a different
gap-filling than anything in this repo.  Swapping them therefore changes spatial
support AND fill method together; a delta cannot be attributed to one alone.
The scale convention does NOT matter (every feature is StandardScaler'd, so a
constant per-band factor cancels); the fill method does.

Field S2 is far better sampled than S1: 25.6% of growing-season days are real
observations, median gap 1 d, max 12 d - against S1's 7.4% / 3 d / 18 d.

Statics (AlphaEarth + Presto) are the ORIGINAL point-based ones here, so this
isolates the dynamic block from the earlier static-swap result (-0.061).
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
FA = "/Users/rouhinmitra/SM_work/Code/Data/field_avg"
S2_COLS = ['ndvi', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a', 'b11', 'b12']
S1_COLS = ['VV', 'VH', 'VV_VH_diff']
SM = {"ne1": "US-Ne1", "ne2": "US-Ne2", "ne3": "US-Ne3"}
_ORIG = dl.DataProcessor.load_and_clean


def install(add_s1):
    s2 = pd.read_csv(f"{FA}/s2_daily_field.csv", parse_dates=["Date"])
    s1 = pd.read_csv(f"{FA}/s1_daily_field.csv", parse_dates=["Date"]) if add_s1 else None

    def patched(self, fp):
        fc = self.feature_config; keep = list(fc.dynamic_cols)
        fc.dynamic_cols = [c for c in keep if c not in S1_COLS]   # S1 merged in after
        try:
            df = _ORIG(self, fp)
        finally:
            fc.dynamic_cols = keep
        if df is None:
            return df
        site = SM[dl._site_from_filepath(str(fp))]
        d = pd.to_datetime(df["Date"]).dt.normalize()
        # --- replace the S2 block ---
        a = s2[s2.Site == site][["Date"] + S2_COLS].rename(
            columns={c: c + "_new" for c in S2_COLS})
        df = df.merge(a, left_on=d, right_on="Date", how="left", suffixes=("", "_d"))
        for c in S2_COLS:
            df[c] = df[c + "_new"]
        df = df.drop(columns=[c + "_new" for c in S2_COLS]
                     + [c for c in ("key_0", "Date_d") if c in df.columns])
        # --- add S1 ---
        if s1 is not None:
            d2 = pd.to_datetime(df["Date"]).dt.normalize()
            b = s1[s1.Site == site][["Date"] + S1_COLS]
            df = df.merge(b, left_on=d2, right_on="Date", how="left", suffixes=("", "_d"))
            df = df.drop(columns=[c for c in ("key_0", "Date_d") if c in df.columns])
        return df

    dl.DataProcessor.load_and_clean = patched


ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["s2", "s2_s1"])
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()
add_s1 = a.arm == "s2_s1"
install(add_s1)

e = rs.build_experiments()["baseline_no_doy"]
cfg = CVConfig(); cfg.seq_length = 20
cfg.dynamic_cols = list(e["dynamic_cols"]) + (S1_COLS if add_s1 else [])
cfg.static_cols = e["static_cols"]; cfg.use_presto_static = e["use_presto_static"]
cfg.add_temporal = False
cfg._exclude_irrigation_static = bool(e["exclude_irrigation_static"])
for k, v in VARIANTS["paper_config_fixed"].items():
    setattr(cfg, k, v)

out = BASE / "outputs" / "model_variants" / f"fielddyn_{a.arm}"
out.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__); logging.disable(logging.INFO)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"field dynamic [{a.arm}] | dynamic_cols={len(cfg.dynamic_cols)}\nseeds: {seeds}\n")

rows = []
for seed in seeds:
    cfg.seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.output_dir = str(out / f"seed{seed}")
    for fold in CV_FOLDS:
        r = rs._run_fold_with_overrides(fold, cfg, log)
        rows.extend(vm.rows_for_fold(variant=f"fielddyn_{a.arm}",
                                     experiment="field_dynamic_s2",
                                     test_station=r["test_station"], dates=r["dates_test"],
                                     y_true=r["y_true"], y_pred=r["y_pred"]))
        print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}", flush=True)
vm.append_rows(rows)
print(f"\ndone -> {out}")
