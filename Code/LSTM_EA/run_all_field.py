"""
ALL FIELD-AVERAGED remote sensing: S1 + S2 + AlphaEarth + Presto.

Every remotely sensed input is field-averaged (strict geometry: polygon eroded
10 m, CDL crop-masked, containment verified 100%).  The in-situ inputs are
untouched, because they ARE point measurements and there is nothing to average:
  SSM, SSM_avg, SWC_PI_F_2_1_1, SWC_PI_F_3_1_1   soil probes
  P_PI_F_1_1_1, P_PI_F_2_2_1, I                  precip / irrigation
  TA, RH, LE, NETRAD                             tower met + fluxes
  precip_jan_apr, irrigation                     annual statics

  DYNAMIC (25) = 11 in-situ + 11 field S2 (ndvi,b2..b12) + 3 field S1 (VV,VH,VV_VH_diff)
  STATIC (194) = 128 field Presto + 64 field AlphaEarth + precip_jan_apr + irrigation

vs paper_config_fixed the changes are: AlphaEarth point->field, Presto point->field,
S2 point->field, and +3 S1 channels.  Architecture, hyperparameters, folds and
seeds are unchanged.
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
FIELD_DIR = "/Users/rouhinmitra/SM_work/Data/Base_field/"          # field AlphaEarth
FIELD_PRESTO = f"{FA}/presto_embeddings_field_interpolated.csv"    # field Presto
S2C = ['ndvi','b2','b3','b4','b5','b6','b7','b8','b8a','b11','b12']
S1C = ['VV','VH','VV_VH_diff']
SM = {"ne1":"US-Ne1","ne2":"US-Ne2","ne3":"US-Ne3"}
_O = dl.DataProcessor.load_and_clean


def install():
    s2 = pd.read_csv(f"{FA}/s2_daily_field.csv", parse_dates=["Date"])
    s1 = pd.read_csv(f"{FA}/s1_daily_field.csv", parse_dates=["Date"])
    def patched(self, fp):
        fc = self.feature_config; keep = list(fc.dynamic_cols)
        fc.dynamic_cols = [c for c in keep if c not in S1C]
        try: df = _O(self, fp)
        finally: fc.dynamic_cols = keep
        if df is None: return df
        site = SM[dl._site_from_filepath(str(fp))]
        d = pd.to_datetime(df["Date"]).dt.normalize()
        a = s2[s2.Site == site][["Date"] + S2C].rename(columns={c: c+"_new" for c in S2C})
        df = df.merge(a, left_on=d, right_on="Date", how="left", suffixes=("", "_d"))
        for c in S2C: df[c] = df[c+"_new"]
        df = df.drop(columns=[c+"_new" for c in S2C]
                     + [c for c in ("key_0","Date_d") if c in df.columns])
        d2 = pd.to_datetime(df["Date"]).dt.normalize()
        df = df.merge(s1[s1.Site == site][["Date"]+S1C], left_on=d2, right_on="Date",
                      how="left", suffixes=("", "_d"))
        return df.drop(columns=[c for c in ("key_0","Date_d") if c in df.columns])
    dl.DataProcessor.load_and_clean = patched


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()
install()

e = rs.build_experiments()["baseline_no_doy"]
cfg = CVConfig(); cfg.seq_length = 20
cfg.dynamic_cols = list(e["dynamic_cols"]) + S1C
cfg.static_cols = e["static_cols"]; cfg.use_presto_static = e["use_presto_static"]
cfg.add_temporal = False
cfg._exclude_irrigation_static = bool(e["exclude_irrigation_static"])
for k, v in VARIANTS["paper_config_fixed"].items(): setattr(cfg, k, v)
cfg.data_dir = FIELD_DIR
cfg.presto_embeddings_path = FIELD_PRESTO

out = BASE/"outputs"/"model_variants"/"all_field"
out.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__); logging.disable(logging.INFO)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"ALL-FIELD | dynamic={len(cfg.dynamic_cols)} static={len(cfg.static_cols)}(+1 irr)")
print(f"  data_dir={cfg.data_dir}\n  presto={Path(cfg.presto_embeddings_path).name}\nseeds: {seeds}\n")

rows = []
for seed in seeds:
    cfg.seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.output_dir = str(out/f"seed{seed}")
    for fold in CV_FOLDS:
        r = rs._run_fold_with_overrides(fold, cfg, log)
        rows.extend(vm.rows_for_fold(variant="all_field", experiment="all_field_averaged",
                                     test_station=r["test_station"], dates=r["dates_test"],
                                     y_true=r["y_true"], y_pred=r["y_pred"]))
        print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}", flush=True)
vm.append_rows(rows)
print(f"\ndone -> {out}")
