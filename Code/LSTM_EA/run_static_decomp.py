"""
Decompose the -0.061 static-swap loss: was it AlphaEarth, Presto, or both?

The earlier field_avg arm swapped BOTH halves of the static block at once, so the
loss could not be attributed.  Four arms, each differing from paper_config_fixed
in exactly one respect:

  ae        AlphaEarth field-averaged   | Presto POINT      (isolates AlphaEarth)
  ae_renorm AlphaEarth field-avg + L2-renormalised | Presto POINT
  presto    AlphaEarth POINT            | Presto field-avg  (isolates Presto)
  both      = the existing field_avg arm, for reference

WHY RENORMALISE.  AlphaEarth embeddings are L2-normalised per pixel (||A|| = 1.0,
verified).  Averaging ~4500 of them drops the norm to 0.982-0.993, so the field
mean is not a valid point on the embedding manifold.  The renorm arm restores
||A|| = 1, isolating the DIRECTION the field average points from the magnitude
artefact of averaging.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sensitivity as rs
import variant_metrics as vm
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS

BASE = Path(__file__).resolve().parent
PT_DIR = "/Users/rouhinmitra/SM_work/Data/Base/"
FD_DIR = "/Users/rouhinmitra/SM_work/Data/Base_field/"
RN_DIR = "/Users/rouhinmitra/SM_work/Data/Base_field_renorm/"
PT_PRESTO = "/Users/rouhinmitra/SM_work/Code/Data/s2_pixels/presto_embeddings_fused_interpolated.csv"
FD_PRESTO = "/Users/rouhinmitra/SM_work/Code/Data/field_avg/presto_embeddings_field_interpolated.csv"
ARMS = {"ae": (FD_DIR, PT_PRESTO), "ae_renorm": (RN_DIR, PT_PRESTO),
        "presto": (PT_DIR, FD_PRESTO)}

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=list(ARMS))
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()
ddir, presto = ARMS[a.arm]

e = rs.build_experiments()["baseline_no_doy"]
cfg = CVConfig(); cfg.seq_length = 20
cfg.dynamic_cols = e["dynamic_cols"]; cfg.static_cols = e["static_cols"]
cfg.use_presto_static = e["use_presto_static"]; cfg.add_temporal = False
cfg._exclude_irrigation_static = bool(e["exclude_irrigation_static"])
for k, v in VARIANTS["paper_config_fixed"].items():
    setattr(cfg, k, v)
cfg.data_dir = ddir
cfg.presto_embeddings_path = presto

out = BASE / "outputs" / "model_variants" / f"statdec_{a.arm}"
out.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__); logging.disable(logging.INFO)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"static decomposition [{a.arm}]\n  data_dir={ddir}\n  presto={Path(presto).name}\nseeds: {seeds}\n")

rows = []
for seed in seeds:
    cfg.seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.output_dir = str(out / f"seed{seed}")
    for fold in CV_FOLDS:
        r = rs._run_fold_with_overrides(fold, cfg, log)
        rows.extend(vm.rows_for_fold(variant=f"statdec_{a.arm}", experiment="static_decomposition",
                                     test_station=r["test_station"], dates=r["dates_test"],
                                     y_true=r["y_true"], y_pred=r["y_pred"]))
        print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}", flush=True)
vm.append_rows(rows)
print(f"\ndone -> {out}")
