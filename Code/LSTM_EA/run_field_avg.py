"""
paper_config_fixed on the FIELD-AVERAGED dataset, 15 seeds.

Only two things differ from the published baseline, both in the 194-dim static block:
  data_dir              Data/Base_field   (A00-A63 = field average, not tower pixel)
  presto_embeddings_path presto_embeddings_field_interpolated.csv
                        (regenerated from field-averaged S1 + S2; the generator was
                         first validated to reproduce the published embeddings from
                         the original point inputs to 3.6e-6)
Everything else - architecture, hyperparameters, folds, seeds, dynamic features,
scoring - is identical to paper_config_fixed.
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
FIELD_DIR = "/Users/rouhinmitra/SM_work/Data/Base_field/"
FIELD_PRESTO = "/Users/rouhinmitra/SM_work/Code/Data/field_avg/presto_embeddings_field_interpolated.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()

exp = rs.build_experiments()["baseline_no_doy"]
cfg = CVConfig(); cfg.seq_length = 20
cfg.dynamic_cols = exp["dynamic_cols"]; cfg.static_cols = exp["static_cols"]
cfg.use_presto_static = exp["use_presto_static"]; cfg.add_temporal = False
cfg._exclude_irrigation_static = bool(exp["exclude_irrigation_static"])
for k, v in VARIANTS["paper_config_fixed"].items():
    setattr(cfg, k, v)
cfg.data_dir = FIELD_DIR
cfg.presto_embeddings_path = FIELD_PRESTO

out = BASE / "outputs" / "model_variants" / "field_avg"
out.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__); logging.disable(logging.INFO)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"field-averaged run | data_dir={cfg.data_dir}\npresto={cfg.presto_embeddings_path}")
print(f"seeds: {seeds}\n")

rows = []
for seed in seeds:
    cfg.seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    cfg.output_dir = str(out / f"seed{seed}")
    for fold in CV_FOLDS:
        r = rs._run_fold_with_overrides(fold, cfg, log)
        rows.extend(vm.rows_for_fold(variant="field_avg", experiment="field_average",
                                     test_station=r["test_station"], dates=r["dates_test"],
                                     y_true=r["y_true"], y_pred=r["y_pred"]))
        print(f"  seed{seed:<3} {r['test_station']}  R2(May-Oct)={r['r2_may_oct']:+.4f}", flush=True)
vm.append_rows(rows)
print("\ndone -> outputs/model_variants/field_avg/")
