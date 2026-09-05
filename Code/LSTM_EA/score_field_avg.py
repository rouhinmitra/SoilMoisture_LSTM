"""Score the field-averaged arm against paper_config_fixed, seed-paired, from checkpoints."""
import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.INFO)
import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS
from run_variant import VARIANTS
from src.config import ModelConfig
from src.models import get_model
from src.evaluator import Evaluator
from src.data_loader import RZSMDataset
from torch.utils.data import DataLoader
import run_season_split_partA as A

BASE = Path(__file__).resolve().parent
MV = BASE / "outputs" / "model_variants"
SEEDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,42]
FIELD_DIR = "/Users/rouhinmitra/SM_work/Data/Base_field/"
FIELD_PRESTO = "/Users/rouhinmitra/SM_work/Code/Data/field_avg/presto_embeddings_field_interpolated.csv"

def cfg(field):
    e = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = e["dynamic_cols"]; c.static_cols = e["static_cols"]
    c.use_presto_static = e["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items(): setattr(c, k, v)
    if field:
        c.data_dir = FIELD_DIR; c.presto_embeddings_path = FIELD_PRESTO
    return c

rows = []
for arm, ck, fld in [("baseline", MV/"paper_config_fixed_sensitivity", False),
                     ("field_avg", MV/"field_avg", True)]:
    c = cfg(fld)
    for fold in CV_FOLDS:
        proc, d = A.make_proc(fold, c, (4, 10))
        ld = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"]), batch_size=256)
        dts = pd.to_datetime(d["dates_test"])
        for seed in SEEDS:
            p = ck / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not p.exists(): continue
            sd = torch.load(p, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers),
                          d["X_d_test"].shape[2], d["X_s_test"].shape[2])
            m.load_state_dict(sd)
            yp, yt = Evaluator(m, proc.scaler_y, str(BASE), device="cpu").predict(ld)
            yp, yt = np.asarray(yp).ravel(), np.asarray(yt).ravel()
            k = np.asarray((dts.month>=5)&(dts.month<=10)&(dts.year<=2023))
            for g, msk in [("ALL", k)] + [(y, k & np.asarray(dts.year==y)) for y in range(2017,2024)]:
                if msk.sum() < 20: continue
                a_, b_ = yt[msk], yp[msk]
                rows.append(dict(arm=arm, site=fold["test"], seed=seed, year=g,
                                 r2=1-((a_-b_)**2).sum()/((a_-a_.mean())**2).sum(),
                                 rmse=float(np.sqrt(((a_-b_)**2).mean())),
                                 offset=float((a_-b_).mean()), n=int(msk.sum())))
    print(f"  {arm}: done", flush=True)
R = pd.DataFrame(rows); R.to_csv(MV/"field_avg_scored.csv", index=False)
print(f"\nscored -> {MV/'field_avg_scored.csv'} ({len(R)} rows)")
