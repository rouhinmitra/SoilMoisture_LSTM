"""Score theta_rel variants in VOLUMETRIC units, seed-paired against paper_config_fixed.

Predictions come out of the network in theta_rel units; they are inverted with the
TEST site's own FC/WP:   RZSM = theta_rel * (FC*100 - WP*100) + WP*100
so every number is directly comparable to the baseline's volumetric R2/offset.
"""
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
import run_theta_rel_step2 as T2

BASE = Path(__file__).resolve().parent
MV = BASE / "outputs" / "model_variants"
SEEDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,42]


def base_cfg():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items(): setattr(c, k, v)
    return c


rows = []
for arm in ["baseline", "theta_A", "theta_B"]:
    import importlib
    from src import data_loader as dl
    importlib.reload(dl); importlib.reload(A)
    c = base_cfg()
    if arm == "baseline":
        ckdir = MV / "paper_config_fixed_sensitivity"
    else:
        T2.install(norm_inputs=(arm == "theta_B"))
        c.target_col = "theta_rel"
        ckdir = MV / f"theta_rel_{arm[-1]}"
    for fold in CV_FOLDS:
        proc, d = A.make_proc(fold, c, (4, 10))
        ld = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"]), batch_size=256)
        dts = pd.to_datetime(d["dates_test"])
        site = fold["test"].lower()
        fc, wp = T2.fcwp(site); span = fc - wp
        for seed in SEEDS:
            ck = ckdir / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not ck.exists(): continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers),
                          d["X_d_test"].shape[2], d["X_s_test"].shape[2])
            m.load_state_dict(sd)
            yp, yt = Evaluator(m, proc.scaler_y, str(BASE), device="cpu").predict(ld)
            yp, yt = np.asarray(yp).ravel(), np.asarray(yt).ravel()
            if arm != "baseline":                 # invert to volumetric
                yp = yp * span + wp
                yt = yt * span + wp
            k = np.asarray((dts.month >= 5) & (dts.month <= 10) & (dts.year <= 2023))
            for grp, mask in [("ALL", k)] + [(y, k & np.asarray(dts.year == y))
                                             for y in range(2017, 2024)]:
                if mask.sum() < 20: continue
                a_, b_ = yt[mask], yp[mask]
                rows.append(dict(arm=arm, site=fold["test"], seed=seed, year=grp,
                                 r2=1 - ((a_-b_)**2).sum()/((a_-a_.mean())**2).sum(),
                                 rmse=float(np.sqrt(((a_-b_)**2).mean())),
                                 offset=float((a_-b_).mean()), n=int(mask.sum())))
    print(f"  {arm}: done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(MV / "theta_rel_step2_scored.csv", index=False)
print(f"\nscored -> {MV/'theta_rel_step2_scored.csv'} ({len(R)} rows)")
