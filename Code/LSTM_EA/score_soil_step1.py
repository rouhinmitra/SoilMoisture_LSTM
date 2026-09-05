"""Score step-1 arms against paper_config_fixed from saved checkpoints, seed-paired."""
import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats as sst
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
import run_soil_static_step1 as S1

BASE = Path(__file__).resolve().parent
MV = BASE / "outputs" / "model_variants"
SEEDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,42]
ARMS = {"baseline":  (MV/"paper_config_fixed_sensitivity", "paper_config_fixed", False),
        "soil_filled":(MV/"soil_static_fc_wp_filled",      "soil_static_fc_wp",  True),
        "soil_asis":  (MV/"soil_static_fc_wp_asis",        "soil_static_fc_wp",  False)}


def cfg_for(variant):
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS[variant].items(): setattr(c, k, v)
    return c


rows = []
for arm, (ckdir, variant, fill) in ARMS.items():
    if fill: S1.install_fill()
    c = cfg_for(variant)
    for fold in CV_FOLDS:
        proc, d = A.make_proc(fold, c, (4, 10))
        ld = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"]), batch_size=256)
        dts = pd.to_datetime(d["dates_test"])
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
            k = (dts.month >= 5) & (dts.month <= 10) & (dts.year <= 2023)
            for grp, mask in [("ALL", k)] + [(y, k & (dts.year == y)) for y in range(2017, 2024)]:
                mask = np.asarray(mask)
                if mask.sum() < 20: continue
                a, b = yt[mask], yp[mask]
                rows.append(dict(arm=arm, site=fold["test"], seed=seed, year=grp,
                                 r2=1 - ((a-b)**2).sum()/((a-a.mean())**2).sum(),
                                 rmse=float(np.sqrt(((a-b)**2).mean())),
                                 offset=float((a-b).mean()), n=int(mask.sum())))
    if fill:  # undo the patch so the next arm is clean
        import importlib, src.data_loader as dl
        importlib.reload(dl); importlib.reload(A)
    print(f"  {arm}: done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(MV / "soil_step1_scored.csv", index=False)
print(f"\nscored -> {MV/'soil_step1_scored.csv'}  ({len(R)} rows)")
