"""Score the S1-feature arms against paper_config_fixed, seed-paired, from checkpoints."""
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
from src import data_loader as dl
from torch.utils.data import DataLoader
import run_season_split_partA as A

BASE = Path(__file__).resolve().parent
MV = BASE / "outputs" / "model_variants"
SEEDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,42]
COLS = ["VV", "VH", "VV_VH_diff"]
SM = {"ne1": "US-Ne1", "ne2": "US-Ne2", "ne3": "US-Ne3"}
S1 = {"point": "/Users/rouhinmitra/SM_work/Code/Data/field_avg/s1_daily_point.csv",
      "field": "/Users/rouhinmitra/SM_work/Code/Data/field_avg/s1_daily_field.csv"}
_ORIG = dl.DataProcessor.load_and_clean


def install(path):
    if path is None:
        dl.DataProcessor.load_and_clean = _ORIG
        return
    s1 = pd.read_csv(path, parse_dates=["Date"])
    def patched(self, fp):
        fc = self.feature_config; keep = list(fc.dynamic_cols)
        fc.dynamic_cols = [c for c in keep if c not in COLS]
        try: df = _ORIG(self, fp)
        finally: fc.dynamic_cols = keep
        if df is None: return df
        sub = s1[s1.Site == SM[dl._site_from_filepath(str(fp))]][["Date"] + COLS]
        d = pd.to_datetime(df["Date"])
        out = df.merge(sub, left_on=d.dt.normalize(), right_on="Date", how="left",
                       suffixes=("", "_s1"))
        return out.drop(columns=[c for c in ("key_0", "Date_s1") if c in out.columns])
    dl.DataProcessor.load_and_clean = patched


def cfg(add_s1):
    e = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = list(e["dynamic_cols"]) + (COLS if add_s1 else [])
    c.static_cols = e["static_cols"]; c.use_presto_static = e["use_presto_static"]
    c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items(): setattr(c, k, v)
    return c


rows = []
for arm, ck, src in [("baseline", MV/"paper_config_fixed_sensitivity", None),
                     ("s1_point", MV/"s1_feat_point", "point"),
                     ("s1_field", MV/"s1_feat_field", "field")]:
    install(S1[src] if src else None)
    import importlib; importlib.reload(A)
    c = cfg(src is not None)
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
install(None)
R = pd.DataFrame(rows); R.to_csv(MV/"s1_features_scored.csv", index=False)
print(f"\nscored -> {MV/'s1_features_scored.csv'} ({len(R)} rows)")
