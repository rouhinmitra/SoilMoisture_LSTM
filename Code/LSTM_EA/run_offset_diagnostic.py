"""
EXPERIMENT 2 - decoupling diagnostic.  No new training runs.

Per site-year systematic offset = mean(observed - predicted) under
paper_config_fixed, averaged over the 15 LOSO seeds, vs SIX pre-specified
candidate drivers.  Six tests, Holm-corrected.  Presto dims are deliberately
NOT tested (128 tests on 21 points would manufacture hits).
"""
import logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy import stats as sstats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor, RZSMDataset, _add_swi_api_channels, PRECIP_COLS
from src.models import get_model
from src.evaluator import Evaluator
from run_variant import VARIANTS

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
CKPT = BASE / "outputs" / "model_variants" / "paper_config_fixed_sensitivity"
DATA = Path("/Users/rouhinmitra/SM_work/Data/Base")
FILES = {"Ne1": "ne1_1_maize.csv", "Ne2": "ne2_1_maize.csv", "Ne3": "ne3_1_maize.csv"}


def build_config():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    return c


def offsets(c):
    """mean(obs - pred) per site-year, averaged over seeds."""
    seeds = sorted(int(d.name[4:]) for d in CKPT.glob("seed*") if d.is_dir())
    recs = []
    for fold in CV_FOLDS:
        dc = DataConfig(train_files=[STATIONS[s] for s in fold["train"]],
                        test_files=[STATIONS[fold["test"]]], data_dir=c.data_dir,
                        seq_length=c.seq_length, nan_threshold=c.nan_threshold,
                        same_year_constraint=True, month_range=(4, 10), year_range=c.year_range,
                        require_contiguous_windows=c.require_contiguous_windows,
                        impute_statics_after_scaling=c.impute_statics_after_scaling)
        fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                           target_col=c.target_col, use_presto_static=c.use_presto_static,
                           presto_embeddings_path=c.presto_embeddings_path)
        proc = DataProcessor(dc, fc)
        d = proc.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)
        loader = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"],
                                        groups=d.get("groups_test")), batch_size=256)
        dates = pd.to_datetime(d["dates_test"])
        for seed in seeds:
            ck = CKPT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not ck.exists():
                continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers),
                          d["X_d_test"].shape[2], d["X_s_test"].shape[2])
            m.load_state_dict(sd)
            yp, yt = Evaluator(m, proc.scaler_y, str(BASE), device="cpu").predict(loader)
            t = pd.DataFrame({"year": dates.year, "res": np.asarray(yt).ravel() - np.asarray(yp).ravel()})
            t = t[t.year.between(2017, 2023)]
            for yr, g in t.groupby("year"):
                recs.append(dict(site=fold["test"], year=int(yr), seed=seed, offset=g.res.mean()))
    R = pd.DataFrame(recs)
    return R.groupby(["site", "year"]).offset.mean().reset_index()


def drivers():
    """Six pre-specified candidates per site-year."""
    irr = pd.read_excel(BASE / "Irrigation-data.xlsx")
    out = []
    for site, fn in FILES.items():
        df = pd.read_csv(DATA / fn)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        df["Year"] = df.Date.dt.year
        df = _add_swi_api_channels(df, (60,), logging.getLogger())   # full record, pre-filter
        p_col = next((x for x in PRECIP_COLS if x in df.columns), None)
        gs = df[df.Date.dt.month.between(5, 10)]
        for yr, g in gs.groupby("Year"):
            if not 2017 <= yr <= 2023:
                continue
            ta, rh = g["TA_1_1_1"], g["RH_1_1_1"]
            es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))          # kPa
            vpd = (es * (1 - rh / 100.0)).mean()
            # LE (W/m2) -> ET (mm/day): *0.0864 MJ/m2/day, / 2.45 MJ/kg
            et = (g["LE_1_1_1"] * 0.0864 / 2.45).mean()
            col = f"US-{site}"
            irr_tot = float(irr.loc[irr.Year == yr, col].iloc[0]) if col in irr.columns and (irr.Year == yr).any() else 0.0
            out.append(dict(site=site, year=int(yr),
                            precip_total=float(g[p_col].sum()) if p_col else np.nan,
                            et_mean=float(et), swi60_mean=float(g["swi_60"].mean()),
                            irrigation=irr_tot, ssm_mean=float(g["SSM_avg"].mean()),
                            vpd_mean=float(vpd)))
    return pd.DataFrame(out)


def main():
    c = build_config()
    off = offsets(c)
    dr = drivers()
    M = off.merge(dr, on=["site", "year"])
    print(f"n site-years = {len(M)}\n")
    print(M.round(3).to_string(index=False))

    cands = ["precip_total", "et_mean", "swi60_mean", "irrigation", "ssm_mean", "vpd_mean"]
    res = []
    for cd in cands:
        r, p = sstats.pearsonr(M[cd], M.offset)
        rs_, ps_ = sstats.spearmanr(M[cd], M.offset)
        res.append(dict(candidate=cd, pearson=r, p_pearson=p, spearman=rs_, p_spearman=ps_))
    R = pd.DataFrame(res)

    # Holm on the Pearson p-values (6 tests)
    R = R.sort_values("p_pearson").reset_index(drop=True)
    mtests = len(R)
    R["holm_alpha"] = [0.05 / (mtests - i) for i in range(mtests)]
    surv, still = [], True
    for i, row in R.iterrows():
        still = still and row.p_pearson <= row.holm_alpha
        surv.append(still)
    R["survives_holm"] = surv

    print("\n=== six pre-specified candidates vs site-year offset (n=%d) ===" % len(M))
    print(R.round(4).to_string(index=False))
    print("\nSurvives Holm at alpha=0.05:",
          ", ".join(R[R.survives_holm].candidate) or "NOTHING")


if __name__ == "__main__":
    main()
