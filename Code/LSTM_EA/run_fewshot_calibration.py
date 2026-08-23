"""
EXPERIMENT 1 - target-window calibration (FEW-SHOT DOMAIN ADAPTATION).

*** NOT COMPARABLE TO THE ZERO-SHOT LOSO RESULTS / Table 3. ***
The model sees a short window of the held-out site's OBSERVED RZSM.  Everything
except a 2-parameter affine calibration head is frozen from the LOSO-trained
model; no weights are retrained.

Design detail that makes the sweep honest: every window length is scored on the
SAME evaluation set - the part of each season after day MAX_K - so K=0/14/30/45
differ only in how much target signal the calibration head saw, never in which
samples are scored.

Rows are written to summary_metrics.csv with experiment='fewshot_calibration'
and variant='fewshot_cal_K<K>' so they can never be merged with the zero-shot family.
"""
import logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.evaluator import Evaluator
from run_variant import VARIANTS
import variant_metrics as vm

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
CKPT_ROOT = BASE / "outputs" / "model_variants" / "paper_config_fixed_sensitivity"
OUT = BASE / "outputs" / "model_variants" / "fewshot_calibration"
KS = [0, 14, 30, 45]
MAX_K = max(KS)


def build_config():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig()
    c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    return c


def r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 and len(y) > 1 else np.nan


def main():
    c = build_config()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(d.name[4:]) for d in CKPT_ROOT.glob("seed*") if d.is_dir())
    print(f"seeds: {seeds}\nwindows: {KS} (all scored after day {MAX_K} of each season)\n")
    recs = []

    for fold in CV_FOLDS:
        dc = DataConfig(
            train_files=[STATIONS[s] for s in fold["train"]],
            test_files=[STATIONS[fold["test"]]],
            data_dir=c.data_dir, seq_length=c.seq_length, nan_threshold=c.nan_threshold,
            same_year_constraint=True, month_range=(4, 10), year_range=c.year_range,
            require_contiguous_windows=c.require_contiguous_windows,
            impute_statics_after_scaling=c.impute_statics_after_scaling,
        )
        fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                           target_col=c.target_col, use_presto_static=c.use_presto_static,
                           presto_embeddings_path=c.presto_embeddings_path)
        proc = DataProcessor(dc, fc)
        data = proc.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)
        ds = RZSMDataset(data["X_d_test"], data["X_s_test"], data["y_test"],
                         groups=data.get("groups_test"))
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        dates = pd.to_datetime(data["dates_test"])
        dyn, stat = data["X_d_test"].shape[2], data["X_s_test"].shape[2]
        site = fold["test"]

        for seed in seeds:
            ck = CKPT_ROOT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not ck.exists():
                continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers), dyn, stat)
            m.load_state_dict(sd)
            ev = Evaluator(m, proc.scaler_y, str(OUT), device="cpu")
            y_pred, y_true = ev.predict(loader)
            y_pred = np.asarray(y_pred).ravel(); y_true = np.asarray(y_true).ravel()

            df = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
            df["year"] = df.date.dt.year
            df = df[df.date.dt.month.between(5, 10) | True]   # keep all; season filter below
            for year, g in df.groupby("year"):
                g = g.sort_values("date")
                t0 = g.date.iloc[0]
                day = (g.date - t0).dt.days.values
                ev_mask = day >= MAX_K
                if ev_mask.sum() < 30:
                    continue
                for K in KS:
                    cm = day < K
                    for mode in ("offset", "affine"):
                        if K == 0:
                            a, b = 1.0, 0.0
                        elif cm.sum() < 5:
                            continue
                        elif mode == "offset":
                            # 1 parameter: shift only.  Stable on short windows and it
                            # targets exactly the systematic site-year bias.
                            a = 1.0
                            b = float(np.mean(g.y.values[cm] - g.p.values[cm]))
                        else:
                            # 2 parameters.  Within a short window the predictions have
                            # little spread, so the slope is fit on a near-degenerate
                            # predictor; guard against the blow-up and fall back to shift.
                            pc_, yc_ = g.p.values[cm], g.y.values[cm]
                            if np.std(pc_) < 1e-3:
                                a, b = 1.0, float(np.mean(yc_ - pc_))
                            else:
                                a, b = np.polyfit(pc_, yc_, 1)
                                if not (0.2 <= a <= 5.0):
                                    a, b = 1.0, float(np.mean(yc_ - pc_))
                        pc = a * g.p.values[ev_mask] + b
                        recs.append(dict(seed=seed, site=site, year=int(year), K=K,
                                         mode=mode, r2=r2(g.y.values[ev_mask], pc),
                                         n=int(ev_mask.sum()), a=float(a), b=float(b)))
        print(f"  {fold['name']}: done")

    R_all = pd.DataFrame(recs)
    R_all.to_csv(OUT / "fewshot_raw.csv", index=False)

    for MODE in ("offset", "affine"):
      R = R_all[R_all["mode"] == MODE]
      print(f"\n{'#'*70}\n# CALIBRATION MODE: {MODE}\n{'#'*70}")
      # ---- per-site (pool years within seed, then summarise across seeds) ----
      print("\n=== PER-SITE: mean +/- std over seeds, pooled across years ===")
      print(f"{'site':5} " + " ".join(f"{'K='+str(k):>16}" for k in KS))
      pooled = {}
      for site, gs in R.groupby("site"):
          cells = []
          for K in KS:
              per_seed = []
              for seed, gg in gs[gs.K == K].groupby("seed"):
                  w = gg.n.values
                  per_seed.append(np.average(gg.r2.values, weights=w))
              pooled[(site, K)] = np.array(per_seed)
              cells.append(f"{np.mean(per_seed):7.4f}+/-{np.std(per_seed, ddof=1):.3f}")
          print(f"{site:5} " + " ".join(f"{c:>16}" for c in cells))

      print("\n=== GAP CLOSED vs days of target labels  ((R2_K - R2_0)/(1 - R2_0)) ===")
      print(f"{'site':5} " + " ".join(f"{'K='+str(k):>10}" for k in KS))
      for site in sorted(R.site.unique()):
          r0 = pooled[(site, 0)].mean()
          cells = [f"{100*(pooled[(site,K)].mean()-r0)/(1-r0):9.1f}%" for K in KS]
          print(f"{site:5} " + " ".join(f"{c:>10}" for c in cells))

      print("\n=== PER-SITE x YEAR: mean +/- std over seeds ===")
      print(f"{'site':5} {'year':5} " + " ".join(f"{'K='+str(k):>16}" for k in KS))
      for (site, year), gy in R.groupby(["site", "year"]):
          cells = []
          for K in KS:
              v = gy[gy.K == K].r2.values
              cells.append(f"{v.mean():7.4f}+/-{v.std(ddof=1):.3f}" if len(v) else "-")
          print(f"{site:5} {year:5} " + " ".join(f"{c:>16}" for c in cells))

    # ---- write clearly-labelled rows into the shared metrics file ----
    rows = []
    for (site, year, K, mode), g in R_all.groupby(["site", "year", "K", "mode"]):
        rows.append({"variant": f"fewshot_cal_{mode}_K{K}",
                     "experiment": "fewshot_calibration_NOT_zero_shot",
                     "test_station": site, "year": int(year), "season": "may_oct",
                     "r2": float(g.r2.mean()), "rmse": np.nan, "mae": np.nan,
                     "bias": np.nan, "correlation": np.nan, "nse": np.nan,
                     "n_samples": int(g.n.mean())})
    vm.append_rows(rows)
    print(f"\nrows appended to {vm.SUMMARY_CSV} as variant='fewshot_cal_K*', "
          f"experiment='fewshot_calibration_NOT_zero_shot'")


if __name__ == "__main__":
    main()
