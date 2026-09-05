"""
LEVEL 3 - JOINT RETRAIN.  The CEILING of what D years of probe data can buy.

Level 1 (head-only adaptation, run_probe_deployment.py --mode head) returned ~0.
That bounds the METHOD, not the DATA.  This bounds the data: for each held-out
site, retrain the full model FROM SCRATCH on the two training sites PLUS a
contiguous deployment block of the held-out site's real RZSM.  If probe data has
value for this architecture at all, it shows up here; if it does not, no cheaper
adaptation method (linear head, FiLM, partial unfreeze) can find any.

Deployment blocks and their evaluation windows:
  one_2017        train +2017          -> eval 2018-2023   (later years)
  two_2017_18     train +2017-18       -> eval 2019-2023
  three_2017_19   train +2017-19       -> eval 2020-2023
  half_2017       train +Apr-Jul 2017  -> eval Aug-Oct 2017 (SAME season)

The half-season block is the within-season extrapolation test that the earlier
run got wrong: it scored the half-season block on LATER YEARS, which asks a
different question.  Here it is scored on the rest of the same season, which is
what the residual-persistence horizon (~17 d) actually speaks to.  Its eval mask
is STRICT by default: a window is scored only if all seq_length of its input
days fall after the training cutoff, so no evaluated window shares a single day
with the deployment block.  The looser "all Aug-Oct targets" variant is recorded
alongside as placement='same_season_overlap'.

Controls, scored on IDENTICAL days, paired seed-for-seed:
  joint      - the retrained model
  zero_shot  - paper_config_fixed_sensitivity checkpoint for the same fold+seed

PREPROCESSING IS HELD FIXED between the two arms: both use the scalers fit on
the two source sites only (what prepare_data gives a LOSO fold).  Refitting the
scalers to include the deployment block would confound the ceiling with a
preprocessing change; the block enters as training samples and nothing else.

MODEL SELECTION also stays source-side: validation is year 2022 of the two
training sites, exactly as run_single_fold does, and the deployment block goes
entirely into the training split.  Because the block is small relative to the
source sites, `block_fit` (the retrained model's R2 ON the deployment block) is
recorded for every run.  If block_fit is high while eval R2 is unmoved, the
model absorbed the probe data and it simply did not transfer -- dilution is
ruled out and the ceiling reading holds.

Rows -> outputs/model_variants/probe_joint_retrain/raw.csv (written incrementally)
     -> summary_metrics.csv with experiment='joint_retrain_ceiling'.
"""
import argparse, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.trainer import Trainer
from run_variant import VARIANTS
import variant_metrics as vm

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
CKPT = BASE / "outputs" / "model_variants" / "paper_config_fixed_sensitivity"
OUT = BASE / "outputs" / "model_variants" / "probe_joint_retrain"
SCRATCH = OUT / "_scratch"
EVAL_LAST = 2023
COMMON_EVAL = (2020, 2023)   # years every block's eval window contains

# name -> (first year, last year, cutoff month in last year or None, placement)
DEPLOY = {
    "half_2017":     (2017, 2017, 7,    "same_season"),
    "one_2017":      (2017, 2017, None, "later_years"),
    "two_2017_18":   (2017, 2018, None, "later_years"),
    "three_2017_19": (2017, 2019, None, "later_years"),
}


def build_config():
    """Identical construction to run_probe_deployment.build_config()."""
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig()
    c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]
    c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]
    c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    return c


def r2(y, p):
    ss = np.sum((y - p) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / st) if st > 0 and len(y) > 1 else np.nan


def prep_fold(fold, c):
    """prepare_data once per fold; every block and seed reuses it."""
    dc = DataConfig(train_files=[STATIONS[s] for s in fold["train"]],
                    test_files=[STATIONS[fold["test"]]], data_dir=c.data_dir,
                    seq_length=c.seq_length, nan_threshold=c.nan_threshold,
                    same_year_constraint=True, month_range=(4, 10),
                    year_range=c.year_range,
                    require_contiguous_windows=c.require_contiguous_windows,
                    impute_statics_after_scaling=c.impute_statics_after_scaling)
    fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                       target_col=c.target_col, use_presto_static=c.use_presto_static,
                       presto_embeddings_path=c.presto_embeddings_path)
    proc = DataProcessor(dc, fc)
    d = proc.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)
    return proc, d


def masks_for(dep, dates, seq_length):
    """Deployment mask and the evaluation masks it licenses."""
    first_y, last_y, cut_m, placement = DEPLOY[dep]
    yr, mo = dates.year.values, dates.month.values

    dep_mask = (yr >= first_y) & (yr <= last_y)
    if cut_m:
        dep_mask &= ~((yr == last_y) & (mo > cut_m))

    evals = {}
    if placement == "later_years":
        evals["later_years"] = np.isin(yr, list(range(last_y + 1, EVAL_LAST + 1)))
    else:
        # Rest of the SAME season.  Strict: the whole seq_length-day input window
        # must fall after the cutoff, so an evaluated window shares no day with
        # the block.  Loose: every target day after the cutoff.
        cutoff = pd.Timestamp(year=last_y, month=cut_m, day=31 if cut_m in (1,3,5,7,8,10,12) else 30)
        same_season = (yr == last_y) & (mo > cut_m)
        win_start = dates - pd.Timedelta(days=seq_length - 1)
        evals["same_season"] = same_season & np.asarray(win_start > cutoff)
        evals["same_season_overlap"] = same_season
    return dep_mask, evals


def load_zero_shot(fold, seed, c, dyn_dim, stat_dim):
    ck = CKPT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                              dropout=c.dropout, num_layers=c.num_layers),
                  dyn_dim, stat_dim)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def predict(model, xd, xs, sy, batch=256):
    model.eval()
    out = []
    for i in range(0, len(xd), batch):
        out.append(model(xd[i:i + batch], xs[i:i + batch]).numpy().ravel())
    p = np.concatenate(out) if out else np.array([])
    return sy.inverse_transform(p.reshape(-1, 1)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="comma-separated; default = all in the checkpoint tree")
    ap.add_argument("--deployments", default=None, help="comma-separated subset of DEPLOY keys")
    ap.add_argument("--folds", default=None, help="comma-separated test-site names, e.g. Ne3,Ne2")
    ap.add_argument("--tag", default="", help="suffix for the output directory (smoke tests)")
    a = ap.parse_args()

    c = build_config()
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else
             sorted(int(d.name[4:]) for d in CKPT.glob("seed*") if d.is_dir()))
    todo = a.deployments.split(",") if a.deployments else list(DEPLOY)
    folds = [f for f in CV_FOLDS if (not a.folds or f["test"] in a.folds.split(","))]

    out_dir = Path(str(OUT) + a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.csv"
    scratch = out_dir / "_scratch"

    n_runs = len(folds) * len(todo) * len(seeds)
    print(f"joint retrain: {len(folds)} folds x {len(todo)} blocks x {len(seeds)} seeds "
          f"= {n_runs} full trainings")
    print(f"blocks: {todo}\nseeds:  {seeds}\n")

    recs, done, t0 = [], 0, time.time()
    for fold in folds:
        site = fold["test"]
        proc, d = prep_fold(fold, c)
        sy = proc.scaler_y

        Xd_tr, Xs_tr, y_tr = d["X_d_train"], d["X_s_train"], d["y_train"]
        dates_tr = pd.to_datetime(d["dates_train"])
        Xd_te, Xs_te, y_te = d["X_d_test"], d["X_s_test"], d["y_test"]
        dates_te = pd.to_datetime(d["dates_test"])
        y_true_te = sy.inverse_transform(y_te[:, -1].reshape(-1, 1)).ravel()

        # source split, exactly as run_single_fold: val = val_years from the
        # two training stations, train = everything else.
        vy = c.val_years if isinstance(c.val_years, (list, tuple)) else (c.val_years,)
        vy = set(vy)
        src_years = dates_tr.year.values
        src_val_i = np.where(np.isin(src_years, list(vy)))[0]
        src_tr_i = np.where(~np.isin(src_years, list(vy)))[0]

        xd_te_t = torch.tensor(Xd_te, dtype=torch.float32)
        xs_te_t = torch.tensor(Xs_te, dtype=torch.float32)
        dyn_dim, stat_dim = Xd_tr.shape[2], Xs_tr.shape[2]

        for dep in todo:
            dep_mask, evals = masks_for(dep, dates_te, c.seq_length)
            if dep_mask.sum() < 50:
                print(f"  {site}/{dep}: deployment block too small ({dep_mask.sum()}), skipped")
                continue
            usable = {k: m for k, m in evals.items() if m.sum() >= 20}
            if not usable:
                print(f"  {site}/{dep}: no usable eval window, skipped")
                continue
            for k, m in evals.items():
                print(f"  {site}/{dep}: block n={int(dep_mask.sum())} "
                      f"({dates_te[dep_mask].min().date()}..{dates_te[dep_mask].max().date()}) "
                      f"| eval[{k}] n={int(m.sum())}"
                      + ("" if m.sum() >= 20 else "  (too small, skipped)"))

            # deployment-block windows, scaled by the SOURCE-fit scalers
            Xd_dep, Xs_dep, y_dep = Xd_te[dep_mask], Xs_te[dep_mask], y_te[dep_mask]
            xd_dep_t = torch.tensor(Xd_dep, dtype=torch.float32)
            xs_dep_t = torch.tensor(Xs_dep, dtype=torch.float32)
            y_dep_true = y_true_te[dep_mask]

            Xd_joint = np.concatenate([Xd_tr[src_tr_i], Xd_dep])
            Xs_joint = np.concatenate([Xs_tr[src_tr_i], Xs_dep])
            y_joint = np.concatenate([y_tr[src_tr_i], y_dep])
            train_ds = RZSMDataset(Xd_joint, Xs_joint, y_joint)
            val_ds = Subset(RZSMDataset(Xd_tr, Xs_tr, y_tr), src_val_i.tolist())
            frac = len(Xd_dep) / len(Xd_joint)
            print(f"      train {len(Xd_joint)} = {len(src_tr_i)} source + {len(Xd_dep)} probe "
                  f"({100*frac:.1f}%) | val {len(src_val_i)} (source {sorted(vy)})")

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                tl = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
                vl = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False)
                model = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                              dropout=c.dropout, num_layers=c.num_layers),
                                  dyn_dim, stat_dim)
                tr = Trainer(model, TrainingConfig(batch_size=c.batch_size, epochs=c.epochs,
                                                   learning_rate=c.learning_rate,
                                                   early_stopping_patience=c.early_stopping_patience,
                                                   weight_decay=c.weight_decay),
                             str(scratch), device="cpu")
                tr.train(tl, vl)
                jm = tr.model
                jm.eval()

                zs = load_zero_shot(fold, seed, c, dyn_dim, stat_dim)
                block_fit = r2(y_dep_true, predict(jm, xd_dep_t, xs_dep_t, sy))

                for placement, ev in usable.items():
                    xd_e = xd_te_t[ev]
                    xs_e = xs_te_t[ev]
                    yv = y_true_te[ev]
                    yv_yr = dates_te.year.values[ev]
                    preds = {"joint": predict(jm, xd_e, xs_e, sy)}
                    if zs is not None:
                        preds["zero_shot"] = predict(zs, xd_e, xs_e, sy)
                    for name, pv in preds.items():
                        base = dict(site=site, deploy=dep, placement=placement, method=name,
                                    seed=seed, block_fit=block_fit, n_block=int(dep_mask.sum()),
                                    probe_frac=frac, best_epoch=tr.best_epoch)
                        recs.append({**base, "year": "ALL", "lag": np.nan,
                                     "r2": r2(yv, pv), "n": len(yv)})
                        if placement == "later_years":
                            ck_ = (yv_yr >= COMMON_EVAL[0]) & (yv_yr <= COMMON_EVAL[1])
                            if ck_.sum() >= 20:
                                recs.append({**base, "year": "COMMON", "lag": np.nan,
                                             "r2": r2(yv[ck_], pv[ck_]), "n": int(ck_.sum())})
                            for y_ in sorted(set(yv_yr)):
                                k = yv_yr == y_
                                if k.sum() < 20:
                                    continue
                                recs.append({**base, "year": int(y_),
                                             "lag": int(y_ - DEPLOY[dep][1]),
                                             "r2": r2(yv[k], pv[k]), "n": int(k.sum())})

                done += 1
                el = time.time() - t0
                pd.DataFrame(recs).to_csv(raw_path, index=False)
                print(f"      [{done:3d}/{n_runs}] {site}/{dep}/seed{seed} "
                      f"ep{tr.best_epoch:3d} block_fit={block_fit:+.3f} "
                      f"| {el/60:.1f} min elapsed, ~{el/done*(n_runs-done)/60:.0f} min left",
                      flush=True)

    R = pd.DataFrame(recs)
    R.to_csv(raw_path, index=False)
    print(f"\nraw -> {raw_path}")
    report(R)

    rows = []
    for (site, dep, plc, meth, y_), g in R.groupby(["site", "deploy", "placement", "method", "year"]):
        rows.append({"variant": f"jointretrain_{dep}_{plc}_{meth}",
                     "experiment": "joint_retrain_ceiling",
                     "test_station": site, "year": y_, "season": "may_oct",
                     "r2": float(np.nanmean(g.r2)), "rmse": np.nan, "mae": np.nan,
                     "bias": np.nan, "correlation": np.nan, "nse": np.nan,
                     "n_samples": int(np.nanmean(g.n))})
    vm.append_rows(rows)
    print("rows -> summary_metrics.csv as variant='jointretrain_*', "
          "experiment='joint_retrain_ceiling'")


def report(R):
    if not len(R):
        return
    A = R[R.year == "ALL"]
    print("\n=== CEILING: joint retrain vs zero-shot, identical days, mean +/- std over seeds ===")
    print(f"{'site':5} {'block':14} {'placement':20} {'joint':>16} {'zero_shot':>16} "
          f"{'delta':>16} {'block_fit':>10}")
    for (site, dep, plc), g in A.groupby(["site", "deploy", "placement"]):
        j = g[g.method == "joint"].sort_values("seed")
        z = g[g.method == "zero_shot"].sort_values("seed")
        if not len(j):
            continue
        cells = [f"{j.r2.mean():7.4f}+/-{j.r2.std(ddof=1):.3f}"]
        if len(z):
            common = sorted(set(j.seed) & set(z.seed))
            dj = j.set_index("seed").loc[common].r2.values
            dz = z.set_index("seed").loc[common].r2.values
            dd = dj - dz
            cells += [f"{z.r2.mean():7.4f}+/-{z.r2.std(ddof=1):.3f}",
                      f"{dd.mean():+7.4f}+/-{dd.std(ddof=1):.3f}"]
        else:
            cells += ["-", "-"]
        print(f"{site:5} {dep:14} {plc:20} " + " ".join(f"{x:>16}" for x in cells)
              + f" {j.block_fit.mean():10.3f}")

    K = R[R.year == "COMMON"]
    if len(K):
        print(f"\n=== SAME DAYS FOR EVERY BLOCK LENGTH ({COMMON_EVAL[0]}-{COMMON_EVAL[1]}) ===")
        print(f"{'site':5} {'block':14} {'joint':>16} {'zero_shot':>16} {'delta':>16}")
        for (site, dep), g in K.groupby(["site", "deploy"]):
            j = g[g.method == "joint"].sort_values("seed")
            z = g[g.method == "zero_shot"].sort_values("seed")
            common = sorted(set(j.seed) & set(z.seed))
            if not common:
                continue
            dj = j.set_index("seed").loc[common].r2.values
            dz = z.set_index("seed").loc[common].r2.values
            dd = dj - dz
            print(f"{site:5} {dep:14} {dj.mean():9.4f}+/-{dj.std(ddof=1):.3f} "
                  f"{dz.mean():9.4f}+/-{dz.std(ddof=1):.3f} "
                  f"{dd.mean():+9.4f}+/-{dd.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
