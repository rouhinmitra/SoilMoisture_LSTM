"""
PROBE DEPLOYMENT VALUE CURVE  -  TARGET-ADAPTED, *NOT* ZERO-SHOT.

Operational question: given a model trained on two sites, how long must a probe
sit in the ground at a third, and for how long after removal does it help?

Deployment is CAUSAL: adapt on a contiguous block of the held-out site's real
RZSM starting at the earliest year, evaluate only on LATER years the adaptation
never saw.  Skill is indexed by years-since-probe-removal.

Methods, all scored on IDENTICAL evaluation windows:
  adapted      - LOSO model adapted on the deployment block (head-only or full)
  zero_shot    - the same LOSO model, unadapted
  climatology  - constant = mean RZSM of the deployment block
  exp_filter   - SWI(T) from SSM with T and the affine map fitted on the block

Rows -> summary_metrics.csv with experiment='target_adapted_NOT_zero_shot'.
"""
import argparse, copy, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor, _add_swi_api_channels
from src.models import get_model
from run_variant import VARIANTS
import variant_metrics as vm

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
CKPT = BASE / "outputs" / "model_variants" / "paper_config_fixed_sensitivity"
DATA = Path("/Users/rouhinmitra/SM_work/Data/Base")
FILES = {"Ne1": "ne1_1_maize.csv", "Ne2": "ne2_1_maize.csv", "Ne3": "ne3_1_maize.csv"}

# name -> (first deployment year, last deployment year, month cutoff in last year or None)
DEPLOY = {"half_2017":     (2017, 2017, 7),
          "one_2017":      (2017, 2017, None),
          "two_2017_18":   (2017, 2018, None),
          "three_2017_19": (2017, 2019, None),
          # single-season blocks in different years: is probe value about WHICH
          # year you instrument rather than HOW LONG?
          "only_2018":     (2018, 2018, None),
          "only_2019":     (2019, 2019, None)}
EVAL_LAST = 2023
COMMON_EVAL = (2020, 2023)   # years every single-season block can be scored on


def build_config():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    return c


def r2(y, p):
    ss = np.sum((y - p) ** 2); st = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / st) if st > 0 and len(y) > 1 else np.nan


def swi_series(site, taus=(1, 2, 5, 10, 20, 40, 60)):
    df = pd.read_csv(DATA / FILES[site])
    df["date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df["Year"] = df.date.dt.year
    df = _add_swi_api_channels(df, taus, logging.getLogger())
    return df.set_index("date")[[f"swi_{t}" for t in taus]], taus


def adapt(model, xd, xs, y, mode, lr, epochs=60, seed=0):
    m = copy.deepcopy(model)
    if mode == "head":
        for p_ in m.parameters():
            p_.requires_grad = False
        for p_ in m.head.parameters():
            p_.requires_grad = True
    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    n = len(y); idx = torch.randperm(n, generator=g)
    nval = max(10, int(0.15 * n))
    vi, ti = idx[:nval], idx[nval:]
    dl = DataLoader(TensorDataset(xd[ti], xs[ti], y[ti]), batch_size=32, shuffle=True,
                    generator=g)
    best, best_state, bad = np.inf, copy.deepcopy(m.state_dict()), 0
    for _ in range(epochs):
        m.train()
        for a, b, t in dl:
            opt.zero_grad()
            loss = crit(m(a, b), t[:, -1:])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(crit(m(xd[vi], xs[vi]), y[vi][:, -1:]))
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best_state); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="head", choices=["head", "full"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--deployments", default=None,
                    help="comma-separated subset of DEPLOY keys (default: all)")
    a = ap.parse_args()
    lr = a.lr if a.lr is not None else (1e-3 if a.mode == "head" else 5e-5)
    c = build_config()
    todo = ([x for x in a.deployments.split(",")] if a.deployments else list(DEPLOY))
    seeds = sorted(int(d.name[4:]) for d in CKPT.glob("seed*") if d.is_dir())
    print(f"mode={a.mode}  lr={lr}  seeds={len(seeds)}\n")
    recs = []

    for fold in CV_FOLDS:
        site = fold["test"]
        dc = DataConfig(train_files=[STATIONS[s] for s in fold["train"]],
                        test_files=[STATIONS[site]], data_dir=c.data_dir,
                        seq_length=c.seq_length, nan_threshold=c.nan_threshold,
                        same_year_constraint=True, month_range=(4, 10), year_range=c.year_range,
                        require_contiguous_windows=c.require_contiguous_windows,
                        impute_statics_after_scaling=c.impute_statics_after_scaling)
        fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                           target_col=c.target_col, use_presto_static=c.use_presto_static,
                           presto_embeddings_path=c.presto_embeddings_path)
        proc = DataProcessor(dc, fc)
        d = proc.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)
        xd = torch.tensor(d["X_d_test"], dtype=torch.float32)
        xs = torch.tensor(d["X_s_test"], dtype=torch.float32)
        yt = torch.tensor(d["y_test"], dtype=torch.float32)
        dates = pd.to_datetime(d["dates_test"])
        yr = dates.year.values; mo = dates.month.values
        sy = proc.scaler_y
        y_true_all = sy.inverse_transform(d["y_test"][:, -1].reshape(-1, 1)).ravel()
        swi_df, taus = swi_series(site)
        swi_al = swi_df.reindex(dates)

        for dep in todo:
            first_y, last_y, cut_m = DEPLOY[dep]
            dep_mask = (yr >= first_y) & (yr <= last_y)
            if cut_m:
                dep_mask &= ~((yr == last_y) & (mo > cut_m))
            ev_years = [y for y in range(last_y + 1, EVAL_LAST + 1)]
            ev_mask = np.isin(yr, ev_years)
            if dep_mask.sum() < 50 or ev_mask.sum() < 50:
                continue

            # --- non-model baselines (seed-independent) ---
            clim = float(y_true_all[dep_mask].mean())
            best_t, best_ab, best_r = None, None, -np.inf
            for t in taus:
                sv = swi_al[f"swi_{t}"].values
                ok = dep_mask & ~np.isnan(sv)
                if ok.sum() < 20:
                    continue
                A, B = np.polyfit(sv[ok], y_true_all[ok], 1)
                rr = r2(y_true_all[ok], A * sv[ok] + B)
                if rr > best_r:
                    best_r, best_t, best_ab = rr, t, (A, B)

            for seed in seeds:
                ck = CKPT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
                if not ck.exists():
                    continue
                sd = torch.load(ck, map_location="cpu", weights_only=False)
                sd = sd.get("model_state_dict", sd)
                base = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                             dropout=c.dropout, num_layers=c.num_layers),
                                 xd.shape[2], xs.shape[2])
                base.load_state_dict(sd); base.eval()
                am = adapt(base, xd[dep_mask], xs[dep_mask], yt[dep_mask], a.mode, lr, seed=seed)
                with torch.no_grad():
                    p_zs = base(xd[ev_mask], xs[ev_mask]).numpy().ravel()
                    p_ad = am(xd[ev_mask], xs[ev_mask]).numpy().ravel()
                p_zs = sy.inverse_transform(p_zs.reshape(-1, 1)).ravel()
                p_ad = sy.inverse_transform(p_ad.reshape(-1, 1)).ravel()
                yv = y_true_all[ev_mask]; yv_yr = yr[ev_mask]
                sv = swi_al[f"swi_{best_t}"].values[ev_mask] if best_t else np.full(len(yv), np.nan)
                p_ef = best_ab[0] * sv + best_ab[1] if best_t else np.full(len(yv), np.nan)
                for name, pv in (("adapted", p_ad), ("zero_shot", p_zs),
                                 ("climatology", np.full(len(yv), clim)), ("exp_filter", p_ef)):
                    recs.append(dict(site=site, deploy=dep, method=name, seed=seed, year="ALL",
                                     lag=np.nan, r2=r2(yv, pv), n=len(yv)))
                    ck_ = (yv_yr >= COMMON_EVAL[0]) & (yv_yr <= COMMON_EVAL[1])
                    if ck_.sum() >= 20:
                        recs.append(dict(site=site, deploy=dep, method=name, seed=seed,
                                         year="COMMON", lag=np.nan,
                                         r2=r2(yv[ck_], pv[ck_]), n=int(ck_.sum())))
                    for y_ in sorted(set(yv_yr)):
                        k = yv_yr == y_
                        if k.sum() < 20:
                            continue
                        recs.append(dict(site=site, deploy=dep, method=name, seed=seed,
                                         year=int(y_), lag=int(y_ - last_y),
                                         r2=r2(yv[k], pv[k]), n=int(k.sum())))
        print(f"  {site}: done")

    R = pd.DataFrame(recs)
    out = BASE / "outputs" / "model_variants" / f"probe_deployment_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "raw.csv", index=False)

    P = R[R.year == "ALL"]
    print(f"\n=== POOLED over evaluation years, mean +/- std over {len(seeds)} seeds ({a.mode}) ===")
    print(f"{'site':5} {'deployment':14} {'eval yrs':9} " + " ".join(f"{m:>16}" for m in
          ["adapted", "zero_shot", "climatology", "exp_filter"]))
    for (site, dep), g in P.groupby(["site", "deploy"]):
        cells = []
        for m in ["adapted", "zero_shot", "climatology", "exp_filter"]:
            v = g[g.method == m].r2.values
            cells.append(f"{np.nanmean(v):7.4f}+/-{np.nanstd(v, ddof=1):.3f}" if len(v) else "-")
        ny = DEPLOY[dep][0]
        print(f"{site:5} {dep:14} {str(ny+1)+'-2023':>9} " + " ".join(f"{x:>16}" for x in cells))

    K = R[R.year == "COMMON"]
    if len(K):
        print(f"\n=== WHICH-YEAR TEST: all blocks scored on the SAME days "
              f"({COMMON_EVAL[0]}-{COMMON_EVAL[1]}) ===")
        print(f"{'site':5} {'deployment':14} " + " ".join(f"{m:>16}" for m in
              ["adapted", "zero_shot", "exp_filter"]))
        for (site, dep), g in K.groupby(["site", "deploy"]):
            cells = []
            for m in ["adapted", "zero_shot", "exp_filter"]:
                v = g[g.method == m].r2.values
                cells.append(f"{np.nanmean(v):7.4f}+/-{np.nanstd(v, ddof=1):.3f}" if len(v) else "-")
            print(f"{site:5} {dep:14} " + " ".join(f"{x:>16}" for x in cells))

    Y = R[~R.year.isin(["ALL", "COMMON"])]
    print(f"\n=== PRIMARY: skill vs YEARS SINCE PROBE REMOVAL ({a.mode}) ===")
    print(f"{'site':5} {'deployment':14} " + " ".join(f"{'lag'+str(l):>15}" for l in range(1, 7)))
    for (site, dep), g in Y.groupby(["site", "deploy"]):
        cells = []
        for l in range(1, 7):
            ad = g[(g.method == "adapted") & (g.lag == l)].r2.values
            zs = g[(g.method == "zero_shot") & (g.lag == l)].r2.values
            cells.append(f"{np.nanmean(ad):+.3f}/{np.nanmean(zs):+.3f}" if len(ad) else "       -")
        print(f"{site:5} {dep:14} " + " ".join(f"{x:>15}" for x in cells))
    print("   (adapted / zero_shot, mean over seeds)")

    rows = []
    for (site, dep, meth, y_), g in R.groupby(["site", "deploy", "method", "year"]):
        rows.append({"variant": f"probe_{a.mode}_{dep}_{meth}",
                     "experiment": "target_adapted_NOT_zero_shot",
                     "test_station": site, "year": y_, "season": "may_oct",
                     "r2": float(np.nanmean(g.r2)), "rmse": np.nan, "mae": np.nan,
                     "bias": np.nan, "correlation": np.nan, "nse": np.nan,
                     "n_samples": int(np.nanmean(g.n))})
    vm.append_rows(rows)
    print(f"\nrows -> summary_metrics.csv as variant='probe_{a.mode}_*', "
          f"experiment='target_adapted_NOT_zero_shot'")


if __name__ == "__main__":
    main()
