"""
INTERANNUAL SHRINKAGE: is the site-year offset just regression to the training mean?

Step 1 (read-only)  Per LOSO fold, regress OBSERVED site-year mean RZSM on
                    PREDICTED site-year mean, separately for the two TRAINING
                    sites and for the HELD-OUT site.  15 seeds.
                    slope ~1 on train and ~0.5 on held-out  -> transfer-specific.
                    slope ~0.5 on both                      -> a property of the loss.
Step 2 (read-only)  How much of the site-year offset does shrinkage account for?
                    predicted_offset = (1 - b) * (obs_site_year_mean - training_mean)
                    reported against the actual offset, as fraction of variance.
Step 3              The leakage-free correction.  Inflation factor f = 1/b with b
                    estimated on the TRAINING sites of that fold ONLY, applied to
                    the held-out site's site-year level, within-season anomalies
                    untouched:
                        new(t) = m + f*(pred_mean - m) + (pred(t) - pred_mean)

HOW LEAKAGE IS PREVENTED (step 3), explicitly:
  * b and m come from the two training sites of that fold and that seed.  The
    held-out site contributes NOTHING to either - not to the slope, not to the
    intercept, not to the training mean m.
  * b and m are re-estimated inside every (fold, seed); nothing is shared across
    folds, so a site never informs its own correction.
  * The validation year (val_years=2022) is EXCLUDED from the training-site fit
    by default, because model selection saw it; the included variant is reported
    alongside.
  * The only held-out-site quantity used is its own PREDICTED site-year mean,
    which is a model output computed from held-out-site INPUTS (SSM, met, ...).
    No held-out RZSM observation enters the calibration at any point.

RETROSPECTIVE, NOT REAL-TIME.  The correction needs a whole season of predictions
before the site-year mean exists, so it can only be applied after the fact.  It
is a post-season recalibration, not something deployable mid-season.

Rows -> outputs/model_variants/shrinkage/*.csv
"""
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sst
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cv import CV_FOLDS
from src.config import ModelConfig
from src.data_loader import RZSMDataset
from src.models import get_model
from src.evaluator import Evaluator
import run_season_split_partA as A
from run_probe_joint_retrain import build_config, r2, CKPT

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "shrinkage"
VAL_YEAR = 2022


def site_year_table(c, seeds):
    """Per fold x seed x site-year: observed and predicted season means, both splits."""
    rows, daily = [], []
    for fold in CV_FOLDS:
        proc, d = A.make_proc(fold, c, (4, 10))
        inv = lambda v: proc.scaler_y.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
        rev = {v: k for k, v in d["group_vocab"].items()}
        lab_tr = np.array([rev[g] for g in d["groups_train"]])
        lab_te = np.array([rev[g] for g in d["groups_test"]])
        y_tr = inv(d["y_train"][:, -1])
        y_te = inv(d["y_test"][:, -1])
        dt_te = pd.to_datetime(d["dates_test"])

        ld_tr = DataLoader(RZSMDataset(d["X_d_train"], d["X_s_train"], d["y_train"]),
                           batch_size=256)
        ld_te = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"]),
                           batch_size=256)
        for seed in seeds:
            ck = CKPT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not ck.exists():
                continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers),
                          d["X_d_train"].shape[2], d["X_s_train"].shape[2])
            m.load_state_dict(sd)
            ev = Evaluator(m, proc.scaler_y, str(BASE), device="cpu")
            p_tr, _ = ev.predict(ld_tr)
            p_te, _ = ev.predict(ld_te)
            p_tr = np.asarray(p_tr).ravel(); p_te = np.asarray(p_te).ravel()

            for split, lab, yv, pv in (("train", lab_tr, y_tr, p_tr),
                                       ("heldout", lab_te, y_te, p_te)):
                for sy in np.unique(lab):
                    k = lab == sy
                    if k.sum() < 30:
                        continue
                    site, yr = sy.split("_")
                    rows.append(dict(fold=fold["name"], seed=seed, split=split,
                                     site=site.replace("ne", "Ne"), year=int(yr),
                                     n=int(k.sum()), obs=float(yv[k].mean()),
                                     pred=float(pv[k].mean())))
            daily.append(pd.DataFrame(dict(fold=fold["name"], seed=seed,
                                           site=[s.split("_")[0].replace("ne", "Ne") for s in lab_te],
                                           year=[int(s.split("_")[1]) for s in lab_te],
                                           date=dt_te, y=y_te, p=p_te)))
        print(f"  {fold['test']}: done")
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def fit(x, y):
    r = sst.linregress(x, y)
    n = len(x)
    tcrit = sst.t.ppf(0.975, n - 2) if n > 2 else np.nan
    return dict(slope=r.slope, intercept=r.intercept, r2=r.rvalue ** 2, n=n,
                se=r.stderr, lo=r.slope - tcrit * r.stderr, hi=r.slope + tcrit * r.stderr,
                p=r.pvalue)


def main():
    c = build_config()
    seeds = sorted(int(p.name[4:]) for p in CKPT.glob("seed*") if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"shrinkage analysis, {len(seeds)} seeds, paper_config_fixed\n")
    T, DLY = site_year_table(c, seeds)
    T.to_csv(OUT / "site_year_means.csv", index=False)

    # ---------------- STEP 1 ----------------
    print("\n" + "=" * 100)
    print("STEP 1  observed site-year mean  ~  predicted site-year mean")
    print("        (slope < 1 = the model compresses interannual variance)")
    print("=" * 100)
    step1 = []
    for excl in (True, False):
        S = T[~((T.split == "train") & (T.year == VAL_YEAR))] if excl else T
        tag = f"train fit excl val year {VAL_YEAR}" if excl else "train fit incl val year"
        print(f"\n  --- {tag} ---")
        print(f"  {'fold':16} {'split':8} {'n_sy':5} {'slope obs~pred':>26} "
              f"{'R2':>7} {'slope pred~obs (b)':>20}")
        for fold in sorted(S.fold.unique()):
            for split in ("train", "heldout"):
                g = S[(S.fold == fold) & (S.split == split)]
                if not len(g):
                    continue
                per = [fit(h.pred.values, h.obs.values) for _, h in g.groupby("seed")
                       if len(h) > 2]
                perb = [fit(h.obs.values, h.pred.values) for _, h in g.groupby("seed")
                        if len(h) > 2]
                sl = np.array([q["slope"] for q in per])
                bb = np.array([q["slope"] for q in perb])
                r2s = np.array([q["r2"] for q in per])
                nsy = int(np.mean([q["n"] for q in per]))
                print(f"  {fold:16} {split:8} {nsy:5d} "
                      f"{sl.mean():+9.3f} +/-{sl.std(ddof=1):.3f} (seed sd)"
                      f"{r2s.mean():8.3f} {bb.mean():+13.3f} +/-{bb.std(ddof=1):.3f}")
                step1.append(dict(excl_val=excl, fold=fold, split=split, n_sy=nsy,
                                  slope_obs_pred=sl.mean(), slope_sd=sl.std(ddof=1),
                                  r2=r2s.mean(), b_pred_obs=bb.mean(), b_sd=bb.std(ddof=1)))
        for split in ("train", "heldout"):
            g = S[S.split == split]
            per = [fit(h.pred.values, h.obs.values) for (_, _), h in g.groupby(["fold", "seed"])
                   if len(h) > 2]
            perb = [fit(h.obs.values, h.pred.values) for (_, _), h in g.groupby(["fold", "seed"])
                    if len(h) > 2]
            sl = np.array([q["slope"] for q in per]); bb = np.array([q["slope"] for q in perb])
            print(f"  {'POOLED':16} {split:8} {'':5} {sl.mean():+9.3f} +/-{sl.std(ddof=1):.3f}"
                  f"{'':13}{np.mean([q['r2'] for q in per]):8.3f} {bb.mean():+13.3f} "
                  f"+/-{bb.std(ddof=1):.3f}")

        # single regression on seed-averaged means, with a proper CI
        print(f"\n  seed-averaged, one regression per split (95% CI on the slope):")
        for split in ("train", "heldout"):
            g = (S[S.split == split].groupby(["fold", "site", "year"])
                 [["obs", "pred"]].mean().reset_index())
            f1 = fit(g.pred.values, g.obs.values)
            f2 = fit(g.obs.values, g.pred.values)
            print(f"    {split:8} n={f1['n']:3d}  obs~pred slope={f1['slope']:+.3f} "
                  f"[{f1['lo']:+.3f},{f1['hi']:+.3f}]  int={f1['intercept']:+8.3f}  "
                  f"R2={f1['r2']:.3f}  |  pred~obs b={f2['slope']:+.3f} "
                  f"[{f2['lo']:+.3f},{f2['hi']:+.3f}]")
    pd.DataFrame(step1).to_csv(OUT / "step1_slopes.csv", index=False)

    # ---------------- STEP 2 ----------------
    print("\n" + "=" * 100)
    print("STEP 2  how much of the site-year offset does shrinkage account for?")
    print("        predicted_offset = (1 - b) * (obs_site_year_mean - training_mean)")
    print("=" * 100)
    S = T[~((T.split == "train") & (T.year == VAL_YEAR))]
    rows = []
    for (fold, seed), g in S.groupby(["fold", "seed"]):
        tr, te = g[g.split == "train"], g[g.split == "heldout"]
        if len(tr) < 3 or not len(te):
            continue
        b_tr = fit(tr.obs.values, tr.pred.values)["slope"]
        b_te = fit(te.obs.values, te.pred.values)["slope"]
        m = tr.obs.mean()
        for r in te.itertuples():
            rows.append(dict(fold=fold, seed=seed, site=r.site, year=r.year,
                             actual=r.obs - r.pred,
                             pred_from_train_b=(1 - b_tr) * (r.obs - m),
                             pred_from_heldout_b=(1 - b_te) * (r.obs - m)))
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "step2_offset_decomposition.csv", index=False)
    G = O.groupby(["site", "year"])[["actual", "pred_from_train_b", "pred_from_heldout_b"]].mean()
    for col, lbl in (("pred_from_heldout_b", "b fit ON the held-out site (descriptive upper bound)"),
                     ("pred_from_train_b", "b fit on TRAINING sites only (leakage-free)")):
        ss_res = ((G.actual - G[col]) ** 2).sum()
        ss_tot = ((G.actual - G.actual.mean()) ** 2).sum()
        rr = np.corrcoef(G.actual, G[col])[0, 1]
        print(f"  {lbl}")
        print(f"     variance of the offset explained = {1 - ss_res/ss_tot:+.3f}   "
              f"corr(actual, predicted) = {rr:+.3f}   corr^2 = {rr**2:.3f}")
    print()
    print(G.round(3).to_string())

    # ---------------- STEP 3 ----------------
    print("\n" + "=" * 100)
    print("STEP 3  leakage-free inflation correction on the held-out site")
    print("=" * 100)
    res = []
    for (fold, seed), g in S.groupby(["fold", "seed"]):
        tr = g[g.split == "train"]
        if len(tr) < 3:
            continue
        b = fit(tr.obs.values, tr.pred.values)["slope"]     # TRAINING SITES ONLY
        m = tr.obs.mean()                                    # TRAINING SITES ONLY
        cal = fit(tr.pred.values, tr.obs.values)             # TRAINING SITES ONLY
        f = 1.0 / b if b > 0.05 else np.nan
        dd = DLY[(DLY.fold == fold) & (DLY.seed == seed)]
        for (site, yr), h in dd.groupby(["site", "year"]):
            pm = h.p.mean()
            base = h.p.values
            infl = m + f * (pm - m) + (base - pm)
            regc = cal["intercept"] + cal["slope"] * pm + (base - pm)
            res.append(dict(fold=fold, seed=seed, site=site, year=yr, b_train=b, f=f,
                            m_train=m, n=len(h),
                            r2_base=r2(h.y.values, base),
                            r2_infl=r2(h.y.values, infl),
                            r2_regcal=r2(h.y.values, regc),
                            off_base=(h.y.values - base).mean(),
                            off_infl=(h.y.values - infl).mean()))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "step3_correction.csv", index=False)

    print(f"  inflation factor f = 1/b estimated per (fold, seed) on TRAINING SITES ONLY:")
    for fold, g in R.groupby("fold"):
        print(f"    {fold:16} b={g.b_train.mean():.3f}+/-{g.b_train.std(ddof=1):.3f}  "
              f"f={g.f.mean():.3f}+/-{g.f.std(ddof=1):.3f}  m={g.m_train.mean():.2f}")

    print(f"\n  PER SITE  (paired over seeds; delta = mean of per-seed differences)")
    print(f"  {'site':5} {'n_sy':5} {'baseline R2':>18} {'inflated R2':>18} {'delta':>18} "
          f"{'regcal R2':>18} {'delta':>18}")
    for site, g in R.groupby("site"):
        per = g.groupby("seed").apply(lambda h: pd.Series({
            "b": r2(np.concatenate([[0]]), np.concatenate([[0]])) if False else h.r2_base.mean(),
            "i": h.r2_infl.mean(), "c": h.r2_regcal.mean()}))
        d1 = per.i - per.b; d2 = per.c - per.b
        print(f"  {site:5} {g.year.nunique():5d} "
              f"{per.b.mean():+11.3f}+/-{per.b.std(ddof=1):.3f} "
              f"{per.i.mean():+11.3f}+/-{per.i.std(ddof=1):.3f} "
              f"{d1.mean():+11.3f}+/-{d1.std(ddof=1):.3f} "
              f"{per.c.mean():+11.3f}+/-{per.c.std(ddof=1):.3f} "
              f"{d2.mean():+11.3f}+/-{d2.std(ddof=1):.3f}")

    print(f"\n  PER SITE-YEAR")
    print(f"  {'site':5} {'year':5} {'base R2':>16} {'inflated R2':>16} {'delta':>16} "
          f"{'|offset| base':>14} {'|offset| infl':>14}")
    for (site, yr), g in R.groupby(["site", "year"]):
        per = g.groupby("seed")[["r2_base", "r2_infl", "off_base", "off_infl"]].mean()
        d = per.r2_infl - per.r2_base
        print(f"  {site:5} {yr:5d} {per.r2_base.mean():+9.3f}+/-{per.r2_base.std(ddof=1):.3f} "
              f"{per.r2_infl.mean():+9.3f}+/-{per.r2_infl.std(ddof=1):.3f} "
              f"{d.mean():+9.3f}+/-{d.std(ddof=1):.3f} "
              f"{abs(per.off_base.mean()):14.3f} {abs(per.off_infl.mean()):14.3f}")

    allp = R.groupby("seed")[["r2_base", "r2_infl", "r2_regcal"]].mean()
    for col, lbl in (("r2_infl", "inflation 1/b"), ("r2_regcal", "regression calibration")):
        d = allp[col] - allp.r2_base
        t, p = sst.ttest_rel(allp[col], allp.r2_base)
        print(f"\n  ALL site-years, {lbl}: base {allp.r2_base.mean():+.3f} -> "
              f"{allp[col].mean():+.3f}, delta {d.mean():+.3f}+/-{d.std(ddof=1):.3f}, "
              f"paired t={t:+.2f} p={p:.4f}, better in {int((d>0).sum())}/{len(d)} seeds")
    ob = R.groupby(["site", "year"])[["off_base", "off_infl"]].mean().abs()
    print(f"  mean |site-year offset|: {ob.off_base.mean():.3f} -> {ob.off_infl.mean():.3f}")
    print(f"\nfiles -> {OUT}")


if __name__ == "__main__":
    main()
