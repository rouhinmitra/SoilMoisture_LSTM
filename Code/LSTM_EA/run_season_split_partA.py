"""
PART A - OFFSET DIAGNOSTIC BY SEASON.  Read-only; no training.

Season (Makki): in-season = Jun-Oct (vegetated), off-season = Nov-May.
Sites are planted late Apr / early May but stay barely covered until end of May.

A PIPELINE FACT THAT REFRAMES THE QUESTION.  The published config filters rows to
month_range=(4,10) BEFORE windowing, so the paper_config_fixed checkpoints were
trained and scored on April-October only.  Of the seven off-season months
(Nov,Dec,Jan,Feb,Mar,Apr,May) the published outputs contain TWO: April and May.
Nov-Mar has never entered the model at all -- although the station CSVs carry
complete year-round records at all three sites.  So this runs the diagnostic at
two scopes and reports both:

  scope='published'  in=Jun-Oct, off=Apr-May.  Exactly the existing outputs;
                     this is the regime any conclusion about the current model
                     has to live in.
  scope='allmonths'  in=Jun-Oct, off=Nov-May.  The same 15 checkpoints scored on
                     the full year.  Windows are built with month_range=(1,12)
                     but scaled by the SCALERS FROM THE PUBLISHED FOLD, so the
                     inputs reaching the network are exactly what training
                     defined; only the evaluation footprint widens.  Nov-Mar is
                     out-of-distribution for these models by construction and is
                     labelled as such, not treated as a fair test.

The transplant is verified, not assumed: predictions on the Apr-Oct dates common
to both scopes must agree to within a tolerance, else the run aborts.

Also computes the MODEL-FREE check of Makki's actual claim -- observed SSM->RZSM
coupling by season, straight from the station CSVs, no model involved.

Rows -> outputs/model_variants/season_split/partA_*.csv
     -> summary_metrics.csv, experiment='season_split', variant='offset_by_season'.
"""
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cv import CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.evaluator import Evaluator
import variant_metrics as vm
from run_probe_joint_retrain import build_config, r2, CKPT

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "season_split"
DATA = Path("/Users/rouhinmitra/SM_work/Data/Base")
FILES = {"Ne1": "ne1_1_maize.csv", "Ne2": "ne2_1_maize.csv", "Ne3": "ne3_1_maize.csv"}

IN_MONTHS = [6, 7, 8, 9, 10]                     # Makki: vegetated growing season
OFF_MONTHS = [11, 12, 1, 2, 3, 4, 5]
LOW_SD = 2.0


def season_of(months):
    return np.where(np.isin(months, IN_MONTHS), "in", "off")


def make_proc(fold, c, month_range, scalers=None):
    dc = DataConfig(train_files=[STATIONS[s] for s in fold["train"]],
                    test_files=[STATIONS[fold["test"]]], data_dir=c.data_dir,
                    seq_length=c.seq_length, nan_threshold=c.nan_threshold,
                    same_year_constraint=True, month_range=month_range,
                    year_range=c.year_range,
                    require_contiguous_windows=c.require_contiguous_windows,
                    impute_statics_after_scaling=c.impute_statics_after_scaling)
    fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                       target_col=c.target_col, use_presto_static=c.use_presto_static,
                       presto_embeddings_path=c.presto_embeddings_path)
    proc = DataProcessor(dc, fc)
    if scalers is not None:
        # Transplant the published fold's fitted scalers and neuter refitting, so
        # the widened month range changes the evaluation footprint and NOTHING else.
        for name, sc in scalers.items():
            setattr(proc, name, sc)
            sc.fit = (lambda *a, **k: sc)
    d = proc.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)
    return proc, d


def predict_all_seeds(fold, c, proc, d, seeds):
    loader = DataLoader(RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"],
                                    groups=d.get("groups_test")), batch_size=256)
    dates = pd.to_datetime(d["dates_test"])
    rows = []
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
        rows.append(pd.DataFrame({"site": fold["test"], "date": dates, "seed": seed,
                                  "y": np.asarray(yt).ravel(), "p": np.asarray(yp).ravel()}))
    return pd.concat(rows, ignore_index=True)


def coupling_by_season():
    """Makki's claim, model-free: observed SSM -> RZSM coupling, per site x season."""
    out = []
    for site, fn in FILES.items():
        df = pd.read_csv(DATA / fn)
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("date").sort_values("date")
        df = df[(df.date.dt.year >= 2017) & (df.date.dt.year <= 2023)]
        df["season"] = season_of(df.date.dt.month.values)
        for (yr, s), g in df.groupby([df.date.dt.year, "season"]):
            g = g[["SSM", "RZSM_25_avg"]].dropna()
            if len(g) < 20:
                continue
            out.append(dict(site=site, year=yr, season=s, n=len(g),
                            r=float(np.corrcoef(g.SSM, g.RZSM_25_avg)[0, 1]),
                            rzsm_sd=float(g.RZSM_25_avg.std())))
    return pd.DataFrame(out)


def main():
    c = build_config()
    seeds = sorted(int(p.name[4:]) for p in CKPT.glob("seed*") if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Part A: season offset diagnostic, {len(seeds)} seeds, read-only\n")
    print(f"in-season  = months {IN_MONTHS}")
    print(f"off-season = months {OFF_MONTHS}\n")

    pubs, alls = [], []
    for fold in CV_FOLDS:
        proc_p, d_p = make_proc(fold, c, (4, 10))
        P = predict_all_seeds(fold, c, proc_p, d_p, seeds)
        pubs.append(P)

        scalers = {"scaler_dyn": proc_p.scaler_dyn, "scaler_stat": proc_p.scaler_stat,
                   "scaler_y": proc_p.scaler_y}
        proc_a, d_a = make_proc(fold, c, (1, 12), scalers=scalers)
        A = predict_all_seeds(fold, c, proc_a, d_a, seeds)
        alls.append(A)

        # verify the transplant on the dates the two scopes share
        mp = P.groupby(["date", "seed"]).p.mean()
        ma = A.groupby(["date", "seed"]).p.mean()
        j = pd.concat([mp.rename("pub"), ma.rename("all")], axis=1).dropna()
        dev = float(np.abs(j.pub - j["all"]).max()) if len(j) else np.nan
        print(f"  {fold['test']}: published n={len(d_p['dates_test'])}, "
              f"all-months n={len(d_a['dates_test'])}, shared {len(j)//len(seeds)} dates, "
              f"max|pred diff| = {dev:.4f}")
        if not np.isnan(dev) and dev > 0.25:
            raise SystemExit(f"ABORT: scaler transplant does not reproduce the published "
                             f"predictions at {fold['test']} (max dev {dev:.3f}). "
                             f"The all-months scope would not be comparable.")

    frames = {"published": pd.concat(pubs, ignore_index=True),
              "allmonths": pd.concat(alls, ignore_index=True)}

    recs = []
    for scope, F in frames.items():
        F = F.copy()
        F["month"] = F.date.dt.month
        F["year"] = F.date.dt.year
        F["season"] = season_of(F.month.values)
        for (site, yr, s), g in F.groupby(["site", "year", "season"]):
            per_seed = g.groupby("seed").apply(
                lambda h: pd.Series({"offset": (h.y - h.p).mean(), "r2": r2(h.y.values, h.p.values)}))
            obs = g.groupby("date").y.first()
            recs.append(dict(scope=scope, site=site, year=yr, season=s,
                             n_days=g.date.nunique(), obs_sd=float(obs.std()),
                             offset=per_seed.offset.mean(), offset_sd=per_seed.offset.std(ddof=1),
                             r2=per_seed.r2.mean(), r2_sd=per_seed.r2.std(ddof=1)))
    S = pd.DataFrame(recs)
    S["low_sd"] = S.obs_sd < LOW_SD
    S.to_csv(OUT / "partA_offset_by_season.csv", index=False)

    C = coupling_by_season()
    C.to_csv(OUT / "partA_coupling_by_season.csv", index=False)
    report(S, frames, C)

    rows = [{"variant": "offset_by_season", "experiment": "season_split",
             "test_station": r.site, "year": int(r.year),
             "season": f"{r.scope}_{r.season}", "r2": r.r2, "rmse": np.nan, "mae": np.nan,
             "bias": r.offset, "correlation": np.nan, "nse": np.nan, "n_samples": int(r.n_days)}
            for r in S.itertuples()]
    vm.append_rows(rows)
    print("\nrows -> summary_metrics.csv as variant='offset_by_season', experiment='season_split'")


def dist(v, label, width=30):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        print(f"  {label:<{width}} (no data)")
        return
    q1, q3 = np.percentile(v, [25, 75])
    print(f"  {label:<{width}} n={len(v):2d}  mean={v.mean():+7.3f}  median={np.median(v):+7.3f}  "
          f"IQR=[{q1:+.3f},{q3:+.3f}]  min={v.min():+7.3f}  max={v.max():+7.3f}")


def report(S, frames, C):
    for scope in ["published", "allmonths"]:
        T = S[S.scope == scope]
        print("\n" + "=" * 104)
        lbl = ("PUBLISHED SCOPE  (Apr-Oct only; off-season = Apr+May)" if scope == "published"
               else "ALL-MONTHS SCOPE  (off-season = Nov-May; Nov-Mar is OUT OF DISTRIBUTION "
                    "for these models)")
        print(f"SCOPE: {lbl}")
        print("=" * 104)
        print(f"{'site':5} {'year':5} {'season':7} {'ndays':6} {'obs_sd':>7} "
              f"{'offset':>18} {'R2':>18}")
        for r in T.sort_values(["site", "year", "season"]).itertuples():
            mark = " *" if r.low_sd else ""
            print(f"{r.site:5} {r.year:5} {r.season:7} {r.n_days:6} {r.obs_sd:7.2f} "
                  f"{r.offset:+9.3f}+/-{r.offset_sd:6.3f} {r.r2:+9.3f}+/-{r.r2_sd:6.3f}{mark}")
        print("  * observed SD < 2.0: R2 denominator untrustworthy")

        print(f"\n  OFFSET distribution across years, per site x season [{scope}]")
        for site in sorted(T.site.unique()):
            for s in ["in", "off"]:
                dist(T[(T.site == site) & (T.season == s)].offset.values, f"{site} {s}-season offset")
        print(f"\n  |OFFSET| distribution (magnitude, which is what the transfer error costs)")
        for site in sorted(T.site.unique()):
            for s in ["in", "off"]:
                dist(np.abs(T[(T.site == site) & (T.season == s)].offset.values),
                     f"{site} {s}-season |offset|")
        print(f"\n  R2 distribution across years, per site x season [{scope}]")
        for site in sorted(T.site.unique()):
            for s in ["in", "off"]:
                sub = T[(T.site == site) & (T.season == s)]
                dist(sub[~sub.low_sd].r2.values, f"{site} {s}-season R2 (trustworthy yrs)")

        print(f"\n  POOLED-BY-SEASON R2 (all days of that season, per site, per seed) [{scope}]")
        F = frames[scope].copy()
        F["season"] = season_of(F.date.dt.month.values)
        print(f"  {'site':5} {'season':7} {'ndays':6} {'obs_sd':>7} {'R2 (mean+/-sd over seeds)':>28}")
        for (site, s), g in F.groupby(["site", "season"]):
            v = g.groupby("seed").apply(lambda h: r2(h.y.values, h.p.values))
            sd = g.groupby("date").y.first().std()
            print(f"  {site:5} {s:7} {g.date.nunique():6} {sd:7.2f} "
                  f"{v.mean():+13.3f}+/-{v.std(ddof=1):.3f}")
        for s in ["in", "off"]:
            g = F[F.season == s]
            v = g.groupby("seed").apply(lambda h: r2(h.y.values, h.p.values))
            print(f"  {'ALL':5} {s:7} {g.date.nunique():6} "
                  f"{g.groupby('date').y.first().std():7.2f} "
                  f"{v.mean():+13.3f}+/-{v.std(ddof=1):.3f}")

    print("\n" + "=" * 104)
    print("MODEL-FREE CHECK OF MAKKI'S CLAIM: observed SSM->RZSM correlation by season")
    print("(station CSVs, all 12 months, no model involved)")
    print("=" * 104)
    print(f"  {'site':5} {'season':7} {'n_yrs':>5} {'mean r':>8} {'median r':>9} {'IQR':>20} "
          f"{'min':>7} {'max':>7}")
    for site in sorted(C.site.unique()):
        for s in ["in", "off"]:
            v = C[(C.site == site) & (C.season == s)].r.values
            if not len(v):
                continue
            q1, q3 = np.percentile(v, [25, 75])
            print(f"  {site:5} {s:7} {len(v):5d} {v.mean():+8.3f} {np.median(v):+9.3f} "
                  f"[{q1:+.3f},{q3:+.3f}]".ljust(0) + f" {v.min():+7.3f} {v.max():+7.3f}")
    print("\n  per site-year, in minus off:")
    piv = C.pivot_table(index=["site", "year"], columns="season", values="r")
    piv["in_minus_off"] = piv["in"] - piv["off"]
    print("   " + piv.round(3).to_string().replace("\n", "\n   "))
    v = piv["in_minus_off"].dropna().values
    print(f"\n   in-season tighter in {int((v>0).sum())}/{len(v)} site-years, "
          f"median difference {np.median(v):+.3f}")


if __name__ == "__main__":
    main()
