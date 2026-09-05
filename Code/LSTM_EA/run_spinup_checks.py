"""
SPIN-UP GATE: two read-only checks on the existing paper_config_fixed outputs.

If the site-year offset is a spin-up artifact - the model enters April blind to
winter recharge and needs time to find the site's level - it leaves two
signatures.  Both are computed from the 15 published checkpoints; no training.

CHECK 1  Seasonal decay of the offset.
  Mean residual (observed - predicted) per site-year in successive HALF-MONTH
  bins, April through October.  Spin-up error => largest in Apr-May, shrinking
  through the season.  Structural offset => flat.
  Note: with seq_length=20, require_contiguous_windows=True and
  month_range=(4,10), the earliest scored day of a season is ~Apr 20, so the
  Apr 1-15 bin is empty by construction and the series effectively starts at
  Apr 16-30.  That is the earliest the published model can be observed at all.

CHECK 2  Offset vs season-start state.
  Each site-year's mean offset against the OBSERVED RZSM in the first 14 days
  of that season.  One pre-specified test, n=21, Pearson and Spearman.
  PRIMARY definition of season start: Apr 1-14 observed RZSM from the station
  CSV - the antecedent state the model enters blind, which exists whether or
  not a window can be scored there.  A secondary definition (the first 14
  SCORED days, ~Apr 20 - May 3) is reported as a robustness check, not as a
  second test.

Both null => the spin-up hypothesis is dead and the winter retrain is skipped.

Rows -> outputs/model_variants/spinup_checks/*.csv
"""
import logging, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cv import CV_FOLDS
import run_season_split_partA as A
from run_probe_joint_retrain import build_config, r2, CKPT

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "spinup_checks"
YEARS = list(range(2017, 2024))


def half_month_bin(dates):
    """'Apr-a' = days 1-15, 'Apr-b' = 16-end."""
    mon = dates.dt.strftime("%b")
    half = np.where(dates.dt.day <= 15, "a", "b")
    return pd.Series([f"{m}-{h}" for m, h in zip(mon, half)], index=dates.index)


BIN_ORDER = [f"{m}-{h}" for m in ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
             for h in ["a", "b"]]


def main():
    c = build_config()
    seeds = sorted(int(p.name[4:]) for p in CKPT.glob("seed*") if p.is_dir())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"spin-up gate: {len(seeds)} seeds, paper_config_fixed, read-only\n")

    frames = []
    for fold in CV_FOLDS:
        proc, d = A.make_proc(fold, c, (4, 10))
        frames.append(A.predict_all_seeds(fold, c, proc, d, seeds))
    F = pd.concat(frames, ignore_index=True)
    F["year"] = F.date.dt.year
    F = F[F.year.isin(YEARS)]
    F["resid"] = F.y - F.p
    F["bin"] = half_month_bin(F.date)
    F.to_csv(OUT / "daily_residuals_by_seed.csv", index=False)

    # ---- CHECK 1 -------------------------------------------------------
    B = (F.groupby(["site", "year", "bin"])
           .apply(lambda g: pd.Series({
               "resid": g.groupby("seed").resid.mean().mean(),
               "n_days": g.date.nunique()}))
           .reset_index())
    B = B[B.n_days >= 5]
    B["abs_resid"] = B.resid.abs()
    B["bi"] = B["bin"].map({b: i for i, b in enumerate(BIN_ORDER)})
    B = B.sort_values(["site", "year", "bi"])
    B.to_csv(OUT / "check1_offset_by_halfmonth.csv", index=False)

    print("=" * 92)
    print("CHECK 1  mean |residual| by half-month bin, pooled over the 21 site-years")
    print("=" * 92)
    print(f"{'bin':8} {'n_siteyrs':>9} {'mean|resid|':>12} {'median|resid|':>14} "
          f"{'mean signed':>12}")
    for b in BIN_ORDER:
        g = B[B["bin"] == b]
        if not len(g):
            print(f"{b:8} {'-':>9}  (no scored days - empty by construction)")
            continue
        print(f"{b:8} {len(g):9d} {g.abs_resid.mean():12.3f} {g.abs_resid.median():14.3f} "
              f"{g.resid.mean():+12.3f}")

    print("\nPer site-year: Spearman rho of |residual| vs bin index "
          "(negative = decaying, i.e. spin-up-like)")
    rows = []
    for (site, yr), g in B.groupby(["site", "year"]):
        if len(g) < 6:
            continue
        rho, p = sst.spearmanr(g.bi, g.abs_resid)
        early = g[g.bi <= 3].abs_resid.mean()      # Apr-b .. May-b
        late = g[g.bi >= 10].abs_resid.mean()      # Sep-a .. Oct-b
        strict = bool(np.all(np.diff(g.abs_resid.values) <= 0))
        rows.append(dict(site=site, year=yr, nbins=len(g), rho=rho, p=p,
                         early=early, late=late, early_minus_late=early - late,
                         strict_monotone=strict))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "check1_decay_per_siteyear.csv", index=False)
    print(D.round(3).to_string(index=False))

    z = np.arctanh(D.rho.clip(-.999, .999))
    t, pz = sst.ttest_1samp(z, 0)
    te, pe = sst.ttest_1samp(D.early_minus_late, 0)
    print(f"\n  site-years with NEGATIVE rho (decaying): {int((D.rho<0).sum())}/{len(D)}")
    print(f"  strictly monotone declining:             {int(D.strict_monotone.sum())}/{len(D)}")
    print(f"  mean rho = {D.rho.mean():+.3f}   Fisher-z one-sample t={t:+.2f}, p={pz:.3f}")
    print(f"  early (Apr-b..May-b) mean|resid| = {D.early.mean():.3f}")
    print(f"  late  (Sep-a..Oct-b) mean|resid| = {D.late.mean():.3f}")
    print(f"  paired early-late = {D.early_minus_late.mean():+.3f} "
          f"(positive = spin-up-like), t={te:+.2f}, p={pe:.3f}, "
          f"early larger in {int((D.early_minus_late>0).sum())}/{len(D)}")

    # ---- CHECK 2 -------------------------------------------------------
    off = (F.groupby(["site", "year"])
             .apply(lambda g: g.groupby("seed").resid.mean().mean())
             .rename("offset").reset_index())

    start_primary, start_scored = [], []
    for site, fn in A.FILES.items():
        raw = pd.read_csv(A.DATA / fn)
        raw["date"] = pd.to_datetime(raw["Date"])
        raw = raw.drop_duplicates("date")
        for yr in YEARS:
            w = raw[(raw.date >= f"{yr}-04-01") & (raw.date <= f"{yr}-04-14")]
            v = w["RZSM_25_avg"].dropna()
            start_primary.append(dict(site=site, year=yr,
                                      start_rzsm=float(v.mean()) if len(v) >= 7 else np.nan,
                                      n=len(v)))
    S1 = pd.DataFrame(start_primary)

    for (site, yr), g in F.groupby(["site", "year"]):
        dd = g.groupby("date").y.first().sort_index()
        start_scored.append(dict(site=site, year=yr,
                                 start_scored=float(dd.iloc[:14].mean()),
                                 first_day=str(dd.index[0].date())))
    S2 = pd.DataFrame(start_scored)

    C = off.merge(S1, on=["site", "year"]).merge(S2, on=["site", "year"])
    C.to_csv(OUT / "check2_offset_vs_season_start.csv", index=False)

    print("\n" + "=" * 92)
    print("CHECK 2  site-year offset vs observed RZSM at season start")
    print("=" * 92)
    print(C.round(3).to_string(index=False))

    def corrs(x, y, label):
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = np.asarray(x)[m], np.asarray(y)[m]
        pr, pp = sst.pearsonr(x, y)
        sr, sp = sst.spearmanr(x, y)
        print(f"  {label:38} n={len(x):2d}  Pearson r={pr:+.3f} (p={pp:.3f})   "
              f"Spearman rho={sr:+.3f} (p={sp:.3f})")

    print("\n  PRE-SPECIFIED TEST")
    corrs(C.start_rzsm.values, C.offset.values, "offset vs Apr 1-14 observed RZSM")
    print("\n  robustness (not a second test)")
    corrs(C.start_scored.values, C.offset.values, "offset vs first 14 SCORED days")
    corrs(C.start_rzsm.values, C.offset.abs().values, "|offset| vs Apr 1-14 observed RZSM")
    print(f"\n  first scored day per season: "
          f"{sorted(set(C.first_day.str[5:]))[:3]} ... "
          f"{sorted(set(C.first_day.str[5:]))[-1]}")
    print(f"\nfiles -> {OUT}")


if __name__ == "__main__":
    main()
