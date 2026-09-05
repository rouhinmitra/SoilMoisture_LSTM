"""
REPLICATE-LEVEL SENSOR SPREAD.  Read-only, raw AmeriFlux files.

Everything to date used SSM_avg / RZSM_25_avg, which ARE the replicate means -
so within-field heterogeneity has never been looked at, and part of what we call
model error may be measurement ambiguity in the target.

COLUMN NAMING, ESTABLISHED EMPIRICALLY, NOT ASSUMED.  AmeriFlux SWC_PI_F_H_V_R:
H = horizontal replicate, V = depth level, R = replicate-of-replicate.  The repo
README glosses SWC_PI_F_2_1_1 as "depth level 2"; that is wrong.  Verified by
reconstructing the averaged columns from the raw probes:
    SSM_avg      = mean_H SWC_PI_F_H_1_1   (MAE 0.0196)
    RZSM_25_avg  = mean_H SWC_PI_F_H_2_1   (MAE 0.0057)
    RZSM_50_avg  = mean_H SWC_PI_F_H_3_1   (MAE 0.139)
    RZSM_100_avg = mean_H SWC_PI_F_H_4_1   (MAE 0.0008)
so V=1 surface, V=2 25cm, V=3 50cm, V=4 100cm, and H indexes the replicates.

REPLICATE COUNT DIFFERS BY SITE - this is a confound for question 3:
    Ne1  H = 1,2,3   (three probes)
    Ne2  H = 2,3     (two probes)
    Ne3  H = 2,3     (two probes)
An SD over 2 probes is a far noisier estimate than over 3, so the cross-site
comparison is ALSO run on the matched H={2,3} pair available at all three sites.

Spread is computed on EXACTLY the days paper_config_fixed scored for that
site-year, so spread and RMSE are strictly comparable.

Rows -> outputs/model_variants/replicate_spread/*.csv
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "replicate_spread"
DATA = Path("/Users/rouhinmitra/SM_work/Data/Base")
RESID = BASE / "outputs" / "model_variants" / "spinup_checks" / "daily_residuals_by_seed.csv"
FILES = {"Ne1": "ne1_1_maize.csv", "Ne2": "ne2_1_maize.csv", "Ne3": "ne3_1_maize.csv"}
DEPTH_NAME = {1: "SSM_surface", 2: "RZSM_25cm", 3: "RZSM_50cm", 4: "RZSM_100cm"}
YEARS = list(range(2017, 2024))


def discover(site):
    """Which H (replicate) and V (depth) indices actually carry data at this site."""
    df = pd.read_csv(DATA / FILES[site])
    df["date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates("date")
    df = df[(df.date.dt.year >= YEARS[0]) & (df.date.dt.year <= YEARS[-1])]
    found = {}
    for c in df.columns:
        if not c.startswith("SWC_PI_F_"):
            continue
        try:
            h, v, r = (int(x) for x in c[len("SWC_PI_F_"):].split("_"))
        except ValueError:
            continue
        if df[c].notna().sum() > 200:
            found.setdefault(v, []).append((h, c))
    return df, {v: sorted(hs) for v, hs in sorted(found.items())}


def spread_frame(df, cols, dates):
    """SD and range across replicates, restricted to the given dates."""
    g = df[df.date.isin(dates)][["date"] + cols].set_index("date")
    n_ok = g.notna().sum(axis=1)
    g = g[n_ok >= 2]
    if not len(g):
        return None
    return pd.DataFrame({"sd": g.std(axis=1, ddof=1), "rng": g.max(axis=1) - g.min(axis=1),
                         "mean": g.mean(axis=1), "n_probes": g.notna().sum(axis=1)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("REPLICATE-LEVEL SENSOR SPREAD  (read-only)\n")

    # model error / offset per site-year, on the days the model actually scored
    F = pd.read_csv(RESID, parse_dates=["date"])
    F = F[F.year.isin(YEARS)]
    perf = (F.groupby(["site", "year", "seed"])
              .apply(lambda h: pd.Series({"rmse": float(np.sqrt(((h.y - h.p) ** 2).mean())),
                                          "offset": float((h.y - h.p).mean())}))
              .groupby(level=[0, 1]).mean().reset_index())
    scored = {(s, y): set(g.date.unique()) for (s, y), g in F.groupby(["site", "year"])}

    print("=" * 104)
    print("AVAILABLE REPLICATES AND DEPTHS PER SITE (discovered, > 200 valid days 2017-2023)")
    print("=" * 104)
    raw, avail = {}, {}
    for site in FILES:
        df, found = discover(site)
        raw[site], avail[site] = df, found
        print(f"  {site}:")
        for v, hs in found.items():
            print(f"     depth V={v} ({DEPTH_NAME.get(v,'?'):12}) : {len(hs)} replicates  "
                  + ", ".join(c for _, c in hs))

    rows = []
    for site in FILES:
        df = raw[site]
        for v in (1, 2):                       # surface and 25 cm
            hs = avail[site].get(v, [])
            if len(hs) < 2:
                continue
            allcols = [c for _, c in hs]
            matched = [c for h, c in hs if h in (2, 3)]
            for yr in YEARS:
                dts = scored.get((site, yr))
                if not dts:
                    continue
                for tag, cols in (("all_probes", allcols), ("matched_H23", matched)):
                    if len(cols) < 2:
                        continue
                    sp = spread_frame(df, cols, dts)
                    if sp is None or len(sp) < 30:
                        continue
                    p = perf[(perf.site == site) & (perf.year == yr)]
                    rows.append(dict(site=site, year=yr, depth=v, var=DEPTH_NAME[v],
                                     probe_set=tag, n_probes=len(cols), n_days=len(sp),
                                     mean_sd=sp.sd.mean(), median_sd=sp.sd.median(),
                                     mean_range=sp.rng.mean(), max_range=sp.rng.max(),
                                     mean_level=sp["mean"].mean(),
                                     rmse=float(p.rmse.iloc[0]) if len(p) else np.nan,
                                     offset=float(p.offset.iloc[0]) if len(p) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "replicate_spread_by_siteyear.csv", index=False)

    A = S[S.probe_set == "all_probes"]

    # ---- Q1 ----
    print("\n" + "=" * 104)
    print("Q1  SEASON-MEAN SPREAD ACROSS REPLICATES, per site-year (on the scored days)")
    print("=" * 104)
    for v in (1, 2):
        print(f"\n  --- {DEPTH_NAME[v]} ---")
        print(f"  {'site':5} {'year':5} {'probes':6} {'ndays':6} {'mean SD':>9} {'median SD':>10} "
              f"{'mean range':>11} {'max range':>10} {'level':>8}")
        for r in A[A.depth == v].sort_values(["site", "year"]).itertuples():
            print(f"  {r.site:5} {r.year:5} {r.n_probes:6} {r.n_days:6} {r.mean_sd:9.3f} "
                  f"{r.median_sd:10.3f} {r.mean_range:11.3f} {r.max_range:10.3f} {r.mean_level:8.2f}")

    # ---- Q2 ----
    print("\n" + "=" * 104)
    print("Q2  SPREAD vs MODEL ERROR (paper_config_fixed RMSE, same site-year, same days)")
    print("    ratio >= 1 => the model is at or below the measurement noise floor")
    print("=" * 104)
    for v in (1, 2):
        sub = A[A.depth == v]
        if v == 2:
            print(f"\n  --- {DEPTH_NAME[v]} (this is the TARGET the model predicts) ---")
        else:
            print(f"\n  --- {DEPTH_NAME[v]} (an input, not the target - context only) ---")
            print("    (model RMSE is for RZSM_25, so the ratio is not meaningful here;"
                  " spread shown for comparison)")
        print(f"  {'site':5} {'year':5} {'mean SD':>9} {'mean range':>11} {'RMSE':>8} "
              f"{'SD/RMSE':>9} {'range/RMSE':>11}")
        for r in sub.sort_values(["site", "year"]).itertuples():
            print(f"  {r.site:5} {r.year:5} {r.mean_sd:9.3f} {r.mean_range:11.3f} "
                  f"{r.rmse:8.3f} {r.mean_sd/r.rmse:9.3f} {r.mean_range/r.rmse:11.3f}")
        if v == 2:
            k = sub.dropna(subset=["rmse"])
            print(f"\n    site-years with SD    >= RMSE: {int((k.mean_sd>=k.rmse).sum())}/{len(k)}")
            print(f"    site-years with RANGE >= RMSE: {int((k.mean_range>=k.rmse).sum())}/{len(k)}")
            print(f"    mean SD/RMSE = {(k.mean_sd/k.rmse).mean():.3f}   "
                  f"mean range/RMSE = {(k.mean_range/k.rmse).mean():.3f}")

    # ---- Q3 ----
    print("\n" + "=" * 104)
    print("Q3  DOES SPREAD VARY BY SITE?  (Ne1 is the worst-performing site)")
    print("=" * 104)
    for tag in ("all_probes", "matched_H23"):
        print(f"\n  --- probe set: {tag} ---")
        for v in (1, 2):
            sub = S[(S.depth == v) & (S.probe_set == tag)]
            if not len(sub):
                continue
            print(f"    {DEPTH_NAME[v]}:")
            for site, g in sub.groupby("site"):
                print(f"      {site}  n_yr={len(g)}  probes={int(g.n_probes.iloc[0])}  "
                      f"mean SD={g.mean_sd.mean():.3f} +/- {g.mean_sd.std(ddof=1):.3f}   "
                      f"mean range={g.mean_range.mean():.3f}   "
                      f"mean RMSE={g.rmse.mean():.3f}")
            grp = [g.mean_sd.values for _, g in sub.groupby("site")]
            if len(grp) == 3:
                f, p = sst.f_oneway(*grp)
                k, pk = sst.kruskal(*grp)
                print(f"      across sites: ANOVA F={f:.2f} p={p:.4f} | Kruskal p={pk:.4f}")

    # ---- Q4 ----
    print("\n" + "=" * 104)
    print("Q4  DOES SPREAD VARY BY YEAR WITHIN A SITE?  (variable-rate irrigation signature)")
    print("=" * 104)
    for v in (1, 2):
        print(f"\n  {DEPTH_NAME[v]}:")
        for site, g in A[A.depth == v].groupby("site"):
            g = g.sort_values("year")
            cv = g.mean_sd.std(ddof=1) / g.mean_sd.mean()
            print(f"    {site}  " + " ".join(f"{int(r.year)}:{r.mean_sd:.2f}" for r in g.itertuples())
                  + f"   |  min={g.mean_sd.min():.2f} max={g.mean_sd.max():.2f} "
                    f"ratio={g.mean_sd.max()/g.mean_sd.min():.2f} CV={cv:.2f}")
    # irrigation context
    print("\n  seasonal irrigation total per site-year (column 'I', Apr-Oct), for context:")
    for site in FILES:
        df = raw[site]
        if "I" not in df.columns:
            print(f"    {site}: no 'I' column")
            continue
        d = df[df.date.dt.month.between(4, 10)]
        tot = d.groupby(d.date.dt.year)["I"].sum()
        print(f"    {site}  " + " ".join(f"{y}:{v:6.1f}" for y, v in tot.items() if y in YEARS))

    # ---- Q5 ----
    print("\n" + "=" * 104)
    print("Q5  SITE-YEAR REPLICATE SPREAD vs SITE-YEAR OFFSET  (n=21; one test per depth)")
    print("=" * 104)
    for v in (1, 2):
        sub = A[A.depth == v].dropna(subset=["offset"])
        x, y = sub.mean_sd.values, sub.offset.values
        pr, pp = sst.pearsonr(x, y)
        sr, sp = sst.spearmanr(x, y)
        print(f"\n  PRE-SPECIFIED - {DEPTH_NAME[v]} spread vs SIGNED offset   n={len(x)}")
        print(f"    Pearson r={pr:+.3f} (p={pp:.4f})   Spearman rho={sr:+.3f} (p={sp:.4f})")
        pr2, pp2 = sst.pearsonr(x, np.abs(y))
        sr2, sp2 = sst.spearmanr(x, np.abs(y))
        print(f"  secondary (magnitude, not the pre-specified test): vs |offset|")
        print(f"    Pearson r={pr2:+.3f} (p={pp2:.4f})   Spearman rho={sr2:+.3f} (p={sp2:.4f})")
        pr3, pp3 = sst.pearsonr(x, sub.rmse.values)
        print(f"  also: spread vs RMSE   Pearson r={pr3:+.3f} (p={pp3:.4f})")
    print(f"\nfiles -> {OUT}")


if __name__ == "__main__":
    main()
