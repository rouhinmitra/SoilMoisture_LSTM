"""
CROP-SIMILARITY ANALYSIS with permutation robustness over unknown years.

Read-only, observational: nothing here touches a model.  The question is not
whether a crop feature helps prediction, it is whether same-crop site-years
resemble each other in SSM->RZSM profile SHAPE after site and year are
controlled.

WETNESS IS DIVIDED OUT BEFORE ANYTHING ELSE.  Within each site-year, RZSM and
SSM are z-scored over the in-season window, so mean level AND amplitude are
gone from every descriptor.  Level is the offset; it is confounded with
year-wetness, and leaving it in would let wet years masquerade as same-crop.

Descriptors (per site-year, Jun-Oct, on the z-scored series):
  slope     OLS coefficient of zRZSM on zSSM
  curv      quadratic coefficient of zRZSM on zSSM + zSSM^2
  lag       argmax over 0..10 d of corr(zSSM[t-k], zRZSM[t])
  peakcorr  the correlation at that lag
  recess    mean 7-day recession slope of zRZSM after each wetting peak
  c3,c2,c1  cubic trajectory of zRZSM vs normalised day-of-season (intercept dropped)

Descriptors are z-scored across site-years so each weighs equally, then pairwise
Euclidean distance gives the similarity matrix.

CROP LABELS.  Known (fixed, not altered):
  Ne1  maize 2016-2021, soy 2022
  Ne2  2016-2020 = soy, maize, soy, maize, maize
  Ne3  2016-2020 = soy, maize, soy, maize, soy
  Unknown: Ne1 2023, Ne2 2021-2023, Ne3 2021-2023  (7 unknown site-years)

A NOTE ON THE ALTERNATION CONSTRAINT.  Ne2 is maize in BOTH 2019 and 2020, so
the known record already violates strict soy/maize alternation.  Applying
alternation to the unknown years would therefore impose a rule the data
contradicts, and it collapses the enumeration to exactly ONE labeling - no
distribution, no robustness check.  Both are reported: the constrained
single labeling for completeness, and the unconstrained 2^7 = 128 labelings as
the actual robustness result.

THE SWITCH TEST OVERRIDES THE PERMUTATIONS.  Its core - does Ne1-2022 (soy)
move away from Ne1's own maize years - uses only KNOWN labels and does not
depend on any labeling of the unknown years.  A favourable permutation in the
grouped test cannot overturn a null switch test.

Rows -> outputs/model_variants/crop_similarity/*.csv
"""
import itertools, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "crop_similarity"
DATA = Path("/Users/rouhinmitra/SM_work/Data/Base")
FILES = {"Ne1": "ne1_1_maize.csv", "Ne2": "ne2_1_maize.csv", "Ne3": "ne3_1_maize.csv"}

YEARS = list(range(2016, 2024))
IN_MONTHS = [6, 7, 8, 9, 10]
MIN_DAYS = 90            # below this a site-year's "shape" is noise
MAX_LAG = 10
RNG = np.random.default_rng(0)

KNOWN = {}
for y in range(2016, 2022):
    KNOWN[("Ne1", y)] = "maize"
KNOWN[("Ne1", 2022)] = "soy"
for y, c in zip(range(2016, 2021), ["soy", "maize", "soy", "maize", "maize"]):
    KNOWN[("Ne2", y)] = c
for y, c in zip(range(2016, 2021), ["soy", "maize", "soy", "maize", "soy"]):
    KNOWN[("Ne3", y)] = c
UNKNOWN = [("Ne1", 2023)] + [("Ne2", y) for y in (2021, 2022, 2023)] \
                          + [("Ne3", y) for y in (2021, 2022, 2023)]

DESCR = ["slope", "curv", "lag", "peakcorr", "recess", "c3", "c2", "c1"]


def z(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * np.nan


def profile(g):
    """Shape descriptors for one site-year.  Level and amplitude are already gone."""
    g = g.sort_values("date")
    zr, zs = z(g.RZSM_25_avg.values), z(g.SSM.values)
    ok = ~(np.isnan(zr) | np.isnan(zs))
    zr, zs = zr[ok], zs[ok]
    n = len(zr)
    if n < MIN_DAYS:
        return None
    slope = np.polyfit(zs, zr, 1)[0]
    curv = np.polyfit(zs, zr, 2)[0]

    best_lag, best_r = 0, -np.inf
    for k in range(0, MAX_LAG + 1):
        a, b = (zs[:-k], zr[k:]) if k else (zs, zr)
        if len(a) < 30:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r, best_lag = r, k

    pk, _ = find_peaks(zr, prominence=0.5)
    slopes = []
    for p in pk:
        seg = zr[p + 1:p + 8]
        if len(seg) >= 5:
            slopes.append(np.polyfit(np.arange(len(seg)), seg, 1)[0])
    recess = float(np.mean(slopes)) if slopes else np.nan

    t = np.linspace(0, 1, n)
    c3, c2, c1, _ = np.polyfit(t, zr, 3)
    return dict(n_days=n, slope=slope, curv=curv, lag=best_lag, peakcorr=best_r,
                recess=recess, c3=c3, c2=c2, c1=c1, n_peaks=len(pk))


def build_profiles():
    rows = []
    for site, fn in FILES.items():
        df = pd.read_csv(DATA / fn)
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("date")
        df = df[df.date.dt.month.isin(IN_MONTHS)]
        for y in YEARS:
            g = df[df.date.dt.year == y]
            if not len(g):
                continue
            p = profile(g)
            if p is None:
                print(f"  DROPPED {site} {y}: only "
                      f"{int((g.SSM.notna()&g.RZSM_25_avg.notna()).sum())} clean in-season days")
                continue
            p.update(site=site, year=y, rzsm_sd=float(g.RZSM_25_avg.std()))
            rows.append(p)
    return pd.DataFrame(rows)


def distance_matrix(P):
    X = P[DESCR].copy()
    for c in DESCR:                      # equal weight per descriptor
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = X.fillna(0.0)
    V = X.values
    n = len(V)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = np.sqrt(((V - V[i]) ** 2).sum(axis=1))
    keys = list(zip(P.site, P.year))
    return D, {k: i for i, k in enumerate(keys)}, keys


def grouped_stats(D, idx, labels, restrict=None):
    """Same-crop vs different-crop pair distances.  restrict: 'cross_site'|'same_year'|None."""
    same, diff = [], []
    ks = [k for k in labels if k in idx]
    for a, b in itertools.combinations(ks, 2):
        if restrict == "cross_site" and a[0] == b[0]:
            continue
        if restrict == "same_year" and a[1] != b[1]:
            continue
        d = D[idx[a], idx[b]]
        (same if labels[a] == labels[b] else diff).append(d)
    if len(same) < 2 or len(diff) < 2:
        return dict(n_same=len(same), n_diff=len(diff), mean_same=np.nan,
                    mean_diff=np.nan, delta=np.nan, d=np.nan)
    s, f = np.array(same), np.array(diff)
    sp = np.sqrt(((len(s) - 1) * s.var(ddof=1) + (len(f) - 1) * f.var(ddof=1))
                 / (len(s) + len(f) - 2))
    return dict(n_same=len(s), n_diff=len(f), mean_same=s.mean(), mean_diff=f.mean(),
                delta=f.mean() - s.mean(), d=(f.mean() - s.mean()) / sp if sp > 0 else np.nan)


def label_shuffle_p(D, idx, labels, restrict=None, nperm=5000):
    obs = grouped_stats(D, idx, labels, restrict)["d"]
    if np.isnan(obs):
        return np.nan, obs
    ks = [k for k in labels if k in idx]
    vals = [labels[k] for k in ks]
    cnt = 0
    for _ in range(nperm):
        perm = RNG.permutation(vals)
        lab = {k: v for k, v in zip(ks, perm)}
        s = grouped_stats(D, idx, lab, restrict)["d"]
        if not np.isnan(s) and s >= obs:
            cnt += 1
    return (cnt + 1) / (nperm + 1), obs


def switch_test(D, idx, labeling=None):
    """Ne1 2016-2021 maize -> 2022 soy.  Site constant, crop changes.  n=1 event."""
    ne1_maize = [("Ne1", y) for y in range(2016, 2022) if ("Ne1", y) in idx]
    tgt = ("Ne1", 2022)
    if tgt not in idx or len(ne1_maize) < 3:
        return None
    d_to_hist = np.array([D[idx[tgt], idx[k]] for k in ne1_maize])
    within = np.array([D[idx[a], idx[b]] for a, b in itertools.combinations(ne1_maize, 2)])
    out = dict(mean_to_maize_history=d_to_hist.mean(),
               within_maize_mean=within.mean(), within_maize_sd=within.std(ddof=1),
               z_vs_maize_cloud=(d_to_hist.mean() - within.mean()) / within.std(ddof=1),
               d_to_Ne2_2022=D[idx[tgt], idx[("Ne2", 2022)]] if ("Ne2", 2022) in idx else np.nan,
               d_to_Ne3_2022=D[idx[tgt], idx[("Ne3", 2022)]] if ("Ne3", 2022) in idx else np.nan)
    if labeling is not None:
        soy22 = [k for k in [("Ne2", 2022), ("Ne3", 2022)]
                 if k in idx and labeling.get(k) == "soy"]
        mz22 = [k for k in [("Ne2", 2022), ("Ne3", 2022)]
                if k in idx and labeling.get(k) == "maize"]
        out["mean_to_sameyear_soy"] = (np.mean([D[idx[tgt], idx[k]] for k in soy22])
                                       if soy22 else np.nan)
        out["mean_to_sameyear_maize"] = (np.mean([D[idx[tgt], idx[k]] for k in mz22])
                                         if mz22 else np.nan)
        # positive => 2022 sits closer to same-year soy than to its own maize history
        out["toward_soy"] = out["mean_to_maize_history"] - out["mean_to_sameyear_soy"]
    return out


def sameyear_test(D, idx, labeling):
    """Ne2 vs Ne3 within a year: is the gap bigger when their crops differ?"""
    disc, conc = [], []
    for y in YEARS:
        a, b = ("Ne2", y), ("Ne3", y)
        if a not in idx or b not in idx:
            continue
        ca, cb = labeling.get(a), labeling.get(b)
        if ca is None or cb is None:
            continue
        (disc if ca != cb else conc).append((y, D[idx[a], idx[b]]))
    md = np.mean([v for _, v in disc]) if disc else np.nan
    mc = np.mean([v for _, v in conc]) if conc else np.nan
    return dict(n_disc=len(disc), n_conc=len(conc), mean_disc=md, mean_conc=mc,
                delta=md - mc, disc_years=[y for y, _ in disc], conc_years=[y for y, _ in conc])


def enumerate_labelings(constrained):
    keys = [k for k in UNKNOWN]
    if constrained:
        # alternate forward from each site's last KNOWN year
        lab = {}
        for site in ["Ne1", "Ne2", "Ne3"]:
            ys = sorted(y for (s, y) in KNOWN if s == site)
            last_y = max(ys); cur = KNOWN[(site, last_y)]
            for y in range(last_y + 1, 2024):
                cur = "soy" if cur == "maize" else "maize"
                if (site, y) in keys:
                    lab[(site, y)] = cur
        return [lab]
    return [dict(zip(keys, combo))
            for combo in itertools.product(["maize", "soy"], repeat=len(keys))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("CROP-SIMILARITY ANALYSIS  (read-only; no model, no crop feature added)\n")
    print("Building shape profiles, Jun-Oct, z-scored within site-year...")
    P = build_profiles()
    D, idx, keys = distance_matrix(P)
    P.to_csv(OUT / "profiles.csv", index=False)

    print("\n" + "=" * 96)
    print("DATA ADEQUACY  (thin site-years distort the similarity matrix)")
    print("=" * 96)
    print(P[["site", "year", "n_days", "rzsm_sd", "n_peaks", "slope", "curv", "lag",
             "peakcorr", "recess"]].round(3).to_string(index=False))
    thin = P[(P.n_days < 120) | (P.rzsm_sd < 2.0)]
    print(f"\n  site-years with < 120 clean in-season days or raw RZSM SD < 2.0: "
          f"{len(thin)}" + ("" if not len(thin) else
          "  -> " + ", ".join(f"{r.site}-{r.year}" for r in thin.itertuples())))
    print(f"  minimum clean in-season days across all site-years: {int(P.n_days.min())} "
          f"({P.loc[P.n_days.idxmin(),'site']}-{int(P.loc[P.n_days.idxmin(),'year'])})")
    print(f"  minimum raw RZSM SD: {P.rzsm_sd.min():.2f}")

    # ---------------- TEST 2 : the switch, labeling-free core ----------------
    print("\n" + "=" * 96)
    print("TEST 2 - THE SWITCH TEST  (site held fixed; Ne1 maize 2016-2021 -> soy 2022)")
    print("  This core uses ONLY known labels and is independent of every permutation.")
    print("=" * 96)
    sw = switch_test(D, idx)
    print(f"  mean distance  Ne1-2022  ->  Ne1's own maize years : {sw['mean_to_maize_history']:.3f}")
    print(f"  mean distance  among Ne1 maize years themselves    : {sw['within_maize_mean']:.3f} "
          f"+/- {sw['within_maize_sd']:.3f}")
    print(f"  z of Ne1-2022 against its own maize cloud          : {sw['z_vs_maize_cloud']:+.3f}")
    print(f"  raw distance   Ne1-2022 -> Ne2-2022 (label-free)   : {sw['d_to_Ne2_2022']:.3f}")
    print(f"  raw distance   Ne1-2022 -> Ne3-2022 (label-free)   : {sw['d_to_Ne3_2022']:.3f}")
    per_year = {y: D[idx[("Ne1", 2022)], idx[("Ne1", y)]]
                for y in range(2016, 2022) if ("Ne1", y) in idx}
    print("  per-year distance from Ne1-2022 to each Ne1 maize year:")
    for y, v in per_year.items():
        print(f"      {y}: {v:.3f}")
    ranks = sorted(((D[idx[("Ne1", 2022)], idx[k]], k) for k in keys if k != ("Ne1", 2022)))
    print("  Ne1-2022's five nearest site-years overall:")
    for v, k in ranks[:5]:
        lab = KNOWN.get(k, "UNKNOWN")
        print(f"      {k[0]}-{k[1]:<5} {v:.3f}   ({lab})")

    # ---------------- TEST 1 & 3 on KNOWN labels only ----------------
    print("\n" + "=" * 96)
    print("TEST 1 - GROUPED SIMILARITY, KNOWN LABELS ONLY (descriptive; crop is confounded "
          "with site and year)")
    print("=" * 96)
    known = {k: v for k, v in KNOWN.items() if k in idx}
    for restrict, lbl in [(None, "all pairs"), ("cross_site", "cross-site pairs only"),
                          ("same_year", "same-year pairs only")]:
        g = grouped_stats(D, idx, known, restrict)
        p, obs = label_shuffle_p(D, idx, known, restrict)
        print(f"  {lbl:24} same n={g['n_same']:3d} mean={g['mean_same']:.3f} | "
              f"diff n={g['n_diff']:3d} mean={g['mean_diff']:.3f} | "
              f"delta={g['delta']:+.3f}  Cohen d={g['d']:+.3f}  "
              f"label-shuffle p={p:.3f}")
    print("  (positive delta / d = same-crop pairs are MORE similar, the crop hypothesis)")

    print("\n" + "=" * 96)
    print("TEST 3 - SAME-YEAR DIFFERENT-CROP, KNOWN LABELS ONLY (year held fixed)")
    print("=" * 96)
    sy = sameyear_test(D, idx, known)
    print(f"  discordant years {sy['disc_years']}: mean Ne2-Ne3 distance = {sy['mean_disc']:.3f}")
    print(f"  concordant years {sy['conc_years']}: mean Ne2-Ne3 distance = {sy['mean_conc']:.3f}")
    print(f"  delta (disc - conc) = {sy['delta']:+.3f}   "
          f"(positive = crop difference widens the gap, the crop hypothesis)")
    for y in YEARS:
        a, b = ("Ne2", y), ("Ne3", y)
        if a in idx and b in idx:
            ca, cb = known.get(a, "?"), known.get(b, "?")
            print(f"      {y}: Ne2={ca:<7} Ne3={cb:<7} distance={D[idx[a], idx[b]]:.3f}")

    # ---------------- PERMUTATION OVER UNKNOWN YEARS ----------------
    for constrained in (True, False):
        labs = enumerate_labelings(constrained)
        tag = "CONSTRAINED (alternation)" if constrained else "UNCONSTRAINED (2^7)"
        print("\n" + "=" * 96)
        print(f"PERMUTATION OVER THE {len(UNKNOWN)} UNKNOWN SITE-YEARS - {tag}: "
              f"{len(labs)} labeling(s)")
        if constrained:
            print("  WARNING: Ne2 is maize in BOTH 2019 and 2020, so the known record already")
            print("  violates strict alternation.  This constraint is imposed against the data")
            print("  and yields a single labeling - it provides NO robustness distribution.")
            print(f"  labeling: " + ", ".join(f"{k[0]}-{k[1]}={v}" for k, v in labs[0].items()))
        print("=" * 96)

        recs = []
        for lab in labs:
            full = dict(known); full.update(lab)
            g_all = grouped_stats(D, idx, full, None)
            g_cs = grouped_stats(D, idx, full, "cross_site")
            s3 = sameyear_test(D, idx, full)
            s2 = switch_test(D, idx, full)
            recs.append(dict(d_all=g_all["d"], d_crosssite=g_cs["d"],
                             sameyear_delta=s3["delta"], n_disc=s3["n_disc"],
                             toward_soy=s2.get("toward_soy", np.nan),
                             to_soy=s2.get("mean_to_sameyear_soy", np.nan),
                             **{f"{k[0]}{k[1]}": v for k, v in lab.items()}))
        R = pd.DataFrame(recs)
        R.to_csv(OUT / f"permutations_{'constrained' if constrained else 'unconstrained'}.csv",
                 index=False)

        if not constrained:
            print(f"  {'statistic':28} {'mean':>8} {'median':>8} {'IQR':>18} "
                  f"{'min':>8} {'max':>8} {'>0':>9}")
            for col, lbl in [("d_all", "Cohen d, all pairs"),
                             ("d_crosssite", "Cohen d, cross-site pairs"),
                             ("sameyear_delta", "same-year disc - conc"),
                             ("toward_soy", "switch: toward same-yr soy")]:
                v = R[col].dropna().values
                if not len(v):
                    continue
                q1, q3 = np.percentile(v, [25, 75])
                print(f"  {lbl:28} {v.mean():+8.3f} {np.median(v):+8.3f} "
                      f"[{q1:+.3f},{q3:+.3f}]".ljust(0)
                      + f" {v.min():+8.3f} {v.max():+8.3f} {int((v>0).sum()):4d}/{len(v)}")
            best = R.loc[R.d_all.idxmax()]
            print("\n  MOST-FAVOURABLE LABELING (best-fitting; this is NOT the answer, it is the")
            print("  upper envelope of what the free labels can manufacture):")
            print("    " + ", ".join(f"{c}={best[c]}" for c in R.columns if c.startswith("Ne")))
            print(f"    Cohen d all pairs = {best.d_all:+.3f}   cross-site = {best.d_crosssite:+.3f}"
                  f"   same-year delta = {best.sameyear_delta:+.3f}")
            worst = R.loc[R.d_all.idxmin()]
            print(f"    least-favourable labeling: Cohen d all pairs = {worst.d_all:+.3f}")
        else:
            r = R.iloc[0]
            print(f"  Cohen d all pairs={r.d_all:+.3f}  cross-site={r.d_crosssite:+.3f}  "
                  f"same-year delta={r.sameyear_delta:+.3f}  switch toward_soy={r.toward_soy:+.3f}")

    print(f"\nfiles -> {OUT}")


if __name__ == "__main__":
    main()
