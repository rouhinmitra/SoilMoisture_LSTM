"""
Predictive check: is the residual forecastable forward in time?

For each site-year: 14-day trailing-mean residual series r(t) under
paper_config_fixed (mean over 15 seeds), then corr(r(t), r(t+delta)) for
delta = 7/14/21/30/45 d, reported as a DISTRIBUTION over the 21 site-years.
Also the per-site-year horizon at which that correlation first drops below 0.5
- the probe-coast period, measured rather than inferred from tau.

CAVEAT built into the design: two 14-day trailing means separated by delta < 14
share (14 - delta) days of data, so delta=7 is inflated by construction.
delta >= 14 uses disjoint windows and is the honest part of the curve.
Read-only; no training.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sstats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_within_season_drivers import build_config, daily_residuals

BASE = Path(__file__).resolve().parent
WIN = 14
DELTAS = [7, 14, 21, 30, 45]
GRID = list(range(3, 61))


def series(M):
    """site-year -> date-indexed 14-day trailing mean residual."""
    out = {}
    for (s, y), g in M.groupby(["site", "year"]):
        g = g.sort_values("date").set_index("date")
        full = g.resid.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        out[(s, y)] = full.rolling(f"{WIN}D", min_periods=max(5, WIN // 2)).mean()
    return out


def fwd_corr(sr, delta):
    a = sr.dropna()
    b = a.reindex(a.index + pd.Timedelta(days=delta))
    m = ~b.isna().values
    if m.sum() < 8:
        return np.nan, int(m.sum())
    x, y = a.values[m], b.values[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, int(m.sum())
    return float(sstats.pearsonr(x, y)[0]), int(m.sum())


def main():
    c = build_config()
    M = daily_residuals(c)
    M["year"] = M.date.dt.year
    M = M[M.year.between(2017, 2023)]
    S = series(M)
    print(f"site-years: {len(S)}   trailing window: {WIN} d\n")

    rows = []
    for k, sr in S.items():
        for d in DELTAS:
            r, n = fwd_corr(sr, d)
            rows.append(dict(site=k[0], year=k[1], delta=d, r=r, npairs=n))
    R = pd.DataFrame(rows)

    print("=== forward correlation of the 14-day-mean residual, across 21 site-years ===")
    print(f"{'delta':>6} {'mean':>7} {'median':>7} {'q25':>7} {'q75':>7} {'frac>0.5':>9} {'n_sy':>5}  note")
    for d in DELTAS:
        g = R[(R.delta == d)].r.dropna()
        note = "OVERLAPPING windows - inflated" if d < WIN else "disjoint windows"
        print(f"{d:6d} {g.mean():7.3f} {g.median():7.3f} {np.percentile(g,25):7.3f} "
              f"{np.percentile(g,75):7.3f} {(g>0.5).mean():9.2f} {len(g):5d}  {note}")

    # ---- per-site-year crossing horizon ----
    print("\n=== per-site-year horizon where forward corr first drops below 0.5 ===")
    cross = []
    for k, sr in S.items():
        prev_d, prev_r, hit = None, None, None
        for d in GRID:
            r, n = fwd_corr(sr, d)
            if np.isnan(r):
                continue
            if r < 0.5:
                if prev_r is not None and prev_r >= 0.5 and prev_r != r:
                    hit = prev_d + (prev_r - 0.5) * (d - prev_d) / (prev_r - r)
                else:
                    hit = d
                break
            prev_d, prev_r = d, r
        cross.append(dict(site=k[0], year=k[1], horizon=hit if hit is not None else np.nan))
    C = pd.DataFrame(cross)
    piv = C.pivot(index="year", columns="site", values="horizon")
    print(piv.round(1).to_string())
    h = C.horizon
    never = int(h.isna().sum())
    print(f"\nhorizon: median {h.median():.1f} d | mean {h.mean():.1f} d | "
          f"IQR [{np.nanpercentile(h,25):.1f}, {np.nanpercentile(h,75):.1f}] | "
          f"min {h.min():.1f} | max {h.max():.1f} | never dropped below 0.5: {never}/21")
    print(f"site-years with horizon <= 14 d: {(h<=14).sum()}/21 | "
          f">= 30 d: {(h>=30).sum()}/21 | >= 45 d: {(h>=45).sum()}/21")
    C.to_csv(BASE/"outputs"/"model_variants"/"residual_persistence_horizon.csv", index=False)
    R.to_csv(BASE/"outputs"/"model_variants"/"residual_forward_corr.csv", index=False)


if __name__ == "__main__":
    main()
