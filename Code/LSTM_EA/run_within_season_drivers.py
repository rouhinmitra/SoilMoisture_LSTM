"""
EXPERIMENT 3 - within-season offset drivers.

Exp 2 tested season-MEAN offsets against season-TOTAL drivers (n=21) and found
nothing.  Exp 1 then showed the offset drifts WITHIN a season, so Exp 2's
resolution could not have detected a driver of that drift.  This tests it.

PRE-SPECIFIED: 6 drivers x 3 lags (0/7/14 d) = 18 tests, Holm at alpha=0.05
=> most significant must clear p <= 0.05/18 = 0.00278.

AUTOCORRELATION GUARD (option a): non-overlapping 14-day block means, so no
point enters twice.  Significance is computed across the 21 site-years (each
contributes ONE correlation, Fisher-z, one-sample t-test), never across daily
points.  The residual autocorrelation scale is estimated and reported.
A circular block permutation (option b) confirms anything that clears Holm.
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
BLOCK = 14
LAGS = [0, 7, 14]
DRIVERS = ["precip", "et", "swi60", "ssm", "vpd", "days_since_rain"]
RAIN_MM = 5.0          # "significant" rain threshold
NTESTS = len(DRIVERS) * len(LAGS)


def build_config():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig(); c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    return c


def daily_residuals(c):
    """mean over seeds of (observed - predicted), per site per date."""
    seeds = sorted(int(d.name[4:]) for d in CKPT.glob("seed*") if d.is_dir())
    frames = []
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
        acc = []
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
            acc.append(np.asarray(yt).ravel() - np.asarray(yp).ravel())
        frames.append(pd.DataFrame({"site": fold["test"], "date": dates,
                                    "resid": np.mean(acc, axis=0)}))
    R = pd.concat(frames, ignore_index=True)
    return R.groupby(["site", "date"], as_index=False).resid.mean()


def daily_drivers():
    out = []
    for site, fn in FILES.items():
        df = pd.read_csv(DATA / fn)
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df["Year"] = df.date.dt.year
        df = _add_swi_api_channels(df, (60,), logging.getLogger())
        p_col = next((x for x in PRECIP_COLS if x in df.columns), None)
        p = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0)
        ta, rh = df["TA_1_1_1"], df["RH_1_1_1"]
        es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
        # days since last daily precip >= RAIN_MM
        dsr, last = np.empty(len(df)), np.nan
        for i, v in enumerate(p.values):
            if v >= RAIN_MM:
                last = i
            dsr[i] = np.nan if np.isnan(last) else i - last
        out.append(pd.DataFrame({"site": site, "date": df.date, "precip": p.values,
                                 "et": (df["LE_1_1_1"] * 0.0864 / 2.45).values,
                                 "swi60": df["swi_60"].values, "ssm": df["SSM_avg"].values,
                                 "vpd": (es * (1 - rh / 100.0)).values,
                                 "days_since_rain": dsr}))
    return pd.concat(out, ignore_index=True)


def ac_scale(x):
    """lag-1 autocorrelation and e-folding time in days."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan
    x = x - x.mean()
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 else np.nan
    tau = -1.0 / np.log(r1) if (r1 is not np.nan and 0 < r1 < 1) else np.nan
    return r1, tau


def main():
    c = build_config()
    print(f"PRE-SPECIFIED: {len(DRIVERS)} drivers x {len(LAGS)} lags = {NTESTS} tests")
    print(f"Holm at alpha=0.05 -> most significant must clear p <= {0.05/NTESTS:.5f}")
    print(f"Guard: non-overlapping {BLOCK}-day block means; significance across "
          f"n=21 site-years (Fisher-z, one-sample t), never across daily points.\n")

    res = daily_residuals(c)
    drv = daily_drivers()
    M = res.merge(drv, on=["site", "date"], how="left")
    M["year"] = M.date.dt.year
    M = M[M.year.between(2017, 2023)].sort_values(["site", "date"])

    # --- autocorrelation scale of the daily residual, per site-year ---
    taus, r1s = [], []
    for (s, y), g in M.groupby(["site", "year"]):
        r1, tau = ac_scale(g.resid.values)
        r1s.append(r1); taus.append(tau)
    r1s = np.array(r1s, float); taus = np.array(taus, float)
    print(f"daily residual autocorrelation: lag-1 r = {np.nanmean(r1s):.3f} "
          f"(range {np.nanmin(r1s):.3f}-{np.nanmax(r1s):.3f})")
    print(f"e-folding scale tau = {np.nanmean(taus):.1f} d "
          f"(median {np.nanmedian(taus):.1f}, max {np.nanmax(taus):.1f})")
    print(f"=> a {BLOCK}-day non-overlapping block is {BLOCK/np.nanmean(taus):.1f} tau wide\n")

    # --- non-overlapping block means, per site-year ---
    rows = []
    for (s, y), g in M.groupby(["site", "year"]):
        g = g.sort_values("date").copy()
        day = (g.date - g.date.iloc[0]).dt.days.values
        g["block"] = day // BLOCK
        for lag in LAGS:
            gl = g.copy()
            if lag:
                lagged = drv[drv.site == s].set_index("date")[DRIVERS]
                gl = gl.join(lagged.reindex(gl.date - pd.Timedelta(days=lag)).set_index(gl.index),
                             rsuffix="_lag")
                for d_ in DRIVERS:
                    gl[d_] = gl[f"{d_}_lag"]
            agg = gl.groupby("block").agg({"resid": "mean", **{d_: "mean" for d_ in DRIVERS}})
            agg = agg.dropna()
            if len(agg) < 6:
                continue
            for d_ in DRIVERS:
                if agg[d_].std() == 0:
                    continue
                r, _ = sstats.pearsonr(agg[d_], agg.resid)
                rows.append(dict(site=s, year=y, driver=d_, lag=lag, r=r, nblocks=len(agg)))
    B = pd.DataFrame(rows)
    print(f"per-site-year correlations computed on {B.nblocks.mean():.1f} blocks on average "
          f"(min {B.nblocks.min()}, max {B.nblocks.max()})\n")

    # --- across-site-year test on Fisher-z ---
    out = []
    for d_ in DRIVERS:
        for lag in LAGS:
            g = B[(B.driver == d_) & (B.lag == lag)]
            z = np.arctanh(np.clip(g.r.values, -0.999, 0.999))
            t, p = sstats.ttest_1samp(z, 0.0)
            out.append(dict(driver=d_, lag=lag, n_site_years=len(g),
                            mean_r=np.tanh(z.mean()), median_r=float(np.median(g.r)),
                            q25=float(np.percentile(g.r, 25)), q75=float(np.percentile(g.r, 75)),
                            frac_pos=float((g.r > 0).mean()), t=float(t), p=float(p)))
    O = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    O["holm_alpha"] = [0.05 / (NTESTS - i) for i in range(len(O))]
    surv, still = [], True
    for _, r_ in O.iterrows():
        still = still and r_.p <= r_.holm_alpha
        surv.append(still)
    O["survives_holm"] = surv

    print("=== 18 pre-specified tests, across n=21 site-years (non-overlapping blocks) ===")
    print(O.round(4).to_string(index=False))
    print("\nSurvives Holm at alpha=0.05:",
          ", ".join(f"{r_.driver}@lag{r_.lag}" for _, r_ in O[O.survives_holm].iterrows()) or "NOTHING")

    B.to_csv(BASE / "outputs" / "model_variants" / "within_season_driver_corrs.csv", index=False)
    print("\nper-site-year correlations ->", BASE / "outputs/model_variants/within_season_driver_corrs.csv")


if __name__ == "__main__":
    main()
