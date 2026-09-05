"""
WITHIN-SEASON REPLICATION ACROSS ADAPTATION YEARS, WITH AN OFFSET-ONLY CONTROL.

The ceiling run found a large within-season gain, but at ONE adaptation year
(2017) - the same year that was a wet outlier in every cross-year run.  This
repeats it for every adaptation year Y in 2017-2022 and adds the control that
decides what the gain actually is.

For each Y and site: adapt on Apr-Jul of Y, score Aug-Oct of Y, 15 seeds.
Strict windows only - all seq_length input days after the Jul 31 cutoff, so no
scored window shares a day with the adaptation block.

Three arms, identical windows, identical seeds, identical scored days:
  zero_shot      - paper_config_fixed_sensitivity checkpoint, no adaptation
  offset_only    - zero_shot + a single scalar b = mean(y - yhat) fit on the
                   Apr-Jul block.  One parameter, no retraining, fully causal.
  joint_retrain  - full retrain from scratch on the two source sites + the
                   Apr-Jul block, exactly as run_probe_joint_retrain does.

THE DECOMPOSITION IS THE POINT.  offset_only vs joint_retrain separates level
from shape:
  offset_only ~= joint_retrain  -> the probe buys a within-season bias
                                   correction and nothing more.
  joint_retrain >> offset_only  -> the probe is teaching coupling/shape.

PREPROCESSING IS IMPORTED, NOT REIMPLEMENTED.  build_config / prep_fold /
load_zero_shot / predict come from run_probe_joint_retrain so this run is
comparable to the ceiling run by construction: scalers fit on the two source
sites only, the block enters as training samples, model selection stays
source-side (validation = val_years from the two SOURCE sites; when Y is itself
a source validation year the two never collide, because they are different
sites).

R2 decomposition recorded per arm as mechanistic colour (the three-arm
comparison, not this split, is the decisive result):
    R2          = 1 - (bias^2 + Var(e)) / Var(y)
    r2_centered = 1 -            Var(e) / Var(y)   <- ORACLE debias, uses the
                                                      eval-window mean
    bias_frac   =     bias^2            / Var(y)
so R2 = r2_centered - bias_frac.  offset_only is the causal counterpart of
r2_centered: the gap between them is how well the Apr-Jul offset predicts the
Aug-Oct offset.

Rows -> outputs/model_variants/within_season_replication/raw.csv
     -> summary_metrics.csv, experiment='within_season_replication',
        variant='withinseason_<arm>_Y<year>'.
"""
import argparse, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cv import CV_FOLDS
from src.config import ModelConfig, TrainingConfig
from src.data_loader import RZSMDataset
from src.models import get_model
from src.trainer import Trainer
import variant_metrics as vm

# Identical preprocessing to the ceiling run, by import rather than by copy.
from run_probe_joint_retrain import build_config, prep_fold, r2, load_zero_shot, predict

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "within_season_replication"

ADAPT_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
CUT_MONTH = 7            # adapt on months <= 7 (Apr-Jul), score months >= 8
LOW_SD = 2.0             # eval-window SD below this = untrustworthy R2 denominator


def season_masks(year, dates, seq_length):
    """Apr-Jul adaptation block and the STRICT Aug-Oct evaluation window."""
    yr, mo = dates.year.values, dates.month.values
    block = (yr == year) & (mo <= CUT_MONTH)
    cutoff = pd.Timestamp(year=year, month=CUT_MONTH, day=31)
    win_start = dates - pd.Timedelta(days=seq_length - 1)
    ev = (yr == year) & (mo > CUT_MONTH) & np.asarray(win_start > cutoff)
    return block, ev


def decompose(y, p):
    """R2 split into an oracle-debiased term and a bias penalty."""
    vy = float(np.var(y))
    if vy <= 0 or len(y) < 2:
        return dict(r2=np.nan, r2_centered=np.nan, bias_frac=np.nan,
                    bias=np.nan, corr=np.nan)
    e = y - p
    bias = float(np.mean(e))
    return dict(r2=r2(y, p),
                r2_centered=float(1 - np.var(e) / vy),
                bias_frac=float(bias ** 2 / vy),
                bias=bias,
                corr=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--years", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    c = build_config()
    from run_probe_joint_retrain import CKPT
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else
             sorted(int(d.name[4:]) for d in CKPT.glob("seed*") if d.is_dir()))
    years = [int(x) for x in a.years.split(",")] if a.years else ADAPT_YEARS
    folds = [f for f in CV_FOLDS if (not a.folds or f["test"] in a.folds.split(","))]

    out_dir = Path(str(OUT) + a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.csv"
    scratch = out_dir / "_scratch"

    n_runs = len(folds) * len(years) * len(seeds)
    print(f"within-season replication: {len(folds)} sites x {len(years)} adaptation years "
          f"x {len(seeds)} seeds = {n_runs} full trainings")
    print(f"years: {years}\nseeds: {seeds}\n")

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

        vy_cfg = c.val_years if isinstance(c.val_years, (list, tuple)) else (c.val_years,)
        src_years = dates_tr.year.values
        src_val_i = np.where(np.isin(src_years, list(vy_cfg)))[0]
        src_tr_i = np.where(~np.isin(src_years, list(vy_cfg)))[0]

        xd_te_t = torch.tensor(Xd_te, dtype=torch.float32)
        xs_te_t = torch.tensor(Xs_te, dtype=torch.float32)
        dyn_dim, stat_dim = Xd_tr.shape[2], Xs_tr.shape[2]

        for Y in years:
            block, ev = season_masks(Y, dates_te, c.seq_length)
            if block.sum() < 50 or ev.sum() < 20:
                print(f"  {site}/Y{Y}: block n={int(block.sum())} eval n={int(ev.sum())} "
                      f"-> too small, SKIPPED")
                continue

            yv = y_true_te[ev]
            y_blk = y_true_te[block]
            ev_sd = float(np.std(yv))
            flag = "  <-- LOW SD, R2 denominator untrustworthy" if ev_sd < LOW_SD else ""
            print(f"  {site}/Y{Y}: block n={int(block.sum())} "
                  f"({dates_te[block].min().date()}..{dates_te[block].max().date()}) | "
                  f"eval n={int(ev.sum())} "
                  f"({dates_te[ev].min().date()}..{dates_te[ev].max().date()}) "
                  f"sd={ev_sd:.2f}{flag}")

            xd_blk = torch.tensor(Xd_te[block], dtype=torch.float32)
            xs_blk = torch.tensor(Xs_te[block], dtype=torch.float32)
            xd_ev = xd_te_t[ev]
            xs_ev = xs_te_t[ev]

            Xd_joint = np.concatenate([Xd_tr[src_tr_i], Xd_te[block]])
            Xs_joint = np.concatenate([Xs_tr[src_tr_i], Xs_te[block]])
            y_joint = np.concatenate([y_tr[src_tr_i], y_te[block]])
            train_ds = RZSMDataset(Xd_joint, Xs_joint, y_joint)
            val_ds = Subset(RZSMDataset(Xd_tr, Xs_tr, y_tr), src_val_i.tolist())
            frac = len(Xd_te[block]) / len(Xd_joint)

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                              dropout=c.dropout, num_layers=c.num_layers),
                                  dyn_dim, stat_dim)
                tr = Trainer(model, TrainingConfig(batch_size=c.batch_size, epochs=c.epochs,
                                                   learning_rate=c.learning_rate,
                                                   early_stopping_patience=c.early_stopping_patience,
                                                   weight_decay=c.weight_decay),
                             str(scratch), device="cpu")
                tr.train(DataLoader(train_ds, batch_size=c.batch_size, shuffle=True),
                         DataLoader(val_ds, batch_size=c.batch_size, shuffle=False))
                jm = tr.model
                jm.eval()

                zs = load_zero_shot(fold, seed, c, dyn_dim, stat_dim)
                if zs is None:
                    print(f"      seed{seed}: no zero-shot checkpoint, skipped")
                    continue

                p_zs = predict(zs, xd_ev, xs_ev, sy)
                # one scalar, fit on the block only
                b = float(np.mean(y_blk - predict(zs, xd_blk, xs_blk, sy)))
                p_off = p_zs + b
                p_jt = predict(jm, xd_ev, xs_ev, sy)
                block_fit = r2(y_blk, predict(jm, xd_blk, xs_blk, sy))

                for arm, pv in (("zero_shot", p_zs), ("offset_only", p_off),
                                ("joint_retrain", p_jt)):
                    recs.append(dict(site=site, adapt_year=Y, arm=arm, seed=seed,
                                     n_eval=int(ev.sum()), n_block=int(block.sum()),
                                     eval_sd=ev_sd, low_sd=ev_sd < LOW_SD,
                                     offset_b=b, probe_frac=frac, block_fit=block_fit,
                                     best_epoch=tr.best_epoch, **decompose(yv, pv)))

                done += 1
                el = time.time() - t0
                pd.DataFrame(recs).to_csv(raw_path, index=False)
                print(f"      [{done:3d}/{n_runs}] {site}/Y{Y}/seed{seed} "
                      f"zs={r2(yv,p_zs):+.3f} off={r2(yv,p_off):+.3f} jt={r2(yv,p_jt):+.3f} "
                      f"b={b:+.2f} | {el/60:.1f} min, ~{el/done*(n_runs-done)/60:.0f} min left",
                      flush=True)

    R = pd.DataFrame(recs)
    R.to_csv(raw_path, index=False)
    print(f"\nraw -> {raw_path}")
    report(R)

    rows = []
    for (site, Y, arm), g in R.groupby(["site", "adapt_year", "arm"]):
        rows.append({"variant": f"withinseason_{arm}_Y{Y}",
                     "experiment": "within_season_replication",
                     "test_station": site, "year": int(Y), "season": "aug_oct",
                     "r2": float(np.nanmean(g.r2)), "rmse": np.nan, "mae": np.nan,
                     "bias": float(np.nanmean(g.bias)), "correlation": float(np.nanmean(g["corr"])),
                     "nse": np.nan, "n_samples": int(np.nanmean(g.n_eval))})
    vm.append_rows(rows)
    print("rows -> summary_metrics.csv as variant='withinseason_*', "
          "experiment='within_season_replication'")


def _paired(R, site, Y, a1, a2):
    g = R[(R.site == site) & (R.adapt_year == Y)]
    x = g[g.arm == a1].set_index("seed").r2
    y = g[g.arm == a2].set_index("seed").r2
    s = sorted(set(x.index) & set(y.index))
    return (x.loc[s] - y.loc[s]).values if s else np.array([])


def report(R):
    if not len(R):
        return
    sites = sorted(R.site.unique())
    years = sorted(R.adapt_year.unique())

    print("\n" + "=" * 108)
    print("PER YEAR x SITE  (paired over seeds; delta = mean of per-seed differences)")
    print("=" * 108)
    hdr = (f"{'site':5} {'Y':5} {'n':4} {'sd':>5} {'zs':>16} {'offset':>16} {'joint':>16} "
           f"{'d(j-zs)':>9} {'d(o-zs)':>9} {'d(j-o)':>9}")
    print(hdr)
    for site in sites:
        for Y in years:
            g = R[(R.site == site) & (R.adapt_year == Y)]
            if not len(g):
                continue
            cell = {}
            for arm in ("zero_shot", "offset_only", "joint_retrain"):
                v = g[g.arm == arm].r2
                cell[arm] = f"{v.mean():7.3f}+/-{v.std(ddof=1):.3f}"
            djz = _paired(R, site, Y, "joint_retrain", "zero_shot")
            doz = _paired(R, site, Y, "offset_only", "zero_shot")
            djo = _paired(R, site, Y, "joint_retrain", "offset_only")
            sd = g.eval_sd.iloc[0]
            mark = " *" if sd < LOW_SD else ""
            print(f"{site:5} {Y:5} {int(g.n_eval.iloc[0]):4} {sd:5.2f} "
                  f"{cell['zero_shot']:>16} {cell['offset_only']:>16} {cell['joint_retrain']:>16} "
                  f"{djz.mean():+9.3f} {doz.mean():+9.3f} {djo.mean():+9.3f}{mark}")
    print("  * eval-window SD < 2.0: R2 denominator untrustworthy, excluded from the verdict")

    print("\n" + "=" * 108)
    print("DISTRIBUTION ACROSS ADAPTATION YEARS (never pooled across years)")
    print("=" * 108)
    for label, a1, a2 in (("joint - zero_shot", "joint_retrain", "zero_shot"),
                          ("offset - zero_shot", "offset_only", "zero_shot"),
                          ("joint - offset  (SHAPE vs LEVEL)", "joint_retrain", "offset_only")):
        print(f"\n{label}")
        print(f"  {'site':5} {'n_yrs':>5} {'mean':>8} {'median':>8} {'IQR':>18} "
              f"{'#pos/6':>8} {'trustworthy yrs only':>28}")
        for site in sites:
            per_year, trust = [], []
            for Y in years:
                dd = _paired(R, site, Y, a1, a2)
                if not len(dd):
                    continue
                per_year.append(dd.mean())
                if not R[(R.site == site) & (R.adapt_year == Y)].low_sd.iloc[0]:
                    trust.append(dd.mean())
            if not per_year:
                continue
            v = np.array(per_year)
            q1, q3 = np.percentile(v, [25, 75])
            t = np.array(trust)
            ts = (f"med={np.median(t):+.3f} #pos={int((t>0).sum())}/{len(t)}"
                  if len(t) else "none")
            print(f"  {site:5} {len(v):5d} {v.mean():+8.3f} {np.median(v):+8.3f} "
                  f"[{q1:+.3f},{q3:+.3f}]".ljust(0) + f"{'':2}{int((v>0).sum())}/{len(v):<6} {ts:>28}")

    print("\n" + "=" * 108)
    print("ABSOLUTE joint_retrain R2 per year x site  (a delta only counts where this is "
          "meaningfully positive)")
    print("=" * 108)
    print(f"  {'site':5} " + " ".join(f"{Y:>14}" for Y in years))
    for site in sites:
        cells = []
        for Y in years:
            g = R[(R.site == site) & (R.adapt_year == Y) & (R.arm == "joint_retrain")]
            if not len(g):
                cells.append("-")
                continue
            mark = "*" if g.low_sd.iloc[0] else " "
            cells.append(f"{g.r2.mean():+6.3f}+/-{g.r2.std(ddof=1):.2f}{mark}")
        print(f"  {site:5} " + " ".join(f"{x:>14}" for x in cells))

    print("\n" + "=" * 108)
    print("R2 DECOMPOSITION  (R2 = r2_centered - bias_frac).  Mechanistic colour only.")
    print("=" * 108)
    print(f"  {'site':5} {'Y':5} " + " ".join(f"{a:>34}" for a in
          ["zero_shot  r2 / cent / biasfrac", "joint      r2 / cent / biasfrac"]))
    for site in sites:
        for Y in years:
            g = R[(R.site == site) & (R.adapt_year == Y)]
            if not len(g):
                continue
            out = []
            for arm in ("zero_shot", "joint_retrain"):
                h = g[g.arm == arm]
                out.append(f"{h.r2.mean():+7.3f} /{h.r2_centered.mean():+7.3f} /"
                           f"{h.bias_frac.mean():7.3f}")
            print(f"  {site:5} {Y:5} " + " ".join(f"{x:>34}" for x in out))


if __name__ == "__main__":
    main()
