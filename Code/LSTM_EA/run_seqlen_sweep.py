"""
Sequence-length sweep, NESTED (not selected on test performance).

For each (fold, seed, seq_length) this records TWO scores:
  val_r2   on the validation split of the TRAINING sites only (val_years=2022).
           This is the only thing selection is ever allowed to see.
  test_r2  on the held-out site, May-Oct 2017-2023.  Reported, never selected on.

Arm 1 (nested) and Arm 2 (full curve) are then both computed from this one table:
  Arm 1  per (fold, seed) pick argmax(val_r2) over seq_length, report its test_r2.
  Arm 2  report test_r2 at every seq_length, labelled descriptive.

A COMPARABILITY NOTE for Arm 2: longer windows produce fewer windows, and with
require_contiguous_windows=True they drop proportionally more near season edges.
So each seq_length is scored on a DIFFERENT set of days.  Per-date predictions
are saved so the curve can also be recomputed on the intersection of days common
to all lengths.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from run_variant import VARIANTS
from src.config import DataConfig, FeatureConfig, ModelConfig, TrainingConfig
from src.data_loader import DataProcessor, RZSMDataset
from src.models import get_model
from src.trainer import Trainer
from src.evaluator import Evaluator

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "model_variants" / "seqlen_sweep"
logging.disable(logging.INFO)


def cfg(seq):
    e = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig()
    c.dynamic_cols = e["dynamic_cols"]; c.static_cols = e["static_cols"]
    c.use_presto_static = e["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["paper_config_fixed"].items():
        setattr(c, k, v)
    c.seq_length = seq
    return c


def r2(y, p):
    ss = ((y - p) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / st) if st > 0 and len(y) > 1 else np.nan


def run(fold, c, seed):
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

    tr_ds = RZSMDataset(d["X_d_train"], d["X_s_train"], d["y_train"])
    te_ds = RZSMDataset(d["X_d_test"], d["X_s_test"], d["y_test"])
    yrs = pd.to_datetime(d["dates_train"]).year
    vy = c.val_years if isinstance(c.val_years, (list, tuple)) else (c.val_years,)
    vi = [i for i, y in enumerate(yrs) if y in set(vy)]
    ti = [i for i, y in enumerate(yrs) if y not in set(vy)]
    tl = DataLoader(Subset(tr_ds, ti), batch_size=c.batch_size, shuffle=True)
    vl = DataLoader(Subset(tr_ds, vi), batch_size=c.batch_size, shuffle=False)

    m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                              dropout=c.dropout, num_layers=c.num_layers),
                  d["X_d_train"].shape[2], d["X_s_train"].shape[2])
    t = Trainer(m, TrainingConfig(batch_size=c.batch_size, epochs=c.epochs,
                                  learning_rate=c.learning_rate,
                                  early_stopping_patience=c.early_stopping_patience,
                                  weight_decay=c.weight_decay),
                str(OUT / "_scratch"), device="cpu")
    t.train(tl, vl)
    ev = Evaluator(t.model, proc.scaler_y, str(BASE), device="cpu")

    # validation score — TRAINING sites only, the sole selection signal
    vp, vt = ev.predict(DataLoader(Subset(tr_ds, vi), batch_size=256, shuffle=False))
    val_r2 = r2(np.asarray(vt).ravel(), np.asarray(vp).ravel())
    # held-out score — reported only
    tp, tt = ev.predict(DataLoader(te_ds, batch_size=256, shuffle=False))
    tp, tt = np.asarray(tp).ravel(), np.asarray(tt).ravel()
    dts = pd.to_datetime(d["dates_test"])
    k = np.asarray((dts.month >= 5) & (dts.month <= 10) & (dts.year <= 2023))
    return val_r2, dts[k], tt[k], tp[k], len(ti), len(vi), int(k.sum())


ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, required=True)
ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,42")
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
c = cfg(a.seq)
seeds = [int(x) for x in a.seeds.split(",")]
print(f"seq_length={a.seq}  seeds={seeds}\n")

rows, preds = [], []
for seed in seeds:
    torch.manual_seed(seed); np.random.seed(seed)
    c.seed = seed
    for fold in CV_FOLDS:
        v, dts, yt, yp, ntr, nva, nte = run(fold, c, seed)
        site = fold["test"]
        rows.append(dict(seq=a.seq, site=site, seed=seed, val_r2=v,
                         test_r2=r2(yt, yp), n_train=ntr, n_val=nva, n_test=nte))
        preds.append(pd.DataFrame(dict(seq=a.seq, site=site, seed=seed,
                                       date=dts, y=yt, p=yp)))
        print(f"  seed{seed:<3} {site}  val_r2={v:+.4f}  test_r2={rows[-1]['test_r2']:+.4f} "
              f"n_test={nte}", flush=True)
pd.DataFrame(rows).to_csv(OUT / f"scores_seq{a.seq}.csv", index=False)
pd.concat(preds, ignore_index=True).to_csv(OUT / f"preds_seq{a.seq}.csv", index=False)
print(f"\ndone -> {OUT}/scores_seq{a.seq}.csv")
