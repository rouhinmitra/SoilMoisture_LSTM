"""D2 gamma/z diagnostic: does the held-out site drive the encoder out of the
training distribution, or do the distributions overlap?  Read-only."""
import logging, sys, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor
from src.models import get_model
from run_variant import VARIANTS

logging.disable(logging.INFO)
log = logging.getLogger("diag")
BASE = Path(__file__).resolve().parent
ROOT = BASE / "outputs" / "model_variants" / "d2_context_film_sensitivity"
PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def build_config():
    exp = rs.build_experiments()["baseline_no_doy"]
    c = CVConfig()
    c.seq_length = 20
    c.dynamic_cols = exp["dynamic_cols"]; c.static_cols = exp["static_cols"]
    c.use_presto_static = exp["use_presto_static"]; c.add_temporal = False
    for k, v in VARIANTS["d2_context_film"].items():
        setattr(c, k, v)
    return c


def fold_data(fold, c):
    dc = DataConfig(
        train_files=[STATIONS[s] for s in fold["train"]],
        test_files=[STATIONS[fold["test"]]],
        data_dir=c.data_dir, seq_length=c.seq_length, nan_threshold=c.nan_threshold,
        same_year_constraint=True, month_range=(4, 10), year_range=c.year_range,
        require_contiguous_windows=c.require_contiguous_windows,
        impute_statics_after_scaling=c.impute_statics_after_scaling,
        swi_api_taus=c.swi_api_taus, context_length=c.context_length,
    )
    fc = FeatureConfig(dynamic_cols=list(c.dynamic_cols), static_cols=c.static_cols,
                       target_col=c.target_col, use_presto_static=c.use_presto_static,
                       presto_embeddings_path=c.presto_embeddings_path)
    p = DataProcessor(dc, fc)
    return p.prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)


def stats(a):
    return np.percentile(a, PCTS)


def main():
    c = build_config()
    seeds = sorted(int(d.name[4:]) for d in ROOT.glob("seed*") if d.is_dir())
    print(f"seeds with saved checkpoints: {seeds}\n")
    agg = {}
    for fold in CV_FOLDS:
        data = fold_data(fold, c)
        Xd_tr = torch.tensor(data["X_d_train"], dtype=torch.float32)
        Xd_te = torch.tensor(data["X_d_test"], dtype=torch.float32)
        dyn, stat = Xd_tr.shape[2], data["X_s_train"].shape[2]
        print("=" * 78)
        print(f"{fold['name']}   train={fold['train']} ({len(Xd_tr)} windows)  "
              f"test={fold['test']} ({len(Xd_te)} windows)")
        print("=" * 78)
        for seed in seeds:
            ck = ROOT / f"seed{seed}" / fold["name"] / "checkpoint_best.pt"
            if not ck.exists():
                continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            m = get_model(ModelConfig(model_type="LSTM", hidden_dim=c.hidden_dim,
                                      dropout=c.dropout, num_layers=c.num_layers,
                                      context_dim=c.context_dim,
                                      context_encoder_type=c.context_encoder_type), dyn, stat)
            m.load_state_dict(sd); m.eval()
            with torch.no_grad():
                z_tr, z_te = m.context(Xd_tr), m.context(Xd_te)
                g_tr = m.film(z_tr).chunk(2, -1)[0]
                g_te = m.film(z_te).chunk(2, -1)[0]
            nz_tr, nz_te = z_tr.norm(dim=1).numpy(), z_te.norm(dim=1).numpy()
            # per-sample gamma magnitude
            ng_tr, ng_te = g_tr.abs().max(1).values.numpy(), g_te.abs().max(1).values.numpy()
            # out-of-range fraction: test values outside the per-unit train min/max envelope
            lo, hi = g_tr.min(0).values, g_tr.max(0).values
            oor = ((g_te < lo) | (g_te > hi)).float().mean().item()
            zlo, zhi = z_tr.min(0).values, z_tr.max(0).values
            zoor = ((z_te < zlo) | (z_te > zhi)).float().mean().item()
            above99 = float((nz_te > np.percentile(nz_tr, 99)).mean())
            agg.setdefault(fold["name"], []).append((oor, zoor, above99,
                                                     nz_te.mean() / nz_tr.mean()))
            print(f"\n  seed {seed}")
            print(f"    ||z||      pct {PCTS}")
            print(f"      train    {np.round(stats(nz_tr), 3)}")
            print(f"      test     {np.round(stats(nz_te), 3)}")
            print(f"    max|gamma| per sample")
            print(f"      train    {np.round(stats(ng_tr), 4)}")
            print(f"      test     {np.round(stats(ng_te), 4)}")
            print(f"    z   out of per-unit train range: {100*zoor:5.2f}% of test values")
            print(f"    gam out of per-unit train range: {100*oor:5.2f}% of test values")
            print(f"    ||z|| test above train p99:      {100*above99:5.2f}% of test windows"
                  f"   (mean ratio test/train = {nz_te.mean()/nz_tr.mean():.3f})")
        print()
    print("=" * 78)
    print("SUMMARY (mean over seeds)")
    print(f"{'fold':20} {'gam OOR%':>9} {'z OOR%':>8} {'||z||>p99%':>11} {'||z|| ratio':>12}")
    for k, v in agg.items():
        a = np.array(v)
        print(f"{k:20} {100*a[:,0].mean():9.2f} {100*a[:,1].mean():8.2f} "
              f"{100*a[:,2].mean():11.2f} {a[:,3].mean():12.3f}")


if __name__ == "__main__":
    main()
