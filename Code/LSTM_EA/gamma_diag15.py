"""
D2 diagnostic, 15 seeds.

Q1: does per-seed fold1 out-of-range rate predict that seed's Ne3 R²?
    (extrapolation mechanism, seed-dependent)  -> correlation + scatter
Q2: do z / gamma cluster by ZERO-PADDING fraction rather than by site?
    D2's 60-day context is clipped at the calendar-year boundary and left-padded,
    so early-season windows carry mostly padding.  If z tracks padding, this whole
    diagnostic measures padding, not site context.
Read-only apart from the scatter PNG it writes.
"""
import logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sensitivity as rs
from run_cv import CVConfig, CV_FOLDS, STATIONS
from src.config import DataConfig, FeatureConfig, ModelConfig
from src.data_loader import DataProcessor
from src.models import get_model
from run_variant import VARIANTS
import plot_style as ps
import matplotlib.pyplot as plt
from scipy import stats as sstats

logging.disable(logging.INFO)
BASE = Path(__file__).resolve().parent
ROOT = BASE / "outputs" / "model_variants" / "d2_context_film_sensitivity"
SUMMARY = BASE / "outputs" / "model_variants" / "summary_metrics.csv"


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
    return DataProcessor(dc, fc).prepare_data(dc.train_files, dc.test_files, val_years=c.val_years)


def pad_fraction(X):
    """Share of context timesteps that are all-zero, i.e. left-padding."""
    return (np.abs(X).sum(axis=2) == 0).mean(axis=1)


def oor(vals_te, vals_tr):
    lo, hi = vals_tr.min(0).values, vals_tr.max(0).values
    return ((vals_te < lo) | (vals_te > hi)).float().mean().item()


def main():
    c = build_config()
    seeds = sorted(int(d.name[4:]) for d in ROOT.glob("seed*") if d.is_dir())
    r2 = pd.read_csv(SUMMARY)
    r2 = r2[(r2.season == "may_oct") & (r2.year.astype(str) == "ALL")
            & (r2.variant.str.startswith("d2_context_film_seed"))]
    r2["seed"] = r2.variant.str.split("seed").str[1].astype(int)

    rows, pad_rows = [], []
    for fold in CV_FOLDS:
        data = fold_data(fold, c)
        Xd_tr = torch.tensor(data["X_d_train"], dtype=torch.float32)
        Xd_te = torch.tensor(data["X_d_test"], dtype=torch.float32)
        pad_tr, pad_te = pad_fraction(data["X_d_train"]), pad_fraction(data["X_d_test"])
        dyn, stat = Xd_tr.shape[2], data["X_s_train"].shape[2]
        if fold is CV_FOLDS[0]:
            print(f"padding: train windows mean pad_frac={pad_tr.mean():.3f} "
                  f"(>0 in {(pad_tr>0).mean()*100:.1f}%), "
                  f"test mean={pad_te.mean():.3f} (>0 in {(pad_te>0).mean()*100:.1f}%)")
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
            rows.append(dict(fold=fold["name"], site=fold["test"], seed=seed,
                             z_oor=oor(z_te, z_tr), g_oor=oor(g_te, g_tr),
                             above99=float((nz_te > np.percentile(nz_tr, 99)).mean()),
                             zratio=float(nz_te.mean() / nz_tr.mean())))
            if seed == seeds[0]:
                for split, pf, nz in (("train", pad_tr, nz_tr), ("test", pad_te, nz_te)):
                    for lo, hi in [(0, .001), (.001, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
                        msk = (pf >= lo) & (pf < hi)
                        if msk.sum() > 5:
                            pad_rows.append(dict(fold=fold["name"], split=split,
                                                 bin=f"[{lo:.2f},{hi:.2f})", n=int(msk.sum()),
                                                 z_mean=float(nz[msk].mean())))
                # correlation of ||z|| with padding, within split
                for split, pf, nz in (("train", pad_tr, nz_tr), ("test", pad_te, nz_te)):
                    if pf.std() > 0:
                        r, p = sstats.pearsonr(pf, nz)
                        print(f"  {fold['name']:20} {split:5} corr(pad_frac, ||z||) = {r:+.3f} (p={p:.1e})")

    df = pd.DataFrame(rows)
    print("\n=== per-seed OOR by fold ===")
    print(df.pivot_table(index="seed", columns="site", values="z_oor").round(4).to_string())

    print("\n=== padding breakdown: mean ||z|| by pad-fraction bin (first seed) ===")
    print(pd.DataFrame(pad_rows).to_string(index=False))

    # Q1: fold1 OOR vs that seed's Ne3 R2
    f1 = df[df.site == "Ne3"].merge(
        r2[r2.test_station == "Ne3"][["seed", "r2"]], on="seed")
    print("\n=== Q1: fold1 (Ne3) OOR vs Ne3 R2 ===")
    print(f1[["seed", "z_oor", "g_oor", "above99", "zratio", "r2"]].round(4).to_string(index=False))
    for col in ["z_oor", "g_oor", "above99", "zratio"]:
        r, p = sstats.pearsonr(f1[col], f1["r2"])
        rs_, ps_ = sstats.spearmanr(f1[col], f1["r2"])
        print(f"  {col:8} vs R2:  pearson {r:+.3f} (p={p:.3f})   spearman {rs_:+.3f} (p={ps_:.3f})   n={len(f1)}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(100 * f1.z_oor, f1.r2, s=45, color=ps.MODEL_COLORS.get("LSTM", "#1f77b4"),
               edgecolor="white", zorder=3)
    for _, r_ in f1.iterrows():
        ax.annotate(int(r_.seed), (100 * r_.z_oor, r_.r2), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    if len(f1) > 2:
        b = np.polyfit(100 * f1.z_oor, f1.r2, 1)
        xs = np.linspace((100 * f1.z_oor).min(), (100 * f1.z_oor).max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="0.4", lw=1, ls="--", zorder=2)
    ps.style_axes(ax, xlabel="z out-of-range at Ne3 (% of test values)",
                  ylabel="Ne3 May–Oct R²", title="D2: encoder extrapolation vs skill (15 seeds)")
    print("\nscatter ->", ps.save_figure(fig, ROOT / "oor_vs_r2_Ne3.png"))


if __name__ == "__main__":
    main()
