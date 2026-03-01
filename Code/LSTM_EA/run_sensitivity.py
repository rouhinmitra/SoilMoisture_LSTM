#!/usr/bin/env python
"""
Feature-group sensitivity (ablation) analysis for LSTM soil-moisture model.

Runs the full leave-one-station-out CV (same folds as run_cv.py) seven times:
  1. Baseline          – all features
  2. No SSM            – remove SSM, SSM_avg, SWC_PI_F_2_1_1, SWC_PI_F_3_1_1
  3. No Precip/Irrig   – remove P_PI_F_*, I, precip_jan_apr, precip_may_oct, irrigation
  4. No Meteorological – remove TA_1_1_1, RH_1_1_1
  5. No Sentinel-2     – remove ndvi, b11, b12, b2–b8a
  6. No Alpha Earth    – remove A00..A63
  7. No Presto         – remove emb_0..emb_127

Collects R², RMSE, NRMSE, MAE, Bias, PBIAS per fold and produces:
  - outputs/sensitivity/sensitivity_results.csv
  - outputs/sensitivity/sensitivity_bar_chart.png
  - Per-experiment scatter plots under outputs/sensitivity/<experiment>/
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from copy import deepcopy
import logging
import time

from run_cv import CVConfig, run_single_fold, CV_FOLDS, STATIONS, create_combined_scatter_plot
from utils.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Feature-group definitions
# ---------------------------------------------------------------------------

ALL_DYNAMIC = [
    'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1',
    'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I',
    'TA_1_1_1', 'RH_1_1_1',
    'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a',
]

SSM_FEATURES = {'SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1'}
PRECIP_FEATURES_DYN = {'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I'}
PRECIP_FEATURES_STAT = {'precip_jan_apr', 'precip_may_oct'}
METEO_FEATURES = {'TA_1_1_1', 'RH_1_1_1'}
S2_FEATURES = {'ndvi', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a'}
AE_FEATURES = {f'A{i:02d}' for i in range(64)}
PRESTO_FEATURES = {f'emb_{k}' for k in range(128)}


def _static_cols(use_presto: bool, exclude_ae: bool = False,
                 exclude_precip: bool = False, exclude_presto: bool = False):
    """Build the static_cols list given ablation flags."""
    cols: List[str] = []
    if use_presto and not exclude_presto:
        cols += [f'emb_{k}' for k in range(128)]
    if not exclude_ae:
        cols += [f'A{i:02d}' for i in range(64)]
    if not exclude_precip:
        cols += ['precip_jan_apr', 'precip_may_oct']
    return cols


def build_experiments() -> Dict[str, dict]:
    """Return an ordered dict of experiment name -> config overrides."""
    experiments = {}

    # 1. Baseline
    experiments['baseline'] = dict(
        description='All features (baseline)',
        dynamic_cols=list(ALL_DYNAMIC),
        static_cols=_static_cols(use_presto=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 2. No SSM
    experiments['no_ssm'] = dict(
        description='Remove SSM features',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES],
        static_cols=_static_cols(use_presto=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 3. No Precipitation / Irrigation
    experiments['no_precip_irrig'] = dict(
        description='Remove precipitation & irrigation features',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in PRECIP_FEATURES_DYN],
        static_cols=_static_cols(use_presto=True, exclude_precip=True),
        use_presto_static=True,
        exclude_irrigation_static=True,
    )

    # 4. No Meteorological
    experiments['no_meteo'] = dict(
        description='Remove meteorological features (TA, RH)',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in METEO_FEATURES],
        static_cols=_static_cols(use_presto=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 5. No Sentinel-2
    experiments['no_sentinel2'] = dict(
        description='Remove Sentinel-2 features',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in S2_FEATURES],
        static_cols=_static_cols(use_presto=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 6. No Alpha Earth embeddings
    experiments['no_alpha_earth'] = dict(
        description='Remove Alpha Earth embeddings (A00-A63)',
        dynamic_cols=list(ALL_DYNAMIC),
        static_cols=_static_cols(use_presto=True, exclude_ae=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 7. No Presto embeddings
    experiments['no_presto'] = dict(
        description='Remove Presto embeddings (emb_0-emb_127)',
        dynamic_cols=list(ALL_DYNAMIC),
        static_cols=_static_cols(use_presto=False),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 8. No SSM + No Sentinel-2 + No Presto (diagnostic for memorization)
    experiments['no_ssm_s2_presto'] = dict(
        description='Remove SSM + Sentinel-2 + Presto (memorization diagnostic)',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES | S2_FEATURES],
        static_cols=_static_cols(use_presto=False),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 9. No Presto + No Alpha Earth
    experiments['no_presto_ae'] = dict(
        description='Remove Presto + Alpha Earth embeddings',
        dynamic_cols=list(ALL_DYNAMIC),
        static_cols=_static_cols(use_presto=False, exclude_ae=True),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 10. No Presto + No Alpha Earth + No Sentinel-2
    experiments['no_presto_ae_s2'] = dict(
        description='Remove Presto + Alpha Earth + Sentinel-2',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in S2_FEATURES],
        static_cols=_static_cols(use_presto=False, exclude_ae=True),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 11. No SSM + No Presto + No Alpha Earth + No Sentinel-2
    experiments['no_ssm_presto_ae_s2'] = dict(
        description='Remove SSM + Presto + Alpha Earth + Sentinel-2',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES | S2_FEATURES],
        static_cols=_static_cols(use_presto=False, exclude_ae=True),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # --- Redundancy / moisture-pathway experiments ---

    # 12. No SSM + No Presto (keep S2, Alpha Earth, meteo, precip) — both moisture sources removed
    experiments['no_ssm_no_presto'] = dict(
        description='Remove SSM + Presto (keep S2, Alpha Earth, meteo, precip)',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES],
        static_cols=_static_cols(use_presto=False),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 13. No SSM + No Presto + No S2 (keep Alpha Earth) — three moisture pathways removed
    experiments['no_ssm_no_presto_no_s2'] = dict(
        description='Remove SSM + Presto + S2 (keep Alpha Earth, meteo, precip)',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES | S2_FEATURES],
        static_cols=_static_cols(use_presto=False),
        use_presto_static=False,
        exclude_irrigation_static=False,
    )

    # 14. No SSM + No S2 (keep Presto) — Presto alone when in-situ and optical moisture gone
    experiments['no_ssm_no_s2'] = dict(
        description='Remove SSM + Sentinel-2 (keep Presto, Alpha Earth, meteo, precip)',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES | S2_FEATURES],
        static_cols=_static_cols(use_presto=True),
        use_presto_static=True,
        exclude_irrigation_static=False,
    )

    # 15. Meteo + precip only (floor) — no SSM, S2, Presto, Alpha Earth, irrigation
    experiments['meteo_precip_only'] = dict(
        description='Only meteo + precip (physical drivers baseline / floor)',
        dynamic_cols=list(METEO_FEATURES | PRECIP_FEATURES_DYN),
        static_cols=['precip_jan_apr', 'precip_may_oct'],
        use_presto_static=False,
        exclude_irrigation_static=True,
    )

    # 16. No SSM + No precip/irrigation
    experiments['no_ssm_no_precip'] = dict(
        description='Remove SSM and all precip/irrigation features',
        dynamic_cols=[c for c in ALL_DYNAMIC if c not in SSM_FEATURES | PRECIP_FEATURES_DYN],
        static_cols=_static_cols(use_presto=True, exclude_precip=True),
        use_presto_static=True,
        exclude_irrigation_static=True,
    )

    return experiments


# ---------------------------------------------------------------------------
# Per-site prediction CSV
# ---------------------------------------------------------------------------

def save_fold_predictions(result: Dict, exp_name: str, data_dir: str, output_dir: Path):
    """Save a CSV with Date, SSM_avg_obs, RZSM_obs, RZSM_pred for the test site."""
    test_station = result['test_station']
    dates_test = result.get('dates_test')
    y_pred = result['y_pred']
    y_true = result['y_true']

    pred_df = pd.DataFrame({
        'RZSM_obs': y_true,
        'RZSM_pred': y_pred,
    })

    if dates_test is not None and len(dates_test) == len(y_true):
        pred_df.insert(0, 'Date', pd.to_datetime(dates_test))

        # Load raw CSV to get SSM_avg on matching dates
        raw_path = Path(data_dir) / STATIONS[test_station]
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path, parse_dates=['Date'])
            ssm_col = 'SSM_avg' if 'SSM_avg' in raw_df.columns else None
            if ssm_col:
                date_to_ssm = dict(zip(raw_df['Date'], raw_df[ssm_col]))
                pred_df['SSM_avg_obs'] = pred_df['Date'].map(date_to_ssm)
            else:
                pred_df['SSM_avg_obs'] = np.nan
        else:
            pred_df['SSM_avg_obs'] = np.nan

        col_order = ['Date', 'SSM_avg_obs', 'RZSM_obs', 'RZSM_pred']
        pred_df = pred_df[[c for c in col_order if c in pred_df.columns]]

    save_dir = output_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'predictions_{test_station}.csv'
    pred_df.to_csv(save_path, index=False)


# ---------------------------------------------------------------------------
# Extra metrics
# ---------------------------------------------------------------------------

def compute_extra_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute NRMSE and PBIAS from raw predictions / observations."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    obs_range = y_true.max() - y_true.min()
    nrmse = rmse / obs_range if obs_range > 0 else float('nan')
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else float('nan')
    return {'nrmse': nrmse, 'pbias': pbias}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def create_sensitivity_bar_chart(summary_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart comparing mean metrics across experiments."""

    experiments = summary_df['experiment'].values
    n = len(experiments)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # R²
    ax = axes[0, 0]
    vals = summary_df['r2_mean'].values
    bars = ax.bar(x, vals, color='steelblue')
    ax.set_ylabel('R²')
    ax.set_title('Mean R²')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # RMSE
    ax = axes[0, 1]
    vals = summary_df['rmse_mean'].values
    bars = ax.bar(x, vals, color='coral')
    ax.set_ylabel('RMSE')
    ax.set_title('Mean RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # NRMSE
    ax = axes[1, 0]
    vals = summary_df['nrmse_mean'].values
    bars = ax.bar(x, vals, color='mediumpurple')
    ax.set_ylabel('NRMSE')
    ax.set_title('Mean NRMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # PBIAS
    ax = axes[1, 1]
    vals = summary_df['pbias_mean'].values
    colors = ['forestgreen' if v >= 0 else 'tomato' for v in vals]
    bars = ax.bar(x, vals, color=colors)
    ax.set_ylabel('PBIAS (%)')
    ax.set_title('Mean Percent Bias')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=35, ha='right', fontsize=8)
    for b, v in zip(bars, vals):
        offset = 0.2 if v >= 0 else -0.5
        ax.text(b.get_x() + b.get_width() / 2, v + offset, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Sensitivity Analysis – Leave-One-Station-Out CV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / 'sensitivity_bar_chart.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to: {save_path}")


def create_loss_curves(all_histories: Dict[str, List[Dict]], output_dir: Path):
    """Plot training and validation loss curves for all experiments on the same chart.

    Parameters
    ----------
    all_histories : dict
        {exp_name: [history_fold1, history_fold2, history_fold3]}
        Each history dict has keys 'train_loss' and 'val_loss' (lists of floats).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.get_cmap('tab10', len(all_histories))

    for idx, (exp_name, fold_histories) in enumerate(all_histories.items()):
        color = cmap(idx)

        # Average the loss curves across folds (truncate to shortest fold)
        min_epochs = min(len(h['train_loss']) for h in fold_histories)
        train_arr = np.array([h['train_loss'][:min_epochs] for h in fold_histories])
        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)

        epochs = np.arange(1, min_epochs + 1)

        ax = axes[0]
        ax.plot(epochs, train_mean, label=exp_name, color=color, linewidth=1.5)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.1)

        if all(h.get('val_loss') for h in fold_histories):
            val_arr = np.array([h['val_loss'][:min_epochs] for h in fold_histories])
            val_mean = val_arr.mean(axis=0)
            val_std = val_arr.std(axis=0)

            ax = axes[1]
            ax.plot(epochs, val_mean, label=exp_name, color=color, linewidth=1.5)
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                             color=color, alpha=0.1)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Validation Loss')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Loss Curves by Experiment (mean +/- std across folds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / 'sensitivity_loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {save_path}")


def create_delta_chart(summary_df: pd.DataFrame, output_dir: Path):
    """Show the change in R² relative to baseline for each ablation."""
    baseline_rows = summary_df.loc[summary_df['experiment'] == 'baseline', 'r2_mean'].values
    if len(baseline_rows) == 0:
        print("Skipping delta chart – baseline experiment not included in this run.")
        return
    baseline_r2 = baseline_rows[0]
    ablations = summary_df[summary_df['experiment'] != 'baseline'].copy()
    if ablations.empty:
        print("Skipping delta chart – no ablation experiments to compare.")
        return
    ablations['delta_r2'] = ablations['r2_mean'] - baseline_r2

    x = np.arange(len(ablations))
    colors = ['forestgreen' if d >= 0 else 'tomato' for d in ablations['delta_r2']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(x, ablations['delta_r2'].values, color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels(ablations['experiment'].values, fontsize=9)
    ax.set_xlabel('Delta R² (relative to baseline)')
    ax.set_title(f'Impact of Removing Feature Groups (Baseline R² = {baseline_r2:.3f})')
    ax.axvline(0, color='k', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    for b, v in zip(bars, ablations['delta_r2'].values):
        offset = 0.002 if v >= 0 else -0.002
        ax.text(v + offset, b.get_y() + b.get_height() / 2,
                f'{v:+.3f}', va='center', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / 'sensitivity_delta_r2.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Delta R² chart saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    all_exp_names = list(build_experiments().keys())
    parser = argparse.ArgumentParser(
        description='Feature sensitivity analysis for LSTM soil-moisture model.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        choices=all_exp_names,
        default=None,
        metavar='EXP',
        help='Run only these experiments. Available:\n  ' + '\n  '.join(all_exp_names)
             + '\nDefault: run all.',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=15,
        metavar='N',
        help='Sequence length for LSTM input (default: 15). Outputs go to outputs/sensitivity/seq<N>/',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path('outputs/sensitivity') / f'seq{args.seq_length}'
    output_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_root), log_level='INFO')
    log = logging.getLogger(__name__)

    all_experiments = build_experiments()

    if args.experiments:
        experiments = {k: all_experiments[k] for k in args.experiments}
    else:
        experiments = all_experiments

    log.info("=" * 70)
    log.info("FEATURE SENSITIVITY ANALYSIS")
    log.info("=" * 70)
    log.info(f"Sequence length: {args.seq_length}")
    log.info(f"Experiments: {list(experiments.keys())}")
    log.info(f"CV folds: {len(CV_FOLDS)}")

    torch.manual_seed(42)
    np.random.seed(42)

    all_rows: List[dict] = []
    all_histories: Dict[str, List[Dict]] = {}
    t0 = time.time()

    for exp_name, exp_cfg in experiments.items():
        log.info(f"\n{'#' * 70}")
        log.info(f"# EXPERIMENT: {exp_name}  –  {exp_cfg['description']}")
        log.info(f"#   dynamic_cols ({len(exp_cfg['dynamic_cols'])}): {exp_cfg['dynamic_cols']}")
        log.info(f"#   static_cols  ({len(exp_cfg['static_cols'])}): first 5 = {exp_cfg['static_cols'][:5]} ...")
        log.info(f"#   use_presto_static={exp_cfg['use_presto_static']}, "
                 f"exclude_irrigation_static={exp_cfg['exclude_irrigation_static']}")
        log.info(f"{'#' * 70}")

        config = CVConfig()
        config.seq_length = args.seq_length
        config.dynamic_cols = exp_cfg['dynamic_cols']
        config.static_cols = exp_cfg['static_cols']
        config.use_presto_static = exp_cfg['use_presto_static']
        config.output_dir = str(output_root / exp_name)

        # Store the irrigation flag so FeatureConfig picks it up
        config._exclude_irrigation_static = exp_cfg['exclude_irrigation_static']

        # Reset seeds per experiment for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        fold_results = []
        for fold in CV_FOLDS:
            result = _run_fold_with_overrides(fold, config, log)
            fold_results.append(result)

            extras = compute_extra_metrics(result['y_pred'], result['y_true'])
            m = result['metrics']
            all_rows.append({
                'experiment': exp_name,
                'fold': result['fold_name'],
                'test_station': result['test_station'],
                'r2': m['r2'],
                'rmse': m['rmse'],
                'nrmse': extras['nrmse'],
                'mae': m['mae'],
                'bias': m['bias'],
                'pbias': extras['pbias'],
                'n_samples': m['n_samples'],
            })

            save_fold_predictions(result, exp_name, config.data_dir, output_root)

        all_histories[exp_name] = [r['history'] for r in fold_results]

        # Per-experiment scatter plots
        try:
            exp_dir = Path(config.output_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            create_combined_scatter_plot(fold_results, exp_dir, model_type=f"LSTM – {exp_name}")
        except Exception as e:
            log.warning(f"Could not create scatter plot for {exp_name}: {e}")

    elapsed = time.time() - t0
    log.info(f"\nAll experiments finished in {elapsed / 60:.1f} minutes")

    # ----- Build results DataFrame and summary -----
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_root / 'sensitivity_results.csv', index=False)
    log.info(f"Full results saved to {output_root / 'sensitivity_results.csv'}")

    summary = (
        results_df
        .groupby('experiment')
        .agg(
            r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
            rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
            nrmse_mean=('nrmse', 'mean'), nrmse_std=('nrmse', 'std'),
            mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
            bias_mean=('bias', 'mean'), bias_std=('bias', 'std'),
            pbias_mean=('pbias', 'mean'), pbias_std=('pbias', 'std'),
        )
        .reset_index()
    )

    # Preserve experiment ordering (baseline first)
    exp_order = list(experiments.keys())
    summary['experiment'] = pd.Categorical(summary['experiment'], categories=exp_order, ordered=True)
    summary = summary.sort_values('experiment').reset_index(drop=True)

    summary.to_csv(output_root / 'sensitivity_summary.csv', index=False)
    log.info(f"Summary saved to {output_root / 'sensitivity_summary.csv'}")

    # ----- Print summary table -----
    print("\n" + "=" * 100)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 100)
    header = (f"{'Experiment':<22} {'R²':>8} {'RMSE':>8} {'NRMSE':>8} "
              f"{'MAE':>8} {'Bias':>8} {'PBIAS%':>8}")
    print(header)
    print("-" * 100)
    for _, row in summary.iterrows():
        print(f"{row['experiment']:<22} "
              f"{row['r2_mean']:>7.4f}  {row['rmse_mean']:>7.4f}  {row['nrmse_mean']:>7.4f}  "
              f"{row['mae_mean']:>7.4f}  {row['bias_mean']:>7.4f}  {row['pbias_mean']:>7.2f}")
    print("=" * 100)

    # ----- Plots -----
    create_loss_curves(all_histories, output_root)
    create_sensitivity_bar_chart(summary, output_root)
    create_delta_chart(summary, output_root)

    log.info("Sensitivity analysis complete!")
    return results_df, summary


# ---------------------------------------------------------------------------
# Helper: run a fold with the custom config (injects exclude_irrigation_static)
# ---------------------------------------------------------------------------

def _run_fold_with_overrides(fold: Dict, config: CVConfig, logger: logging.Logger) -> Dict:
    """Thin wrapper around run_single_fold that injects exclude_irrigation_static."""
    from src.config import FeatureConfig

    exclude_irr = getattr(config, '_exclude_irrigation_static', False)

    _orig_init = FeatureConfig.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.exclude_irrigation_static = exclude_irr

    FeatureConfig.__init__ = _patched_init
    try:
        result = run_single_fold(fold, config, logger)
    finally:
        FeatureConfig.__init__ = _orig_init

    return result


if __name__ == '__main__':
    main()
