#!/usr/bin/env python
"""
Plot time series by year for sensitivity experiments (baseline and no_ssm).

Reads prediction CSVs from run_sensitivity.py output, splits by site and year,
and saves one figure per (site, year, experiment) with SSM observed, RZSM predicted,
and RZSM observed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SITES = ['Ne1', 'Ne2', 'Ne3']
DEFAULT_EXPERIMENTS = ['baseline', 'no_ssm']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot sensitivity time series by year (SSM observed, RZSM pred/obs).',
    )
    parser.add_argument(
        '--sensitivity-root',
        type=str,
        default='outputs/sensitivity',
        help='Root directory containing <experiment>/predictions_<site>.csv',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override save directory; default is <sensitivity-root>/timeseries_by_year',
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=DEFAULT_EXPERIMENTS,
        choices=['baseline', 'no_ssm'],
        metavar='EXP',
        help=f'Experiments to plot (default: {DEFAULT_EXPERIMENTS})',
    )
    parser.add_argument(
        '--sites',
        nargs='+',
        default=DEFAULT_SITES,
        metavar='SITE',
        help=f'Sites to plot (default: {DEFAULT_SITES})',
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², bias, RMSE, PBIAS, NRMSE, and correlation. Returns nan for invalid."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {'r2': np.nan, 'bias': np.nan, 'rmse': np.nan, 'pbias': np.nan, 'nrmse': np.nan, 'corr': np.nan}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = np.mean(y_pred - y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan
    obs_range = y_true.max() - y_true.min()
    nrmse = rmse / obs_range if obs_range > 0 else np.nan
    corr = np.corrcoef(y_true, y_pred)[0, 1] if n >= 2 else np.nan
    return {'r2': r2, 'bias': bias, 'rmse': rmse, 'pbias': pbias, 'nrmse': nrmse, 'corr': corr}


def plot_timeseries_for_year(df: pd.DataFrame, site: str, year: int, experiment: str, save_path: Path):
    """Plot SSM observed, RZSM predicted, RZSM observed for one site/year and save."""
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = df['Date']

    if 'SSM_avg_obs' in df.columns and df['SSM_avg_obs'].notna().any():
        ax.plot(x, df['SSM_avg_obs'], color='#2ecc71', label='SSM observed', linewidth=1.2, alpha=0.9)
    if 'RZSM_pred' in df.columns and df['RZSM_pred'].notna().any():
        ax.plot(x, df['RZSM_pred'], color='#3498db', label='RZSM predicted', linewidth=1.2, alpha=0.9)
    if 'RZSM_obs' in df.columns and df['RZSM_obs'].notna().any():
        ax.plot(x, df['RZSM_obs'], color='#e74c3c', label='RZSM observed', linewidth=1.2, alpha=0.9)

    ax.set_xlabel('Date')
    ax.set_ylabel('Soil moisture')
    ax.set_ylim(0, 45)
    ax.set_title(f'{site} {year} – {experiment}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    root = Path(args.sensitivity_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / 'timeseries_by_year'

    # Collect per-year metrics for printing
    year_stats = []

    for experiment in args.experiments:
        for site in args.sites:
            csv_path = root / experiment / f'predictions_{site}.csv'
            if not csv_path.exists():
                print(f'Skipping (missing): {csv_path}')
                continue

            try:
                df = pd.read_csv(csv_path, parse_dates=['Date'])
            except Exception as e:
                print(f'Error reading {csv_path}: {e}')
                continue

            if 'Date' not in df.columns or 'RZSM_obs' not in df.columns or 'RZSM_pred' not in df.columns:
                print(f'Skipping (missing cols): {csv_path}')
                continue

            df['year'] = df['Date'].dt.year
            for year, group in df.groupby('year'):
                if group.empty or len(group) < 2:
                    continue
                save_path = out_dir / experiment / f'{site}_{year}.png'
                plot_timeseries_for_year(group, site, year, experiment, save_path)
                print(f'Saved: {save_path}')

                y_true = group['RZSM_obs'].values
                y_pred = group['RZSM_pred'].values
                m = compute_metrics(y_true, y_pred)
                year_stats.append({
                    'experiment': experiment,
                    'site': site,
                    'year': year,
                    **m,
                })

    # ----- Per-year stats table -----
    if year_stats:
        stats_df = pd.DataFrame(year_stats)
        print('\n' + '=' * 100)
        print('PER-YEAR STATS (R², bias, RMSE, PBIAS, NRMSE, correlation)')
        print('=' * 100)
        for exp in args.experiments:
            sub = stats_df[stats_df['experiment'] == exp]
            if sub.empty:
                continue
            print(f'\n--- {exp} ---')
            print(f"{'site':<6} {'year':<6} {'R2':>8} {'bias':>8} {'RMSE':>8} {'PBIAS%':>8} {'NRMSE':>8} {'corr':>8}")
            print('-' * 70)
            for _, row in sub.sort_values(['site', 'year']).iterrows():
                print(f"{row['site']:<6} {int(row['year']):<6} {row['r2']:>8.4f} {row['bias']:>8.4f} {row['rmse']:>8.4f} "
                      f"{row['pbias']:>8.2f} {row['nrmse']:>8.4f} {row['corr']:>8.4f}")

    # ----- Average growing season (May–October) R² per experiment -----
    print('\n' + '=' * 60)
    print('AVERAGE GROWING SEASON R² (May–October, out-of-sample)')
    print('=' * 60)
    for experiment in args.experiments:
        r2_list = []
        for site in args.sites:
            csv_path = root / experiment / f'predictions_{site}.csv'
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, parse_dates=['Date'])
            except Exception:
                continue
            if 'Date' not in df.columns or 'RZSM_obs' not in df.columns or 'RZSM_pred' not in df.columns:
                continue
            # May = 5 .. October = 10
            df['month'] = df['Date'].dt.month
            growing = df[df['month'].isin([5, 6, 7, 8, 9, 10])]
            if len(growing) < 2:
                continue
            y_true = growing['RZSM_obs'].values
            y_pred = growing['RZSM_pred'].values
            m = compute_metrics(y_true, y_pred)
            r2_list.append(m['r2'])
        if r2_list:
            avg_r2 = np.nanmean(r2_list)
            print(f'  {experiment:<12}  mean R² = {avg_r2:.4f}  (over {len(r2_list)} stations)')
        else:
            print(f'  {experiment:<12}  (no data)')

    print(f'\nPlots saved under: {out_dir}')


if __name__ == '__main__':
    main()
