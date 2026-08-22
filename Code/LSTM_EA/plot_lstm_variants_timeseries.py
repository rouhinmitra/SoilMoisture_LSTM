#!/usr/bin/env python
"""
Plot May–October time series for four LSTM variants plus observed SSM and RZSM.

Reads sensitivity prediction CSVs (baseline, no_ssm, no_presto, no_ssm_no_presto),
merges on Date per site, filters to May–October, and produces one figure per station
with one subplot per year (e.g. 2017–2024). Each subplot shows SSM observed, RZSM observed,
and four LSTM prediction series; one shared legend per figure.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SITES = ['Ne1', 'Ne2', 'Ne3']

# Experiment ID -> display label (order defines plot order)
EXPERIMENTS = {
    'baseline': 'LSTM(all)',
    'no_ssm': 'LSTM(no ssm)',
    'no_presto': 'LSTM(no embeddings)',
    'no_ssm_no_presto': 'LSTM(no ssm and embeddings)',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot LSTM variants time series (May–Oct): SSM/RZSM observed + 4 models.',
    )
    parser.add_argument(
        '--sensitivity-root',
        type=str,
        default=None,
        help='Root directory containing <experiment>/predictions_<site>.csv (default: outputs/sensitivity/seq20)',
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=20,
        help='Sequence length used in sensitivity run; used only if --sensitivity-root not set',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override save directory; default is <sensitivity-root>/lstm_variants_timeseries',
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
    """Compute R², bias, RMSE. Returns nan for invalid."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {'r2': np.nan, 'bias': np.nan, 'rmse': np.nan}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = np.mean(y_pred - y_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {'r2': r2, 'bias': bias, 'rmse': rmse}


def load_merged_site(root: Path, site: str) -> pd.DataFrame | None:
    """Load the four experiment CSVs for one site and merge on Date. Returns None if any missing."""
    base_path = root / 'baseline' / f'predictions_{site}.csv'
    if not base_path.exists():
        print(f'Skipping site {site}: missing {base_path}')
        return None
    try:
        merged = pd.read_csv(base_path, parse_dates=['Date'])
    except Exception as e:
        print(f'Skipping site {site}: error reading {base_path}: {e}')
        return None
    if 'Date' not in merged.columns or 'RZSM_obs' not in merged.columns or 'RZSM_pred' not in merged.columns:
        print(f'Skipping site {site}: missing cols in {base_path}')
        return None
    merged = merged.rename(columns={'RZSM_pred': 'RZSM_baseline'})
    for exp_id in EXPERIMENTS:
        if exp_id == 'baseline':
            continue
        csv_path = root / exp_id / f'predictions_{site}.csv'
        if not csv_path.exists():
            print(f'Skipping site {site}: missing {csv_path}')
            return None
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
        except Exception as e:
            print(f'Skipping site {site}: error reading {csv_path}: {e}')
            return None
        if 'Date' not in df.columns or 'RZSM_pred' not in df.columns:
            return None
        merged = merged.merge(
            df[['Date', 'RZSM_pred']].rename(columns={'RZSM_pred': f'RZSM_{exp_id}'}),
            on='Date',
            how='inner',
        )
    return merged


# Font sizes for all plot text
FONTSIZE_TITLE = 14
FONTSIZE_AXIS = 12
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12
FONTSIZE_SUPTITLE = 16

# Line styles for the six series (used for both plotting and legend)
SERIES_STYLES = [
    ('SSM_avg_obs', 'SSM observed', 'gray', '-'),
    ('RZSM_obs', 'RZSM 25cm observed', 'black', '-'),
    ('RZSM_baseline', 'RZSM LSTM(all features)', 'red', '-.'),
    ('RZSM_no_ssm', 'RZSM LSTM(no SSM)', 'blue', '-.'),
    ('RZSM_no_presto', 'RZSM LSTM(no Embeddings)', 'green', '-.'),
    ('RZSM_no_ssm_no_presto', 'RZSM LSTM(no SSM and Embeddings)', 'orange', '-.'),
]


def plot_one_year_on_ax(
    df: pd.DataFrame,
    ax: plt.Axes,
    year: int,
    show_legend: bool = True,
) -> None:
    """Plot SSM observed, RZSM observed, and four LSTM series on the given ax (May–Oct)."""
    if df.empty or len(df) < 2:
        return
    x = df['Date']

    scale = 100.0  # plot values as fraction (divide by 100)
    for col, label, color, ls in SERIES_STYLES:
        if col not in df.columns or not df[col].notna().any():
            continue
        ax.plot(x, df[col] / scale, color=color, linestyle=ls, linewidth=1.2, alpha=0.9, label=label if show_legend else None)

    # Metrics box (baseline for this year's May–Oct; R², RMSE, Bias in scaled units)
    # if 'RZSM_baseline' in df.columns and 'RZSM_obs' in df.columns:
    #     y_true = df['RZSM_obs'].values / scale
    #     y_pred = df['RZSM_baseline'].values / scale
    #     m = compute_metrics(y_true, y_pred)
    #     text = f"RMSE = {m['rmse']:.4f}\nBias = {m['bias']:.4f}"
    #     ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8,
    #             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_ylim(0.18, 0.45)
    xmin, xmax = df['Date'].min(), df['Date'].max()
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_title(str(year), fontsize=FONTSIZE_TITLE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
    ax.tick_params(axis='x', rotation=0)


def plot_site_all_years(
    growing: pd.DataFrame,
    site: str,
    save_path: Path,
    years: list[int] | None = None,
) -> None:
    """One figure per site: one subplot per year (May–Oct), single legend at bottom."""
    if years is None:
        years = sorted(growing['year'].unique())
    years = [y for y in years if y in growing['year'].values and len(growing[growing['year'] == y]) >= 2]
    if not years:
        return

    n_years = len(years)
    ncols = min(4, n_years)
    nrows = (n_years + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, year in enumerate(years):
        ax = axes.flat[idx]
        group = growing[growing['year'] == year]
        # Only add legend labels on the first subplot so we get one legend per figure
        plot_one_year_on_ax(group, ax, year, show_legend=(idx == 0))
    for idx in range(n_years, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    for i in range(n_years):
        ax = axes.flat[i]
        if i % ncols == 0:
            ax.set_ylabel('RZSM 25cm ($m^3/m^3$)', fontsize=FONTSIZE_AXIS)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Date', fontsize=FONTSIZE_AXIS)
        else:
            ax.set_xlabel('')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=FONTSIZE_LEGEND, frameon=True)
    fig.suptitle(f'US-{site}  May–October', fontsize=FONTSIZE_SUPTITLE, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    root = Path(args.sensitivity_root) if args.sensitivity_root else Path(f'outputs/sensitivity/seq{args.seq_length}')
    out_dir = Path(args.output_dir) if args.output_dir else root / 'lstm_variants_timeseries'

    may_oct_months = [5, 6, 7, 8, 9, 10]
    count = 0

    for site in args.sites:
        merged = load_merged_site(root, site)
        if merged is None:
            continue
        merged['year'] = merged['Date'].dt.year
        merged['month'] = merged['Date'].dt.month
        growing = merged[merged['month'].isin(may_oct_months)]

        save_path = out_dir / f'{site}_may_oct_all_years.png'
        plot_site_all_years(growing, site, save_path)
        print(f'Saved: {save_path}')
        count += 1

    print(f'\nPlots saved under: {out_dir} (total: {count})')


if __name__ == '__main__':
    main()
