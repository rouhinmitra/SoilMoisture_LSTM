#!/usr/bin/env python3
"""
Analysis 1: Climatological Baseline.
Computes daily climatological mean RZSM per site and evaluates it as a naive baseline.
Outputs: climatological_baseline.csv, climatological_baseline.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from .config import TARGET_COL
from .data_loading import load_site_data


def climatological_baseline(site_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """
    Compute a daily climatological mean RZSM per site (average across years
    for each DOY) and evaluate it as a naive prediction baseline.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: CLIMATOLOGICAL BASELINE")
    print("=" * 70)

    results = []
    fig, axes = plt.subplots(1, len(site_dfs), figsize=(6 * len(site_dfs), 5), squeeze=False)

    for idx, (site_name, df) in enumerate(site_dfs.items()):
        if TARGET_COL not in df.columns:
            print(f"  {site_name}: target column missing, skipping")
            continue

        df_valid = df.dropna(subset=[TARGET_COL]).copy()
        if len(df_valid) == 0:
            continue

        clim = df_valid.groupby('DOY')[TARGET_COL].mean().to_dict()
        df_valid['RZSM_clim'] = df_valid['DOY'].map(clim)
        df_eval = df_valid.dropna(subset=['RZSM_clim'])

        y_true = df_eval[TARGET_COL].values
        y_pred = df_eval['RZSM_clim'].values

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        results.append({'site': site_name, 'r2_climatology': r2, 'rmse_climatology': rmse,
                        'n_samples': len(df_eval)})
        print(f"  {site_name}: Climatological baseline R² = {r2:.4f}, RMSE = {rmse:.2f}")

        ax = axes[0, idx]
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c='teal')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel('Observed RZSM')
        ax.set_ylabel('Climatological Mean RZSM')
        ax.set_title(f'{site_name}\nR² = {r2:.3f}, RMSE = {rmse:.2f}')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Climatological Baseline (DOY Mean) vs Observed RZSM', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'climatological_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'climatological_baseline.csv', index=False)
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Climatological baseline analysis.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing site CSVs')
    parser.add_argument('--presto_path', type=str, default=None,
                        help='Path to Presto embeddings CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs/moisture_proxy_analysis',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_dfs = load_site_data(args.data_dir, args.presto_path)
    if not site_dfs:
        raise SystemExit("No site data loaded. Check --data_dir.")
    climatological_baseline(site_dfs, output_dir)
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
