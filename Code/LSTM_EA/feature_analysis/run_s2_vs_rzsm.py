#!/usr/bin/env python3
"""
Analysis 4: Sentinel-2 Bands vs RZSM.
Scatter plots and correlations for each S2 band vs RZSM.
Outputs: s2_vs_rzsm_scatter.png, s2_band_correlations.png, s2_band_correlations.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from .config import S2_COLS, TARGET_COL
from .data_loading import get_combined_df, get_available_cols, load_site_data


def s2_vs_rzsm_scatter(site_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Scatter plots and correlations for each S2 band vs RZSM."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: SENTINEL-2 BANDS vs RZSM CORRELATION")
    print("=" * 70)

    combined = get_combined_df(site_dfs)
    avail_s2 = get_available_cols(combined, S2_COLS)

    if not avail_s2:
        print("  No S2 columns available, skipping")
        return

    corr_results = []
    for col in avail_s2:
        valid = combined[[col, TARGET_COL, 'DOY']].dropna()
        if len(valid) < 20:
            continue
        r = np.corrcoef(valid[col], valid[TARGET_COL])[0, 1]
        corr_results.append({'band': col, 'pearson_r': r, 'n': len(valid)})
        print(f"  {col:>6s} vs RZSM: r = {r:+.4f}")

    key_bands = [b for b in ['b11', 'b12', 'ndvi'] if b in avail_s2]
    if not key_bands:
        key_bands = avail_s2[:3]

    n_plots = len(key_bands)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

    for idx, band in enumerate(key_bands):
        valid = combined[[band, TARGET_COL, 'Site']].dropna()
        ax = axes[0, idx]
        site_colors = {'ne1': '#1f77b4', 'ne2': '#ff7f0e', 'ne3': '#2ca02c'}
        for site in valid['Site'].unique():
            site_data = valid[valid['Site'] == site]
            ax.scatter(site_data[band], site_data[TARGET_COL],
                      alpha=0.3, s=8, label=site.upper(),
                      c=site_colors.get(site, 'gray'))
        r = np.corrcoef(valid[band], valid[TARGET_COL])[0, 1]
        ax.set_xlabel(f'{band.upper()}', fontsize=11)
        ax.set_ylabel('RZSM', fontsize=11)
        ax.set_title(f'{band.upper()} vs RZSM\nr = {r:+.3f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sentinel-2 Bands as Moisture Proxy', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 's2_vs_rzsm_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    if corr_results:
        corr_df = pd.DataFrame(corr_results).sort_values('pearson_r')
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['steelblue' if r > 0 else 'coral' for r in corr_df['pearson_r']]
        ax.barh(corr_df['band'], corr_df['pearson_r'], color=colors)
        ax.set_xlabel('Pearson Correlation with RZSM')
        ax.set_title('Sentinel-2 Band Correlations with RZSM')
        ax.axvline(0, color='k', lw=0.8)
        ax.grid(axis='x', alpha=0.3)
        for i, (_, row) in enumerate(corr_df.iterrows()):
            ax.text(row['pearson_r'] + 0.01 * np.sign(row['pearson_r']),
                    i, f'{row["pearson_r"]:+.3f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / 's2_band_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        pd.DataFrame(corr_results).to_csv(output_dir / 's2_band_correlations.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Sentinel-2 bands vs RZSM analysis.')
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
    s2_vs_rzsm_scatter(site_dfs, output_dir)
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
