#!/usr/bin/env python3
"""
Analysis 2: Cross-Site RZSM Correlation.
Pairwise RZSM correlation and R² between sites.
Outputs: cross_site_correlation.csv, cross_site_correlation.png, cross_site_timeseries.png
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


def cross_site_correlation(site_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """
    Compute pairwise RZSM correlation and R² between sites.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CROSS-SITE RZSM CORRELATION")
    print("=" * 70)

    merged = None
    for site_name, df in site_dfs.items():
        if TARGET_COL not in df.columns:
            continue
        site_col = df[['Date', TARGET_COL]].rename(columns={TARGET_COL: f'RZSM_{site_name}'})
        if merged is None:
            merged = site_col
        else:
            merged = merged.merge(site_col, on='Date', how='inner')

    if merged is None or len(merged) < 10:
        print("  Not enough overlapping data for cross-site correlation")
        return pd.DataFrame()

    rzsm_cols = [c for c in merged.columns if c.startswith('RZSM_')]
    merged_clean = merged.dropna(subset=rzsm_cols)
    print(f"  Overlapping days: {len(merged_clean)}")

    corr_matrix = merged_clean[rzsm_cols].corr()
    print("\n  Pearson correlation matrix:")
    print(corr_matrix.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n  Pairwise R² (using one site to predict another):")
    sites = list(site_dfs.keys())
    r2_results = []
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if i >= j:
                continue
            c1, c2 = f'RZSM_{s1}', f'RZSM_{s2}'
            if c1 in merged_clean.columns and c2 in merged_clean.columns:
                vals = merged_clean[[c1, c2]].dropna()
                y1, y2 = vals[c1].values, vals[c2].values
                ss_res = np.sum((y1 - y2) ** 2)
                ss_tot = np.sum((y1 - np.mean(y1)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                corr = np.corrcoef(y1, y2)[0, 1]
                r2_results.append({'site_pair': f'{s1}-{s2}', 'pearson_r': corr,
                                  'r2_direct': r2, 'n_days': len(vals)})
                print(f"    {s1} vs {s2}: r = {corr:.4f}, R² = {r2:.4f}")

    fig, axes = plt.subplots(1, len(r2_results), figsize=(6 * len(r2_results), 5), squeeze=False)
    for idx, res in enumerate(r2_results):
        s1, s2 = res['site_pair'].split('-')
        c1, c2 = f'RZSM_{s1}', f'RZSM_{s2}'
        vals = merged_clean[[c1, c2]].dropna()
        ax = axes[0, idx]
        ax.scatter(vals[c1], vals[c2], alpha=0.3, s=8, c='darkorange')
        lims = [min(vals[c1].min(), vals[c2].min()), max(vals[c1].max(), vals[c2].max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel(f'RZSM {s1}')
        ax.set_ylabel(f'RZSM {s2}')
        ax.set_title(f'{s1} vs {s2}\nr = {res["pearson_r"]:.3f}')
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Site RZSM Correlation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_site_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in rzsm_cols:
        label = col.replace('RZSM_', '')
        ax.plot(merged_clean['Date'], merged_clean[col], label=label, alpha=0.8, lw=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('RZSM')
    ax.set_title('RZSM Time Series — All Sites')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_site_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    r2_df = pd.DataFrame(r2_results)
    r2_df.to_csv(output_dir / 'cross_site_correlation.csv', index=False)
    return r2_df


def main():
    parser = argparse.ArgumentParser(description='Cross-site RZSM correlation analysis.')
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
    cross_site_correlation(site_dfs, output_dir)
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
