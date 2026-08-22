#!/usr/bin/env python3
"""
Analysis 5: Presto Embedding Analysis.
PCA colored by RZSM/site, per-dimension correlation with RZSM and SSM.
Outputs: presto_pca.png, presto_dimension_correlations.png/csv, presto_top_dimension_scatter.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import PRESTO_COLS, TARGET_COL
from .data_loading import get_combined_df, get_available_cols, load_site_data


def presto_embedding_analysis(site_dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    PCA of Presto embeddings colored by RZSM; per-dimension correlation with RZSM and SSM.
    Skips gracefully if Presto columns are missing.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: PRESTO EMBEDDING ANALYSIS")
    print("=" * 70)

    combined = get_combined_df(site_dfs)
    avail_presto = get_available_cols(combined, PRESTO_COLS)

    if len(avail_presto) < 10:
        print("  Presto columns not available, skipping")
        return

    analysis_cols = avail_presto + [TARGET_COL] + get_available_cols(combined, ['SSM_avg', 'SSM'])
    valid = combined[analysis_cols + ['Site']].dropna(subset=avail_presto + [TARGET_COL])

    if len(valid) < 50:
        print(f"  Only {len(valid)} valid rows, skipping")
        return

    print(f"  Analyzing {len(avail_presto)} Presto dimensions, {len(valid)} samples")

    X_presto = valid[avail_presto].values
    y_rzsm = valid[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_presto)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_rzsm, cmap='viridis',
                        alpha=0.4, s=8)
    plt.colorbar(sc, ax=axes[0], label='RZSM')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('Presto PCA — colored by RZSM')
    axes[0].grid(True, alpha=0.3)

    site_colors = {'ne1': '#1f77b4', 'ne2': '#ff7f0e', 'ne3': '#2ca02c'}
    for site in valid['Site'].unique():
        mask = valid['Site'].values == site
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=8,
                       label=site.upper(), c=site_colors.get(site, 'gray'))
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('Presto PCA — colored by Site')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Presto Embedding Structure', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'presto_pca.png', dpi=150, bbox_inches='tight')
    plt.close()

    dim_corrs = []
    for col in avail_presto:
        r_rzsm = np.corrcoef(valid[col].values, y_rzsm)[0, 1]
        row = {'dimension': col, 'corr_with_RZSM': r_rzsm}
        for ssm_col in ['SSM_avg', 'SSM']:
            if ssm_col in valid.columns:
                ssm_valid = valid[[col, ssm_col]].dropna()
                if len(ssm_valid) > 20:
                    r_ssm = np.corrcoef(ssm_valid[col], ssm_valid[ssm_col])[0, 1]
                    row['corr_with_SSM'] = r_ssm
                    break
        dim_corrs.append(row)

    dim_df = pd.DataFrame(dim_corrs)
    dim_df = dim_df.sort_values('corr_with_RZSM', ascending=False)

    print("\n  Top 10 Presto dimensions correlated with RZSM:")
    for _, row in dim_df.head(10).iterrows():
        ssm_str = f", SSM r={row.get('corr_with_SSM', np.nan):.3f}" if 'corr_with_SSM' in row else ""
        print(f"    {row['dimension']}: RZSM r={row['corr_with_RZSM']:+.4f}{ssm_str}")
    print("\n  Bottom 10 Presto dimensions (most negative correlation):")
    for _, row in dim_df.tail(10).iterrows():
        ssm_str = f", SSM r={row.get('corr_with_SSM', np.nan):.3f}" if 'corr_with_SSM' in row else ""
        print(f"    {row['dimension']}: RZSM r={row['corr_with_RZSM']:+.4f}{ssm_str}")

    fig, ax = plt.subplots(figsize=(14, 5))
    dim_indices = range(len(dim_df))
    ax.bar(dim_indices, dim_df['corr_with_RZSM'].values, alpha=0.7, label='corr with RZSM')
    if 'corr_with_SSM' in dim_df.columns:
        ax.scatter(dim_indices, dim_df['corr_with_SSM'].values, c='red', s=10, alpha=0.6,
                  label='corr with SSM', zorder=5)
    ax.set_xlabel('Presto Dimension (sorted by RZSM correlation)')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Per-Dimension Presto Correlation with RZSM and SSM')
    ax.axhline(0, color='k', lw=0.8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'presto_dimension_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    if 'corr_with_SSM' in dim_df.columns:
        top_dim = dim_df.iloc[0]['dimension']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        valid_sub = valid[[top_dim, TARGET_COL]].dropna()
        axes[0].scatter(valid_sub[top_dim], valid_sub[TARGET_COL], alpha=0.3, s=8, c='teal')
        r = np.corrcoef(valid_sub[top_dim], valid_sub[TARGET_COL])[0, 1]
        axes[0].set_xlabel(top_dim)
        axes[0].set_ylabel('RZSM')
        axes[0].set_title(f'{top_dim} vs RZSM (r = {r:+.3f})')
        axes[0].grid(True, alpha=0.3)
        for ssm_col in ['SSM_avg', 'SSM']:
            if ssm_col in valid.columns:
                valid_sub2 = valid[[top_dim, ssm_col]].dropna()
                axes[1].scatter(valid_sub2[top_dim], valid_sub2[ssm_col], alpha=0.3, s=8, c='darkorange')
                r2 = np.corrcoef(valid_sub2[top_dim], valid_sub2[ssm_col])[0, 1]
                axes[1].set_xlabel(top_dim)
                axes[1].set_ylabel(ssm_col)
                axes[1].set_title(f'{top_dim} vs {ssm_col} (r = {r2:+.3f})')
                axes[1].grid(True, alpha=0.3)
                break
        plt.suptitle(f'Most RZSM-Correlated Presto Dimension: {top_dim}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'presto_top_dimension_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

    dim_df.to_csv(output_dir / 'presto_dimension_correlations.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Presto embedding analysis.')
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
    presto_embedding_analysis(site_dfs, output_dir)
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
