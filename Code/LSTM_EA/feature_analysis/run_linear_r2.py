#!/usr/bin/env python3
"""
Analysis 3: Feature-Group Linear Regression R² vs RZSM.
Ridge regression from each feature group to RZSM with 5-fold CV (no LSTM, same-day only).
Outputs: feature_group_linear_r2.csv, feature_group_linear_r2.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .config import (
    AE_COLS,
    METEO_PRECIP_COLS,
    PRESTO_COLS,
    S2_COLS,
    SSM_COLS,
    TARGET_COL,
)
from .data_loading import get_combined_df, get_available_cols, has_presto, load_site_data


def feature_group_linear_regression(
    site_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    has_presto: bool = False,
) -> pd.DataFrame:
    """
    Fit Ridge regression from each feature group to RZSM.
    5-fold CV, no LSTM, no temporal context — same-day features → RZSM.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: FEATURE-GROUP LINEAR REGRESSION R² vs RZSM")
    print("=" * 70)

    combined = get_combined_df(site_dfs)

    feature_groups = {
        'SSM (4 features)': SSM_COLS,
        'Sentinel-2 (11 features)': S2_COLS,
        'Meteo + Precip (5 features)': METEO_PRECIP_COLS,
        'Alpha Earth (64 features)': AE_COLS,
    }
    if has_presto:
        feature_groups['Presto (128 features)'] = PRESTO_COLS
        feature_groups['Presto + Alpha Earth (192 features)'] = PRESTO_COLS + AE_COLS

    feature_groups['SSM + Meteo + Precip'] = SSM_COLS + METEO_PRECIP_COLS
    feature_groups['SSM + S2'] = SSM_COLS + S2_COLS
    feature_groups['S2 + Meteo + Precip'] = S2_COLS + METEO_PRECIP_COLS
    feature_groups['All dynamic (no embeddings)'] = SSM_COLS + S2_COLS + METEO_PRECIP_COLS
    if has_presto:
        feature_groups['All features'] = SSM_COLS + S2_COLS + METEO_PRECIP_COLS + AE_COLS + PRESTO_COLS

    results = []
    for group_name, cols in feature_groups.items():
        avail_cols = get_available_cols(combined, cols)
        if not avail_cols:
            print(f"  {group_name}: no columns available, skipping")
            continue

        subset = combined[avail_cols + [TARGET_COL]].dropna()
        if len(subset) < 50:
            print(f"  {group_name}: only {len(subset)} valid rows, skipping")
            continue

        X = subset[avail_cols].values
        y = subset[TARGET_COL].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        r2_mean = scores.mean()
        r2_std = scores.std()

        results.append({
            'feature_group': group_name,
            'n_features': len(avail_cols),
            'n_samples': len(subset),
            'r2_cv_mean': r2_mean,
            'r2_cv_std': r2_std,
        })
        print(f"  {group_name:<40s} R² = {r2_mean:.4f} ± {r2_std:.4f}  "
              f"({len(avail_cols)} features, {len(subset)} samples)")

    results_df = pd.DataFrame(results).sort_values('r2_cv_mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5)))
    y_pos = np.arange(len(results_df))
    bars = ax.barh(y_pos, results_df['r2_cv_mean'].values, xerr=results_df['r2_cv_std'].values,
                   color='steelblue', edgecolor='navy', alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['feature_group'].values, fontsize=9)
    ax.set_xlabel('5-Fold CV R² (Ridge Regression)')
    ax.set_title('Linear Predictability of RZSM from Each Feature Group\n(no LSTM, no temporal context — same-day only)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, results_df['r2_cv_mean'].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_group_linear_r2.png', dpi=150, bbox_inches='tight')
    plt.close()

    results_df.to_csv(output_dir / 'feature_group_linear_r2.csv', index=False)
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Feature-group linear regression R² analysis.')
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
    hp = has_presto(site_dfs)
    feature_group_linear_regression(site_dfs, output_dir, has_presto=hp)
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
