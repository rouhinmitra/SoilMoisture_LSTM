#!/usr/bin/env python3
"""
Analysis 6: Summary.
Reads climatological_baseline.csv, cross_site_correlation.csv, feature_group_linear_r2.csv
from output_dir and writes a combined narrative summary.txt.
"""

import argparse
from pathlib import Path

import pandas as pd


def create_summary(output_dir: Path) -> None:
    """
    Read saved CSVs from output_dir and write summary.txt.
    Handles missing files gracefully (e.g. only a subset of analyses were run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_lines = []

    summary_lines.append("MOISTURE PROXY DIAGNOSTIC ANALYSIS — SUMMARY")
    summary_lines.append("=" * 60)

    clim_path = output_dir / 'climatological_baseline.csv'
    if clim_path.exists():
        clim_results = pd.read_csv(clim_path)
        if len(clim_results) > 0:
            mean_clim_r2 = clim_results['r2_climatology'].mean()
            summary_lines.append(f"\n1. CLIMATOLOGICAL BASELINE:")
            summary_lines.append(f"   Mean R² (DOY climatology): {mean_clim_r2:.4f}")
            for _, row in clim_results.iterrows():
                summary_lines.append(f"   {row['site']}: R² = {row['r2_climatology']:.4f}")
            summary_lines.append(f"   → If model R² is close to {mean_clim_r2:.3f}, the model")
            summary_lines.append(f"     is mainly learning seasonal climatology.")

    cross_path = output_dir / 'cross_site_correlation.csv'
    if cross_path.exists():
        cross_site_results = pd.read_csv(cross_path)
        if len(cross_site_results) > 0:
            mean_r = cross_site_results['pearson_r'].mean()
            summary_lines.append(f"\n2. CROSS-SITE CORRELATION:")
            summary_lines.append(f"   Mean inter-site Pearson r: {mean_r:.4f}")
            for _, row in cross_site_results.iterrows():
                summary_lines.append(f"   {row['site_pair']}: r = {row['pearson_r']:.4f}")
            if mean_r > 0.7:
                summary_lines.append(f"   → Sites are highly correlated; leave-one-out CV")
                summary_lines.append(f"     is an easier task than true spatial generalization.")

    linear_path = output_dir / 'feature_group_linear_r2.csv'
    if linear_path.exists():
        linear_r2_results = pd.read_csv(linear_path)
        if len(linear_r2_results) > 0:
            summary_lines.append(f"\n3. FEATURE-GROUP LINEAR R² (Ridge, 5-fold CV):")
            for _, row in linear_r2_results.iterrows():
                summary_lines.append(f"   {row['feature_group']:<40s} R² = {row['r2_cv_mean']:.4f} ± {row['r2_cv_std']:.4f}")
            summary_lines.append(f"\n   → If Presto alone ≈ SSM alone, they encode redundant")
            summary_lines.append(f"     soil moisture information (Presto via S1 SAR).")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_file = output_dir / 'summary.txt'
    summary_file.write_text(summary_text, encoding='utf-8')
    print(f"\nSummary written to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate summary from saved analysis CSVs.')
    parser.add_argument('--output_dir', type=str, default='outputs/moisture_proxy_analysis',
                        help='Directory containing climatological_baseline.csv, etc.')
    args = parser.parse_args()
    create_summary(Path(args.output_dir))


if __name__ == '__main__':
    main()
