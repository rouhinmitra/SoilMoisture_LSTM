#!/usr/bin/env python3
"""
Orchestrator: run all moisture proxy diagnostic analyses in order.
Loads data once, runs Analyses 1–6, writes summary from saved CSVs.
"""
# Allow running as script: python feature_analysis/run_all.py (from LSTM_EA dir)
if __name__ == '__main__' and not __package__:
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    runpy.run_module('feature_analysis.run_all', run_name='__main__')
    sys.exit(0)

import argparse
import sys
from pathlib import Path

from .config import YEAR_RANGE
from .data_loading import load_site_data, has_presto
from .run_climatology import climatological_baseline
from .run_cross_site import cross_site_correlation
from .run_linear_r2 import feature_group_linear_regression
from .run_s2_vs_rzsm import s2_vs_rzsm_scatter
from .run_presto_embedding import presto_embedding_analysis
from .run_summary import create_summary


def main():
    parser = argparse.ArgumentParser(
        description='Moisture proxy diagnostic analysis for RZSM LSTM ablation study.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing site CSVs (ne1_1_maize.csv, etc.)')
    parser.add_argument('--presto_path', type=str, default=None,
                        help='Path to Presto interpolated embeddings CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='outputs/moisture_proxy_analysis',
                        help='Output directory for plots and results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MOISTURE PROXY DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print(f"Data dir:    {args.data_dir}")
    print(f"Presto path: {args.presto_path or 'Not provided'}")
    print(f"Output dir:  {output_dir}")
    print(f"Year range:  {YEAR_RANGE}")

    print("\n--- Loading data ---")
    site_dfs = load_site_data(args.data_dir, args.presto_path)
    if not site_dfs:
        print("ERROR: No site data loaded. Check --data_dir path.")
        sys.exit(1)

    hp = has_presto(site_dfs)
    print(f"\nPresto embeddings available: {hp}")

    climatological_baseline(site_dfs, output_dir)
    cross_site_correlation(site_dfs, output_dir)
    feature_group_linear_regression(site_dfs, output_dir, has_presto=hp)
    s2_vs_rzsm_scatter(site_dfs, output_dir)
    if hp:
        presto_embedding_analysis(site_dfs, output_dir)
    create_summary(output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
