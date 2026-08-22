#!/usr/bin/env python3
"""
Moisture Proxy Diagnostic Analysis
====================================
Entry point that runs the full diagnostic suite (all analyses).

Usage:
    python moisture_proxy_analysis.py --data_dir /path/to/data [--presto_path /path/to/presto.csv]

To run individual analyses or from repo root:
    python -m feature_analysis.run_all --data_dir /path/to/data
    python -m feature_analysis.run_climatology --data_dir /path/to/data
    python -m feature_analysis.run_summary --output_dir outputs/moisture_proxy_analysis
"""

from feature_analysis.run_all import main

if __name__ == '__main__':
    main()
