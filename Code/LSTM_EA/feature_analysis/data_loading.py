"""
Data loading for moisture proxy diagnostic analysis.
Loads site CSVs, optionally merges Presto embeddings, filters by year range.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import PRESTO_COLS, STATION_FILES, YEAR_RANGE


def load_site_data(data_dir: str, presto_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load and merge data for all sites."""
    data_dir = Path(data_dir)
    site_dfs = {}

    # Load Presto if available
    presto_df = None
    if presto_path and Path(presto_path).exists():
        print(f"Loading Presto embeddings from: {presto_path}")
        presto_df = pd.read_csv(presto_path, parse_dates=['Date'] if 'Date' in
                                pd.read_csv(presto_path, nrows=0).columns else [])
        if 'Date' not in presto_df.columns and 'date' in presto_df.columns:
            presto_df = presto_df.rename(columns={'date': 'Date'})
        presto_df['Date'] = pd.to_datetime(presto_df['Date'])

        # Normalize site names
        if 'Site' in presto_df.columns:
            presto_df['Site'] = presto_df['Site'].str.lower().str.strip()
        elif 'site' in presto_df.columns:
            presto_df = presto_df.rename(columns={'site': 'Site'})
            presto_df['Site'] = presto_df['Site'].str.lower().str.strip()
        print(f"  Presto shape: {presto_df.shape}")
        print(f"  Presto sites: {presto_df['Site'].unique() if 'Site' in presto_df.columns else 'N/A'}")

    for site_name, filename in STATION_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping {site_name}")
            continue

        df = pd.read_csv(filepath, parse_dates=['Date'])
        df['Site'] = site_name.lower()
        df['Year'] = df['Date'].dt.year
        df['DOY'] = df['Date'].dt.dayofyear

        # Filter year range
        df = df[(df['Year'] >= YEAR_RANGE[0]) & (df['Year'] <= YEAR_RANGE[1])].copy()

        # Merge Presto if available
        if presto_df is not None and 'Site' in presto_df.columns:
            site_key = site_name.lower()
            for fmt in [site_key, f'us-{site_key}', f'US-{site_name}', site_name]:
                presto_site = presto_df[presto_df['Site'] == fmt.lower()]
                if len(presto_site) > 0:
                    df = df.merge(presto_site.drop(columns=['Site'], errors='ignore'),
                                  on='Date', how='left')
                    print(f"  Merged {len(presto_site)} Presto rows for {site_name} (key: {fmt})")
                    break
            else:
                print(f"  WARNING: No Presto data found for {site_name}")

        df = df.sort_values('Date').reset_index(drop=True)
        site_dfs[site_name] = df
        print(f"Loaded {site_name}: {len(df)} rows, {df['Year'].min()}-{df['Year'].max()}")

    return site_dfs


def get_combined_df(site_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all sites into a single DataFrame."""
    return pd.concat(site_dfs.values(), ignore_index=True)


def get_available_cols(df: pd.DataFrame, col_list: List[str]) -> List[str]:
    """Return only columns that exist in the dataframe."""
    return [c for c in col_list if c in df.columns]


def has_presto(site_dfs: Dict[str, pd.DataFrame]) -> bool:
    """Return True if any site DataFrame has Presto embedding columns."""
    return any(
        any(c in df.columns for c in PRESTO_COLS[:5])
        for df in site_dfs.values()
    )
