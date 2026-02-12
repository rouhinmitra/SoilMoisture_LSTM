#%%
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
#%%
SEQ_LENGTH = 30       # Lookback window (days)
HIDDEN_DIM = 32       # Hidden state size
DROPOUT = 0.4         # Regularization
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Feature Definitions
DYN_COLS = ['SSM', 'SSM_avg', 'SWC_PI_F_2_1_1', 'SWC_PI_F_3_1_1', 'P_PI_F_1_1_1', 'P_PI_F_2_2_1', 'I', 'TA_1_1_1', 'RH_1_1_1'] # Dynamic inputs
STAT_COLS = [f'A{i:02d}' for i in range(64)]                # Alpha Earth A00-A63
PRESTO_STAT_COLS = [f'emb_{k}' for k in range(128)]         # Presto 128-d embeddings
TARGET_COL = 'RZSM_25_avg'                                      # Target

# Presto embeddings (interpolated daily per site)
USE_PRESTO_STATIC = True   # Set False to use STAT_COLS (A00-A63) instead
_script_dir = os.path.dirname(os.path.abspath(__file__))
PRESTO_EMBED_PATH = os.path.join(_script_dir, '..', 'Data', 's2_pixels', 'presto_embeddings_fused_interpolated.csv')


def _site_from_filename(file_name):
    """Infer site id from filename for merge with Presto (e.g. US-Ne1)."""
    s = str(file_name).lower()
    if 'ne1' in s:
        return 'US-Ne1'
    if 'ne2' in s:
        return 'US-Ne2'
    if 'ne3' in s:
        return 'US-Ne3'
    return None


def load_presto_interpolated(path=None):
    """Load Presto embeddings interpolated to daily (Date, Site, emb_0..emb_127)."""
    path = path or PRESTO_EMBED_PATH
    presto = pd.read_csv(path)
    presto['Date'] = pd.to_datetime(presto['Date'])
    # Normalize Site to match (e.g. US-Ne1)
    if 'Site' in presto.columns:
        presto['Site'] = presto['Site'].astype(str).str.strip()
    return presto


def merge_presto_into_dfs(dfs, presto_df):
    """Merge Presto embeddings into each dataframe on Date and Site (left join)."""
    emb_cols = [c for c in presto_df.columns if c.startswith('emb_')]
    merged = []
    for df in dfs:
        df = df.copy()
        if 'Site' not in df.columns or df['Site'].isna().all():
            merged.append(df)
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        before = len(df)
        df = df.merge(
            presto_df[['Date', 'Site'] + emb_cols],
            on=['Date', 'Site'],
            how='left'
        )
        merged.append(df)
    return merged


def load_and_clean(file_path):
    """Load csv files from the directory. Adds Site from filename for Presto merge."""
    import os
    files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(file_path, file))
        df['Date'] = pd.to_datetime(df['Date'])
        site = _site_from_filename(file)
        if site is not None:
            df['Site'] = site
        elif 'Name' in df.columns:
            df['Site'] = str(df['Name'].iloc[0]).strip()
        else:
            df['Site'] = None
        dfs.append(df)
    return dfs

def create_sliding_windows(dfs, dynamic_cols, static_cols, target_col, seq_length = 15,nan_threshold = 0.1):
    """Create sliding windows for each dataframe, make sure the window has data from same year only 
    and dates are not missing"""

    all_windows = []

    for df in dfs:
        df = df.sort_values(by='Date').copy()
        df = df.dropna(subset=dynamic_cols + static_cols + [target_col])
        df = df.drop_duplicates(subset='Date')
        df = df.reset_index(drop=True)
        
        # add year column 
        df['Year'] = df['Date'].dt.year

        # All columns we want 
        all_cols = dynamic_cols + static_cols + [target_col]

        # Sliding through the dataframe 
        for i in range(len(df) - seq_length + 1):
            window = df.iloc[i:i+seq_length].copy()

            # Check if all dates are from the same year 
            if window["Year"].nunique() > 1:
                continue
            
            # Check NaN percentage for each column in this window 
            drop_sequence = False 
            for col in all_cols:
                nan_pct = window[col].isna().sum() / len(window)
                if nan_pct > nan_threshold:
                    drop_sequence = True
                    break
            if drop_sequence:
                continue
            # Interpolate missing values 
            window_interpolated = window.copy()
            for col in all_cols:
                if window_interpolated[col].isna().any():
                    window_interpolated[col] = window_interpolated[col].interpolate(method='linear',limit_direction='both')
            
            X_dynamic = window_interpolated[dynamic_cols].values
            X_static = window_interpolated[static_cols].values
            y = window_interpolated[target_col].values

            # Append to the list 
            all_windows.append({
                'X_dynamic': X_dynamic,
                'X_static': X_static,
                'y': y,
                'Date': window_interpolated['Date'].iloc[0]
            })
    return all_windows
# %%
if __name__ == "__main__":
    file_path = '/Users/rouhinmitra/SM_work/Data/Base/'
    dfs = load_and_clean(file_path)
    print(dfs)

    # Optionally merge Presto embeddings (daily per site) and use as static features
    static_cols = PRESTO_STAT_COLS if USE_PRESTO_STATIC else STAT_COLS
    if USE_PRESTO_STATIC:
        import os
        if os.path.exists(PRESTO_EMBED_PATH):
            presto_df = load_presto_interpolated()
            dfs = merge_presto_into_dfs(dfs, presto_df)
            print(f"Merged Presto embeddings; using {len(static_cols)} static cols (emb_0..emb_127)")
        else:
            print(f"PRESTO_EMBED_PATH not found: {PRESTO_EMBED_PATH}, falling back to STAT_COLS (A00-A63)")
            static_cols = STAT_COLS

    lstm_data = create_sliding_windows(dfs, DYN_COLS, static_cols, TARGET_COL, seq_length=SEQ_LENGTH)
    print(lstm_data)
#%%
for df in dfs:
    print(df.Name.iloc[0])
    print(df.shape[0])
    print(df.Date.iloc[0])
    print(df.columns.tolist())
# %%
lstm_data
# %%
