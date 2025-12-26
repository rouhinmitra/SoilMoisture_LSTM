#%%
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
TARGET_COL = 'RZSM_25_avg'                                      # Target

def load_and_clean(file_path):
    """Load csv files from the directory"""
    import os
    files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
    dfs = []
    for file in files:  #
        df = pd.read_csv(os.path.join(file_path, file))
        df['Date'] = pd.to_datetime(df['Date'])
        dfs.append(df)
    return dfs

def create_sliding_windows(dfs, dynamic_cols, static_cols, target_col, seq_length = 15,nan_threshold = 0.1):
    """Create sliding windows for each dataframe, make sure the window has data from same year only 
    and dates are not missing"""

    all_windows = []

    for df in dfs:
        df = df.sort_values(by='Date').copy()
        df = df.dropna(subset=DYN_COLS + STAT_COLS + [TARGET_COL])
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
    
    lstm_data = create_sliding_windows(dfs, DYN_COLS, STAT_COLS, TARGET_COL)
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
