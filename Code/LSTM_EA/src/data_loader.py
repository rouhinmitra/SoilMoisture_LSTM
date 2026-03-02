"""
Data loading and preprocessing pipeline for EA-LSTM.
Matches the logic from prep_data.py exactly.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

from .config import DataConfig, FeatureConfig
from .features import FeatureEngineer
from .presto_embeddings import load_presto_embeddings, load_presto_interpolated_df, get_embedding, PRESTO_DIM

logger = logging.getLogger(__name__)


def _site_from_filepath(filepath: str) -> Optional[str]:
    """Infer site id (ne1, ne2, ne3) from file path or return None."""
    s = str(filepath).lower()
    if "ne1" in s:
        return "ne1"
    if "ne2" in s:
        return "ne2"
    if "ne3" in s:
        return "ne3"
    return None

# Precipitation columns (first available used for accumulated precip)
PRECIP_COLS = ["P_PI_F_1_1_1", "P_PI_F_2_2_1"]


def _add_annual_precip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add accumulated precipitation features (computed from full-year data):
    - precip_jan_apr: sum of daily precip Jan 1–Apr 30 for that year
    - precip_may_oct: sum of daily precip May 1–Oct 31 (summer) for that year
    """
    df = df.copy()
    if "Year" not in df.columns:
        df["Year"] = df["Date"].dt.year
    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.month
    if "Day" not in df.columns:
        df["Day"] = df["Date"].dt.day

    precip_col = None
    for c in PRECIP_COLS:
        if c in df.columns:
            precip_col = c
            break
    if precip_col is None:
        df["precip_jan_apr"] = 0.0
        df["precip_may_oct"] = 0.0
        return df

    jan_apr_mask = (df["Month"] < 4) | ((df["Month"] == 4) & (df["Day"] <= 30))
    summer_mask = (df["Month"] >= 5) & (df["Month"] <= 10)

    by_year = {}
    for year in df["Year"].unique():
        jy = df.loc[(df["Year"] == year) & jan_apr_mask, precip_col]
        sy = df.loc[(df["Year"] == year) & summer_mask, precip_col]
        by_year[year] = {
            "precip_jan_apr": jy.sum() if len(jy) else 0.0,
            "precip_may_oct": sy.sum() if len(sy) else 0.0,
        }

    df["precip_jan_apr"] = df["Year"].map(lambda y: by_year.get(y, {}).get("precip_jan_apr", 0.0))
    df["precip_may_oct"] = df["Year"].map(lambda y: by_year.get(y, {}).get("precip_may_oct", 0.0))
    return df


class DataProcessor:
    """
    Handles all data loading, cleaning, and windowing operations.
    Integrates with prep_data.py logic for sliding windows.
    """
    
    def __init__(self, data_config: DataConfig, feature_config: FeatureConfig):
        self.data_config = data_config
        self.feature_config = feature_config
        self.scaler_dyn = StandardScaler()
        self.scaler_stat = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self._irrigation_map = None
        self._validate_config()
        
    def _validate_config(self):
        """Validate that configuration is internally consistent"""
        if self.data_config.seq_length < 1:
            raise ValueError("Sequence length must be at least 1")
        if not 0 <= self.data_config.nan_threshold <= 1:
            raise ValueError("NaN threshold must be between 0 and 1")
    
    def _load_irrigation_data(self):
        """Load irrigation data from Excel file and create a mapping dictionary."""
        if self._irrigation_map is not None:
            return self._irrigation_map
        
        try:
            # Try to find Irrigation-data.xlsx relative to data_dir or script location
            script_dir = Path(__file__).parent.parent
            irrigation_file = script_dir / 'Irrigation-data.xlsx'
            
            if not irrigation_file.exists():
                # Try relative to data_dir
                irrigation_file = Path(self.data_config.data_dir).parent / 'Irrigation-data.xlsx'
            
            if not irrigation_file.exists():
                logger.warning(f"Irrigation-data.xlsx not found, irrigation column will be set to 0")
                self._irrigation_map = {}
                return self._irrigation_map
            
            irrigation_df = pd.read_excel(irrigation_file)
            irrigation_map = {}
            
            # Map US-Ne1 to ne1
            if 'US-Ne1' in irrigation_df.columns and 'Year' in irrigation_df.columns:
                irrigation_map['ne1'] = dict(zip(irrigation_df['Year'], irrigation_df['US-Ne1']))
            
            # Map US-Ne2 to ne2
            if 'US-Ne2' in irrigation_df.columns and 'Year' in irrigation_df.columns:
                irrigation_map['ne2'] = dict(zip(irrigation_df['Year'], irrigation_df['US-Ne2']))
            
            self._irrigation_map = irrigation_map
            return irrigation_map
        except Exception as e:
            logger.warning(f"Error loading irrigation data: {e}, irrigation column will be set to 0")
            self._irrigation_map = {}
            return self._irrigation_map
    
    def load_and_clean(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load and clean a single CSV file with error handling.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file (relative to data_dir)
        
        Returns
        -------
        pd.DataFrame or None
            Cleaned DataFrame or None if loading failed
        """
        try:
            # Construct full path
            full_path = Path(self.data_config.data_dir) / filepath
            
            if not full_path.exists():
                logger.error(f"File not found: {full_path}")
                return None
            
            # Load data
            df = pd.read_csv(full_path)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            
            # Get all required columns
            dynamic_cols = self.feature_config.get_all_dynamic_cols()
            static_cols = self.feature_config.static_cols
            target_col = self.feature_config.target_col
            required_cols = dynamic_cols + static_cols + [target_col, 'Date']
            
            # Validate required columns (exclude irrigation, computed precip, and Presto emb_* merged later)
            computed_static = {'precip_jan_apr', 'precip_may_oct'}
            base_static = [c for c in static_cols if c not in computed_static and not (c.startswith('emb_'))]
            base_required = [col for col in self.feature_config.dynamic_cols if col != 'irrigation'] + base_static + [target_col, 'Date']
            missing_cols = set(base_required) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns in {filepath}: {missing_cols}")
                return None
            
            # Parse dates and add Site for Presto daily merge (from filepath: ne1, ne2, ne3)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Site'] = _site_from_filepath(filepath)
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in {filepath}, dropping them")
                df = df.dropna(subset=['Date'])
            
            if len(df) == 0:
                logger.error(f"No valid data remaining in {filepath}")
                return None
            
            # Sort and remove duplicates
            df = df.sort_values('Date').reset_index(drop=True)
            dup_count = df.duplicated(subset='Date').sum()
            if dup_count > 0:
                logger.warning(f"Removing {dup_count} duplicate dates from {filepath}")
                df = df.drop_duplicates(subset='Date', keep='first')

            # Add Year and accumulated precipitation (from full-year data) before month filter
            df['Year'] = df['Date'].dt.year
            df = _add_annual_precip_features(df)

            # Optional: restrict to a range of months (e.g. May–October)
            if self.data_config.month_range is not None:
                start_m, end_m = self.data_config.month_range
                n_before = len(df)
                df = df[(df['Date'].dt.month >= start_m) & (df['Date'].dt.month <= end_m)].copy()
                df = df.reset_index(drop=True)
                logger.info(f"  Filtered to months {start_m}-{end_m}: {len(df)} rows (dropped {n_before - len(df)})")

            if len(df) == 0:
                logger.warning(f"No rows left in {filepath} after month filter")
                return None

            # Optional: restrict to a range of years (e.g. 2017–2024)
            if getattr(self.data_config, 'year_range', None) is not None:
                start_year, end_year = self.data_config.year_range
                n_before = len(df)
                df = df[(df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)].copy()
                df = df.reset_index(drop=True)
                logger.info(f"  Filtered to years {start_year}-{end_year}: {len(df)} rows (dropped {n_before - len(df)})")
                if len(df) == 0:
                    logger.warning(f"No rows left in {filepath} after year filter")
                    return None

            # Always add irrigation column (can be used as dynamic or static feature)
            irrigation_map = self._load_irrigation_data()
            if 'Year' not in df.columns:
                df['Year'] = df['Date'].dt.year
            # Identify site name from filepath (e.g., 'ne1', 'ne2', 'ne3')
            site_name = None
            filepath_lower = filepath.lower()
            if 'ne1' in filepath_lower:
                site_name = 'ne1'
            elif 'ne2' in filepath_lower:
                site_name = 'ne2'
            elif 'ne3' in filepath_lower:
                site_name = 'ne3'
            else:
                # Try to get from Name column if available
                if 'Name' in df.columns and len(df) > 0:
                    name_value = str(df['Name'].iloc[0]).lower()
                    if 'ne1' in name_value:
                        site_name = 'ne1'
                    elif 'ne2' in name_value:
                        site_name = 'ne2'
                    elif 'ne3' in name_value:
                        site_name = 'ne3'
            
            # Add irrigation column
            if site_name == 'ne3':
                # For ne3, set all irrigation values to 0
                df['irrigation'] = 0.0
            elif site_name in irrigation_map:
                # For ne1 and ne2, map irrigation values by year
                df['irrigation'] = df['Year'].map(irrigation_map[site_name]).fillna(0.0)
            else:
                # If site not identified or not in map, set to 0
                df['irrigation'] = 0.0
            
            # Ensure irrigation is numeric
            df['irrigation'] = pd.to_numeric(df['irrigation'], errors='coerce').fillna(0.0)
            
            # Add temporal features if requested
            if self.feature_config.add_temporal:
                try:
                    df = self.feature_engineer.add_temporal_features(
                        df, self.feature_config.temporal_features
                    )
                except Exception as e:
                    logger.error(f"Error adding temporal features: {e}")
                    return None
            
            logger.info(f"Successfully processed {len(df)} rows from {filepath}")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {filepath}: {e}", exc_info=True)
            return None
    
    def create_sliding_windows(
        self, 
        df: pd.DataFrame, 
        dynamic_cols: List[str],
        static_cols: List[str],
        target_col: str,
        return_dates: bool = False,
        *,
        site_name: Optional[str] = None,
        presto_lookup: Optional[Dict] = None,
        scaler_stat: Optional[StandardScaler] = None,
    ):
        """
        Create sliding windows EXACTLY matching prep_data.py behavior:
        - Check NaN percentage in each window (threshold = 10% by default)
        - Interpolate if under threshold
        - Return full sequences for X_dynamic, X_static, and y
        - If presto_lookup is provided, static = Presto(site,year) + static_cols (e.g. precip), then scaled.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame
        dynamic_cols : List[str]
            Dynamic feature column names
        static_cols : List[str]
            Static feature column names (when Presto: only precip_jan_apr, precip_may_oct)
        target_col : str
            Target column name
        return_dates : bool
            If True, also return array of last date per window (target date).
        site_name : str, optional
            Site id (e.g. ne1) when using Presto embeddings.
        presto_lookup : dict, optional
            (site, year) -> 128-d array from load_presto_embeddings().
        scaler_stat : StandardScaler, optional
            When using Presto, used to scale the composed static vector (presto + precip).

        Returns
        -------
        If return_dates=False: Tuple[np.ndarray, np.ndarray, np.ndarray]
        If return_dates=True: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_dynamic, X_static, y [, dates]
        """
        use_presto = presto_lookup is not None
        if use_presto and (site_name is None or scaler_stat is None):
            raise ValueError("When presto_lookup is set, site_name and scaler_stat must be provided")
        
        if df is None or len(df) == 0:
            logger.warning("Empty dataframe provided to create_sliding_windows")
            out = (np.array([]), np.array([]), np.array([]))
            if return_dates:
                out = out + (np.array([], dtype='datetime64[ns]'),)
            return out
        
        seq_length = self.data_config.seq_length
        nan_threshold = self.data_config.nan_threshold  # Default 0.1 (10%)
        
        # Check if we have enough data
        if len(df) < seq_length:
            logger.warning(f"Dataframe has {len(df)} rows, need at least {seq_length}")
            out = (np.array([]), np.array([]), np.array([]))
            if return_dates:
                out = out + (np.array([], dtype='datetime64[ns]'),)
            return out
        
        # Validate columns exist
        all_cols = dynamic_cols + static_cols + [target_col]
        feature_cols = dynamic_cols + static_cols  # target is never interpolated
        missing = set(all_cols) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            out = (np.array([]), np.array([]), np.array([]))
            if return_dates:
                out = out + (np.array([], dtype='datetime64[ns]'),)
            return out
        
        df = df.copy()
        df = df.reset_index(drop=True)
        df['Year'] = df['Date'].dt.year
        
        all_windows = []
        all_dates = []  # last date of each window (target date)
        skipped_year = 0
        skipped_nan = 0
        skipped_target_nan = 0
        interpolated_count = 0
        
        for i in range(len(df) - seq_length + 1):
            window = df.iloc[i:i+seq_length].copy()
            
            # Check if all dates are from the same year
            if self.data_config.same_year_constraint and window['Year'].nunique() > 1:
                skipped_year += 1
                continue
            
            # Drop window if target has any NaN (no interpolation for target)
            if window[target_col].isna().any():
                skipped_target_nan += 1
                continue
            
            # Check NaN percentage for each feature column (not target)
            drop_sequence = False
            for col in feature_cols:
                try:
                    nan_pct = window[col].isna().sum() / seq_length
                    if nan_pct > nan_threshold:
                        drop_sequence = True
                        break
                except Exception as e:
                    logger.warning(f"Error checking NaN for column {col}: {e}")
                    drop_sequence = True
                    break
            
            if drop_sequence:
                skipped_nan += 1
                continue
            
            # Interpolate missing values in features only (never interpolate target)
            has_nan = window[feature_cols].isna().any().any()
            if has_nan:
                try:
                    window_interpolated = window.copy()
                    for col in feature_cols:
                        if window[col].isna().any():
                            window_interpolated[col] = window[col].interpolate(
                                method='linear', 
                                limit_direction='both'
                            )
                            # Final check - if still NaN, use forward/backward fill
                            if window_interpolated[col].isna().any():
                                window_interpolated[col] = (window_interpolated[col]
                                                           .ffill()
                                                           .bfill())
                    
                    # Final validation - no NaNs should remain in features
                    if window_interpolated[feature_cols].isna().any().any():
                        logger.debug(f"Window at index {i} still contains NaN in features after interpolation")
                        skipped_nan += 1
                        continue
                    
                    interpolated_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error interpolating window at index {i}: {e}")
                    skipped_nan += 1
                    continue
            else:
                window_interpolated = window
            
            # Extract features and target (target is raw, never interpolated)
            try:
                X_dynamic = window_interpolated[dynamic_cols].values  # (seq_length, num_dyn_features)
                y = window[target_col].values  # (seq_length,) - use raw target, no interpolation
                if use_presto:
                    year = int(window_interpolated['Year'].iloc[0])
                    presto_128 = get_embedding(presto_lookup, site_name, year)
                    # (seq_length, 128) then (seq_length, len(static_cols)) = Alpha Earth + precip
                    presto_broadcast = np.broadcast_to(presto_128, (seq_length, PRESTO_DIM))
                    static_part = window_interpolated[static_cols].values  # (seq_length, len(static_cols))
                    X_static_base = np.hstack([presto_broadcast, static_part])  # (seq_length, 128 + len(static_cols))
                    X_static_base = scaler_stat.transform(X_static_base)
                else:
                    X_static_base = window_interpolated[static_cols].values    # (seq_length, num_stat_features)
                
                # Check if irrigation should be added as static feature
                irrigation_as_static = False
                exclude_irr = getattr(self.feature_config, 'exclude_irrigation_static', False)
                if 'irrigation' in window_interpolated.columns and 'irrigation' not in dynamic_cols and not exclude_irr:
                    irrigation_as_static = True
                    irrigation_value = window_interpolated['irrigation'].iloc[0]
                    irrigation_array = np.full((seq_length, 1), irrigation_value)
                    X_static = np.concatenate([X_static_base, irrigation_array], axis=1)
                else:
                    X_static = X_static_base
                
                # Validate shapes
                if X_dynamic.shape != (seq_length, len(dynamic_cols)):
                    logger.warning(f"Unexpected dynamic feature shape: {X_dynamic.shape}")
                    continue
                expected_static_dim = X_static_base.shape[1] + (1 if irrigation_as_static else 0)
                if X_static.shape != (seq_length, expected_static_dim):
                    logger.warning(f"Unexpected static feature shape: {X_static.shape}, expected ({seq_length}, {expected_static_dim})")
                    continue
                if y.shape != (seq_length,):
                    logger.warning(f"Unexpected target shape: {y.shape}")
                    continue
                
                all_windows.append((X_dynamic, X_static, y))
                if return_dates:
                    all_dates.append(window_interpolated['Date'].iloc[-1])
                
            except Exception as e:
                logger.warning(f"Error extracting features at index {i}: {e}")
                continue
        
        # Log statistics
        logger.info(f"Created {len(all_windows)} windows from {len(df)} rows")
        if skipped_year > 0:
            logger.info(f"  Skipped {skipped_year} windows (different years)")
        if skipped_target_nan > 0:
            logger.info(f"  Skipped {skipped_target_nan} windows (target had NaN)")
        if skipped_nan > 0:
            logger.info(f"  Skipped {skipped_nan} windows (features NaN > {nan_threshold*100:.0f}%)")
        if interpolated_count > 0:
            logger.info(f"  Interpolated {interpolated_count} windows (features only; target never interpolated)")
        
        if not all_windows:
            logger.warning("No valid windows created!")
            out = (np.array([]), np.array([]), np.array([]))
            if return_dates:
                out = out + (np.array([], dtype='datetime64[ns]'),)
            return out
        
        try:
            X_d, X_s, y = zip(*all_windows)
            out = (np.array(X_d), np.array(X_s), np.array(y))
            if return_dates:
                dates_arr = pd.to_datetime(all_dates).to_numpy(dtype='datetime64[ns]')
                out = out + (dates_arr,)
            return out
        except Exception as e:
            logger.error(f"Error converting windows to arrays: {e}")
            out = (np.array([]), np.array([]), np.array([]))
            if return_dates:
                out = out + (np.array([], dtype='datetime64[ns]'),)
            return out
    
    def prepare_data(
        self, 
        train_files: List[str], 
        test_files: List[str],
        fit_scalers: bool = True
    ) -> Dict:
        """
        Complete data preparation pipeline with comprehensive error handling.
        
        Parameters
        ----------
        train_files : List[str]
            List of training file paths (relative to data_dir)
        test_files : List[str]
            List of test file paths (relative to data_dir)
        fit_scalers : bool
            Whether to fit scalers (True for new training, False for inference)
        
        Returns
        -------
        Dict
            Dictionary containing:
            - X_d_train, X_s_train, y_train: Training data
            - X_d_test, X_s_test, y_test: Test data
        """
        
        logger.info("="*60)
        logger.info("Starting data preparation pipeline...")
        logger.info(f"NaN threshold: {self.data_config.nan_threshold*100:.0f}% per column per window")
        logger.info("="*60)
        
        # Get dynamic columns (add temporal if needed)
        dynamic_cols = self.feature_config.get_all_dynamic_cols()
        static_cols = self.feature_config.static_cols
        target_col = self.feature_config.target_col
        use_presto_static = getattr(self.feature_config, 'use_presto_static', False)
        presto_df_daily = None  # daily Presto merged into dfs; when set we do not use presto_lookup
        if use_presto_static:
            path = Path(self.feature_config.presto_embeddings_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            presto_df_daily = load_presto_interpolated_df(str(path))
            logger.info(f"Using Presto embeddings at daily scale (merge on Date+Site); static = Presto 128-d + AE + precip")
        
        # Check if irrigation should be handled as static (if it's not in dynamic_cols)
        exclude_irr = getattr(self.feature_config, 'exclude_irrigation_static', False)
        irrigation_as_static = 'irrigation' not in dynamic_cols and not exclude_irr
        
        logger.info(f"Features: {len(dynamic_cols)} dynamic, {len(static_cols)} static" + (
            " (Presto daily merged)" if use_presto_static else ""
        ))
        if irrigation_as_static:
            logger.info("Irrigation will be added as static feature (input gate control)")
        
        # Load all dataframes
        logger.info(f"\nLoading {len(train_files)} training files...")
        train_dfs = []
        for f in train_files:
            df = self.load_and_clean(f)
            if df is not None and len(df) > 0:
                train_dfs.append(df)
            else:
                logger.warning(f"Failed to load training file: {f}")
        
        if not train_dfs:
            raise ValueError("No valid training data loaded!")
        
        logger.info(f"\nLoading {len(test_files)} test files...")
        test_dfs = []
        for f in test_files:
            df = self.load_and_clean(f)
            if df is not None and len(df) > 0:
                test_dfs.append(df)
            else:
                logger.warning(f"Failed to load test file: {f}")
        
        if not test_dfs:
            raise ValueError("No valid test data loaded!")

        # Merge daily Presto into each dataframe on (Date, Site) when using daily Presto
        if presto_df_daily is not None:
            for i, df in enumerate(train_dfs):
                if "Site" not in df.columns or df["Site"].isna().all():
                    logger.warning(f"Train file {train_files[i]} has no Site; skipping Presto merge")
                    continue
                train_dfs[i] = df.merge(presto_df_daily, on=["Date", "Site"], how="left")
            for i, df in enumerate(test_dfs):
                if "Site" not in df.columns or df["Site"].isna().all():
                    logger.warning(f"Test file {test_files[i]} has no Site; skipping Presto merge")
                    continue
                test_dfs[i] = df.merge(presto_df_daily, on=["Date", "Site"], how="left")
            logger.info("Merged daily Presto embeddings into train and test dataframes")
        
        # Fit scalers on training data
        if fit_scalers:
            logger.info("\nFitting scalers on training data...")
            try:
                train_concat = pd.concat(train_dfs, ignore_index=True)
                
                # Check for infinite values
                cols_for_inf = dynamic_cols + static_cols + [target_col]
                for col in cols_for_inf:
                    if col in train_concat.columns:
                        inf_count = np.isinf(train_concat[col]).sum()
                        if inf_count > 0:
                            logger.warning(f"Found {inf_count} infinite values in {col}, replacing with NaN")
                            train_concat[col] = train_concat[col].replace([np.inf, -np.inf], np.nan)
                
                self.scaler_dyn.fit(train_concat[dynamic_cols].dropna())
                
                # Static scaler: when Presto daily merged, static_cols include emb_* and are in train_concat
                # Skip when no static columns (e.g. meteo_precip_only with empty static_cols)
                static_cols_for_scaling = static_cols.copy()
                if irrigation_as_static and 'irrigation' in train_concat.columns:
                    static_cols_for_scaling = static_cols + ['irrigation']
                if static_cols_for_scaling:
                    self.scaler_stat.fit(train_concat[static_cols_for_scaling].dropna())
                
                self.scaler_y.fit(train_concat[[target_col]].dropna())
                
                logger.info("Scalers fitted successfully (StandardScaler - zero mean, unit variance)")
                logger.info(f"  Target: mean={self.scaler_y.mean_[0]:.4f}, std={self.scaler_y.scale_[0]:.4f}")
            except Exception as e:
                logger.error(f"Error fitting scalers: {e}", exc_info=True)
                raise
        
        # Process training data
        logger.info("\nProcessing training data...")
        X_d_train_list, X_s_train_list, y_train_list = [], [], []
        for i, df in enumerate(train_dfs):
            try:
                df_scaled = df.copy()
                df_scaled[dynamic_cols] = self.scaler_dyn.transform(df[dynamic_cols])
                static_cols_for_scaling = static_cols.copy()
                if irrigation_as_static and 'irrigation' in df.columns:
                    static_cols_for_scaling = static_cols + ['irrigation']
                # Fill NaN in static (e.g. Presto merge misses) so scaler.transform does not fail
                if static_cols_for_scaling:
                    static_vals = df[static_cols_for_scaling].fillna(0.0)
                    df_scaled[static_cols_for_scaling] = self.scaler_stat.transform(static_vals)
                df_scaled[target_col] = self.scaler_y.transform(df[[target_col]])
                
                xd, xs, y = self.create_sliding_windows(df_scaled, dynamic_cols, static_cols, target_col)
                if len(xd) > 0:
                    X_d_train_list.append(xd)
                    X_s_train_list.append(xs)
                    y_train_list.append(y)
                    logger.info(f"  File {i+1} ({train_files[i]}): {len(xd)} windows")
            except Exception as e:
                logger.error(f"Error processing training file {i+1}: {e}", exc_info=True)
                continue
        
        if not X_d_train_list:
            raise ValueError("No valid training samples created!")
        
        # Process test data (with dates for time series plotting)
        logger.info("\nProcessing test data...")
        X_d_test_list, X_s_test_list, y_test_list, dates_test_list = [], [], [], []
        for i, df in enumerate(test_dfs):
            try:
                df_scaled = df.copy()
                df_scaled[dynamic_cols] = self.scaler_dyn.transform(df[dynamic_cols])
                static_cols_for_scaling = static_cols.copy()
                if irrigation_as_static and 'irrigation' in df.columns:
                    static_cols_for_scaling = static_cols + ['irrigation']
                if static_cols_for_scaling:
                    static_vals = df[static_cols_for_scaling].fillna(0.0)
                    df_scaled[static_cols_for_scaling] = self.scaler_stat.transform(static_vals)
                df_scaled[target_col] = self.scaler_y.transform(df[[target_col]])
                
                out = self.create_sliding_windows(df_scaled, dynamic_cols, static_cols, target_col, return_dates=True)
                xd, xs, y = out[0], out[1], out[2]
                dates_test_list.append(out[3])
                if len(xd) > 0:
                    X_d_test_list.append(xd)
                    X_s_test_list.append(xs)
                    y_test_list.append(y)
                    logger.info(f"  File {i+1} ({test_files[i]}): {len(xd)} windows")
            except Exception as e:
                logger.error(f"Error processing test file {i+1}: {e}", exc_info=True)
                continue
        
        if not X_d_test_list:
            raise ValueError("No valid test samples created!")
        
        result = {
            'X_d_train': np.concatenate(X_d_train_list),
            'X_s_train': np.concatenate(X_s_train_list),
            'y_train': np.concatenate(y_train_list),
            'X_d_test': np.concatenate(X_d_test_list),
            'X_s_test': np.concatenate(X_s_test_list),
            'y_test': np.concatenate(y_test_list),
            'dates_test': np.concatenate(dates_test_list),
        }
        
        logger.info("\n" + "="*60)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Training samples: {len(result['y_train'])}")
        logger.info(f"Test samples: {len(result['y_test'])}")
        logger.info(f"Shapes:")
        logger.info(f"  X_d_train: {result['X_d_train'].shape}")
        logger.info(f"  X_s_train: {result['X_s_train'].shape}")
        logger.info(f"  y_train: {result['y_train'].shape}")
        
        return result


class RZSMDataset(Dataset):
    """
    PyTorch Dataset for RZSM (Root Zone Soil Moisture) data.
    
    Parameters
    ----------
    x_d : np.ndarray
        Dynamic features, shape (num_samples, seq_length, num_dyn_features)
    x_s : np.ndarray
        Static features, shape (num_samples, seq_length, num_stat_features)
    y : np.ndarray
        Target values, shape (num_samples, seq_length)
    """
    
    def __init__(self, x_d: np.ndarray, x_s: np.ndarray, y: np.ndarray):
        if len(x_d) == 0 or len(x_s) == 0 or len(y) == 0:
            raise ValueError("Cannot create dataset from empty arrays")
        
        if len(x_d) != len(x_s) or len(x_d) != len(y):
            raise ValueError(f"Mismatched lengths: x_d={len(x_d)}, x_s={len(x_s)}, y={len(y)}")
        
        self.x_d = torch.tensor(x_d, dtype=torch.float32)
        self.x_s = torch.tensor(x_s, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Check for NaN or Inf
        if torch.isnan(self.x_d).any() or torch.isinf(self.x_d).any():
            raise ValueError("x_d contains NaN or Inf values")
        if torch.isnan(self.x_s).any() or torch.isinf(self.x_s).any():
            raise ValueError("x_s contains NaN or Inf values")
        if torch.isnan(self.y).any() or torch.isinf(self.y).any():
            raise ValueError("y contains NaN or Inf values")
        
        logger.info(f"Dataset created with {len(self)} samples")
        logger.info(f"  x_d shape: {tuple(self.x_d.shape)}")
        logger.info(f"  x_s shape: {tuple(self.x_s.shape)}")
        logger.info(f"  y shape: {tuple(self.y.shape)}")
        
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_d[i], self.x_s[i], self.y[i]


