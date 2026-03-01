"""
Feature engineering utilities for EA-LSTM.
Handles temporal features, lagged features, and rolling statistics.
"""
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles all feature engineering operations"""
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Add temporal features to dataframe.
        
        Uses cyclical encoding (sin/cos) to capture periodic nature of time.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'Date' column
        features : List[str]
            List of temporal features to add. Options:
            - 'doy_sin', 'doy_cos': cyclical day of year
            - 'month_sin', 'month_cos': cyclical month
            - 'day_of_year': raw day of year (1-365)
            - 'month': raw month (1-12)
            - 'season': season indicator (1-4)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added temporal features
        """
        df = df.copy()
        
        if 'Date' not in df.columns:
            raise ValueError("DataFrame must have 'Date' column")
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Day of year features
        if 'doy_sin' in features or 'doy_cos' in features or 'day_of_year' in features:
            df['day_of_year'] = df['Date'].dt.dayofyear
            
            if 'doy_sin' in features:
                df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            if 'doy_cos' in features:
                df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            
            # Remove raw day_of_year if not requested
            if 'day_of_year' not in features:
                df = df.drop(columns=['day_of_year'])
        
        # Month features
        if 'month_sin' in features or 'month_cos' in features or 'month' in features:
            month = df['Date'].dt.month
            
            if 'month_sin' in features:
                df['month_sin'] = np.sin(2 * np.pi * month / 12)
            if 'month_cos' in features:
                df['month_cos'] = np.cos(2 * np.pi * month / 12)
            if 'month' in features:
                df['month'] = month
        
        # Season indicator (1=winter, 2=spring, 3=summer, 4=fall)
        if 'season' in features:
            df['season'] = (df['Date'].dt.month % 12 + 3) // 3
        
        # Week of year
        if 'week_sin' in features or 'week_cos' in features:
            week = df['Date'].dt.isocalendar().week
            if 'week_sin' in features:
                df['week_sin'] = np.sin(2 * np.pi * week / 52)
            if 'week_cos' in features:
                df['week_cos'] = np.cos(2 * np.pi * week / 52)
        
        logger.info(f"Added temporal features: {[f for f in features if f in df.columns]}")
        return df
    
    @staticmethod
    def create_lagged_features(
        df: pd.DataFrame, 
        cols: List[str], 
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        cols : List[str]
            Columns to create lags for
        lags : List[int]
            Lag values (e.g., [1, 7, 14] for 1-day, 1-week, 2-week lags)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added lag features
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        logger.info(f"Created {len(cols) * len(lags)} lagged features")
        return df
    
    @staticmethod
    def create_rolling_features(
        df: pd.DataFrame, 
        cols: List[str], 
        windows: List[int],
        stats: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        cols : List[str]
            Columns to create rolling stats for
        windows : List[int]
            Window sizes (e.g., [7, 14, 30] for 1-week, 2-week, 1-month)
        stats : List[str]
            Statistics to compute. Options: 'mean', 'std', 'min', 'max', 'median'
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added rolling features
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            for window in windows:
                rolling = df[col].rolling(window=window, min_periods=1)
                
                if 'mean' in stats:
                    df[f'{col}_roll{window}_mean'] = rolling.mean()
                if 'std' in stats:
                    df[f'{col}_roll{window}_std'] = rolling.std()
                if 'min' in stats:
                    df[f'{col}_roll{window}_min'] = rolling.min()
                if 'max' in stats:
                    df[f'{col}_roll{window}_max'] = rolling.max()
                if 'median' in stats:
                    df[f'{col}_roll{window}_median'] = rolling.median()
        
        logger.info(f"Created rolling features for {len(cols)} columns with windows {windows}")
        return df
    
    @staticmethod
    def create_diff_features(
        df: pd.DataFrame,
        cols: List[str],
        periods: List[int] = [1]
    ) -> pd.DataFrame:
        """
        Create difference features (rate of change).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        cols : List[str]
            Columns to create differences for
        periods : List[int]
            Difference periods (default [1] for daily change)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added difference features
        """
        df = df.copy()
        
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            for period in periods:
                df[f'{col}_diff{period}'] = df[col].diff(periods=period)
        
        logger.info(f"Created difference features for {len(cols)} columns")
        return df


