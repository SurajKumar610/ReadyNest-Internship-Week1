import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def perform_label_encoding(df, columns):
    """
    Apply label encoding to categorical columns.
    """
    engineered_df = df.copy()
    encoders = {}
    for col in columns:
        if col not in engineered_df.columns:
            continue
        le = LabelEncoder()
        # Handle nulls by converting to string first
        engineered_df[col] = engineered_df[col].astype(str)
        engineered_df[col] = le.fit_transform(engineered_df[col])
        encoders[col] = le
    return engineered_df, encoders

def perform_one_hot_encoding(df, columns, drop_first=True):
    """
    Apply one-hot encoding to categorical columns.
    """
    # Ensure they exist in df
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return df.copy()
    return pd.get_dummies(df, columns=valid_cols, drop_first=drop_first)

def scale_features(df, columns, method='standard'):
    """
    Scale numerical features.
    method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    """
    engineered_df = df.copy()
    if not columns:
        return engineered_df, None
        
    valid_cols = [col for col in columns if col in engineered_df.columns and pd.api.types.is_numeric_dtype(engineered_df[col])]
    if not valid_cols:
        return engineered_df, None
        
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    engineered_df[valid_cols] = scaler.fit_transform(engineered_df[valid_cols].fillna(0))
    return engineered_df, scaler

def add_date_features(df, date_column):
    """
    Extract useful date components from a date column.
    """
    engineered_df = df.copy()
    if date_column not in engineered_df.columns:
        return engineered_df
        
    # Ensure it's datetime
    if not pd.api.types.is_datetime64_any_dtype(engineered_df[date_column]):
        engineered_df[date_column] = pd.to_datetime(engineered_df[date_column], errors='coerce')
        
    # Extract features
    prefix = date_column + "_"
    engineered_df[prefix + 'year'] = engineered_df[date_column].dt.year
    engineered_df[prefix + 'quarter'] = engineered_df[date_column].dt.quarter
    engineered_df[prefix + 'month'] = engineered_df[date_column].dt.month
    engineered_df[prefix + 'day'] = engineered_df[date_column].dt.day
    engineered_df[prefix + 'dayofweek'] = engineered_df[date_column].dt.dayofweek
    engineered_df[prefix + 'is_weekend'] = engineered_df[prefix + 'dayofweek'].isin([5, 6]).astype(int)
    
    return engineered_df

def create_derived_features(df, formula_type, **kwargs):
    """
    Calculate common derived metrics.
    formula_type: 'profit_margin' (needs 'profit' and 'revenue' columns)
                  'growth_rate' (needs 'current' and 'previous' values)
                  'custom_ratio' (needs 'num_col' and 'den_col')
    """
    engineered_df = df.copy()
    if formula_type == 'profit_margin':
        profit_col = kwargs.get('profit', 'profit')
        revenue_col = kwargs.get('revenue', 'revenue')
        if profit_col in engineered_df.columns and revenue_col in engineered_df.columns:
            # Avoid division by zero
            engineered_df['profit_margin'] = np.where(
                engineered_df[revenue_col] != 0, 
                engineered_df[profit_col] / engineered_df[revenue_col], 
                0
            )
    elif formula_type == 'custom_ratio':
        num_col = kwargs.get('num_col')
        den_col = kwargs.get('den_col')
        new_col = kwargs.get('new_col', f"{num_col}_to_{den_col}_ratio")
        if num_col in engineered_df.columns and den_col in engineered_df.columns:
            engineered_df[new_col] = np.where(
                engineered_df[den_col] != 0,
                engineered_df[num_col] / engineered_df[den_col],
                0
            )
            
    return engineered_df
