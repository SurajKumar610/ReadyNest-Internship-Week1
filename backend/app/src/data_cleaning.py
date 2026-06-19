import pandas as pd
import numpy as np

def impute_missing_values(df, columns, strategy='mean', fill_value=None):
    """
    Impute missing values in selected columns.
    strategy: 'mean', 'median', 'mode', 'drop', 'constant'
    """
    cleaned_df = df.copy()
    for col in columns:
        if col not in cleaned_df.columns:
            continue
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif strategy == 'mode':
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
        elif strategy == 'constant':
            cleaned_df[col] = cleaned_df[col].fillna(fill_value if fill_value is not None else 0)
        elif strategy == 'drop':
            cleaned_df = cleaned_df.dropna(subset=[col])
            
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataset.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def convert_data_types(df, column_type_mapping):
    """
    Convert column data types based on mapping dict: {column: new_type}
    new_type can be 'int', 'float', 'datetime', 'string', 'category'.
    """
    cleaned_df = df.copy()
    for col, new_type in column_type_mapping.items():
        if col not in cleaned_df.columns:
            continue
        try:
            if new_type == 'datetime':
                # Try inferring format or generic parsing
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            elif new_type == 'int':
                # Replace infs and nulls before cast
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0).astype(int)
            elif new_type == 'float':
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            elif new_type == 'string':
                cleaned_df[col] = cleaned_df[col].astype(str)
            elif new_type == 'category':
                cleaned_df[col] = cleaned_df[col].astype('category')
        except Exception as e:
            raise ValueError(f"Failed to convert {col} to {new_type}: {str(e)}")
    return cleaned_df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean series where True indicates outlier.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df), index=df.index)
        
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df, column, threshold=3.0):
    """
    Detect outliers using Z-score method.
    Returns boolean series where True indicates outlier.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series([False] * len(df), index=df.index)
        
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return pd.Series([False] * len(df), index=df.index)
    z_scores = (df[column] - mean) / std
    return np.abs(z_scores) > threshold

def handle_outliers(df, column, method='iqr', action='drop', threshold=1.5, fill_value=None):
    """
    Handle outliers by dropping them, capping them, or replacing with a fill value.
    action: 'drop', 'cap', 'replace'
    """
    cleaned_df = df.copy()
    if method == 'iqr':
        outliers = detect_outliers_iqr(cleaned_df, column, threshold)
    else:
        outliers = detect_outliers_zscore(cleaned_df, column, threshold)
        
    if action == 'drop':
        cleaned_df = cleaned_df[~outliers]
    elif action == 'replace':
        replace_val = fill_value if fill_value is not None else cleaned_df[column].median()
        cleaned_df.loc[outliers, column] = replace_val
    elif action == 'cap':
        q1 = cleaned_df[column].quantile(0.25)
        q3 = cleaned_df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        cleaned_df[column] = np.clip(cleaned_df[column], lower_bound, upper_bound)
        
    return cleaned_df

def standardize_text(df, columns, action='both'):
    """
    Standardize text columns.
    action: 'lowercase', 'trim', 'both'
    """
    cleaned_df = df.copy()
    for col in columns:
        if col not in cleaned_df.columns:
            continue
        cleaned_df[col] = cleaned_df[col].astype(str)
        if action in ['lowercase', 'both']:
            cleaned_df[col] = cleaned_df[col].str.lower()
        if action in ['trim', 'both']:
            cleaned_df[col] = cleaned_df[col].str.strip()
    return cleaned_df
