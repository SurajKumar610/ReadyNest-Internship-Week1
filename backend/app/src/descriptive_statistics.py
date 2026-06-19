import pandas as pd
import numpy as np

def calculate_descriptive_statistics(df, columns=None):
    """
    Generate detailed descriptive statistics for numerical columns.
    Returns a pandas DataFrame where columns represent statistical metrics
    and rows represent the analyzed numerical features.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        if series.empty:
            continue
            
        # Modes can have multiple values, we take the first one
        mode_val = series.mode()
        mode_res = mode_val.iloc[0] if not mode_val.empty else np.nan
        
        stats[col] = {
            "Count": len(series),
            "Mean": series.mean(),
            "Median": series.median(),
            "Mode": mode_res,
            "Min": series.min(),
            "Max": series.max(),
            "Range": series.max() - series.min(),
            "Variance": series.var(),
            "Std Dev": series.std(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),
            "25% (Q1)": series.quantile(0.25),
            "50% (Q2)": series.quantile(0.50),
            "75% (Q3)": series.quantile(0.75),
            "Missing Values": df[col].isnull().sum(),
            "Missing %": (df[col].isnull().sum() / len(df)) * 100
        }
        
    return pd.DataFrame(stats).T
