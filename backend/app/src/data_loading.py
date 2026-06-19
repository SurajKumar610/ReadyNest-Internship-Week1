import pandas as pd
import json
import requests
from io import StringIO, BytesIO
from sqlalchemy import create_engine

def load_from_file(file_path_or_bytes, file_type, **kwargs):
    """
    Load data from local file or bytes object.
    file_type can be 'csv', 'xlsx', 'json'.
    """
    try:
        if file_type.lower() == 'csv':
            return pd.read_csv(file_path_or_bytes, **kwargs)
        elif file_type.lower() in ['xlsx', 'xls']:
            return pd.read_excel(file_path_or_bytes, **kwargs)
        elif file_type.lower() == 'json':
            return pd.read_json(file_path_or_bytes, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise IOError(f"Error loading {file_type} file: {str(e)}")

def load_from_database(connection_uri, query):
    """
    Load data from database via SQLAlchemy engine.
    Supports PostgreSQL, MySQL, SQLite, etc.
    """
    try:
        engine = create_engine(connection_uri)
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        raise ConnectionError(f"Database error: {str(e)}")

def load_from_api(url, params=None, headers=None, json_path=None):
    """
    Load data from REST API endpoints.
    json_path can be used to select a nested list (e.g. 'data.records').
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Navigate nested JSON if path is specified
        if json_path:
            keys = json_path.split('.')
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key, {})
                elif isinstance(data, list) and key.isdigit():
                    data = data[int(key)]
                else:
                    raise KeyError(f"Key '{key}' not found in API response.")
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If it's a single record or dictionary of columns
            return pd.DataFrame([data])
        else:
            raise ValueError("API returned content that could not be parsed into a dataframe.")
    except Exception as e:
        raise IOError(f"API Connection error: {str(e)}")

def get_dataset_metadata(df):
    """
    Generate dataset metadata summary.
    """
    if df is None or df.empty:
        return {
            "shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "missing_values": {},
            "preview": pd.DataFrame()
        }
    
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "preview": df.head(10)
    }
