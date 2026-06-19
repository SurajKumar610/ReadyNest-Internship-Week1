from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional, Any

class DatasetBase(BaseModel):
    filename: str
    status: str
    shape_rows: int
    shape_cols: int
    uploaded_at: datetime

class DatasetCreate(BaseModel):
    id: str
    filename: str
    filepath: str
    shape_rows: int
    shape_cols: int
    columns_list: str  # Serialized JSON
    missing_counts: str  # Serialized JSON
    data_types: str  # Serialized JSON
    status: str = "raw"

class DatasetResponse(BaseModel):
    id: str
    filename: str
    shape_rows: int
    shape_cols: int
    columns_list: List[str]
    missing_counts: Dict[str, int]
    data_types: Dict[str, str]
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class CleanRequest(BaseModel):
    remove_duplicates: bool = False
    impute_col: Optional[str] = None
    impute_strategy: Optional[str] = None  # mean, median, mode, drop, constant
    fill_value: Optional[str] = None
    outlier_col: Optional[str] = None
    outlier_method: Optional[str] = None  # iqr, zscore
    outlier_action: Optional[str] = None  # cap, drop, replace

class FeatureEngineeringRequest(BaseModel):
    date_col: Optional[str] = None
    scale_cols: Optional[List[str]] = None
    scale_method: Optional[str] = None  # standard, minmax
    encode_cols: Optional[List[str]] = None
    encode_method: Optional[str] = None  # Label Encoding, One-Hot Encoding
    ratio_type: Optional[str] = None  # profit_margin, custom_ratio
    num_col: Optional[str] = None
    den_col: Optional[str] = None

class ForecastRequest(BaseModel):
    date_col: str
    val_col: str
    steps: int = 12
    model_type: str = "arima"  # arima, exponential_smoothing, moving_average
    freq: str = "ME"  # ME, W, D

class PredictiveModelingRequest(BaseModel):
    target: str
    features: List[str]
    task_mode: str = "Auto Detect"  # Auto Detect, Regression, Classification
    algo: str = "linear"  # linear, ridge, decision_tree, random_forest, logistic

class ABTestRequest(BaseModel):
    group_col: str
    metric_col: str
    variant_a: str
    variant_b: str
    test_type: str = "T-Test"  # T-Test, Chi-Square Test, Proportions Z-Test
