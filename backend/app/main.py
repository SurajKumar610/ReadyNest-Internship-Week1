import os
import sys
import uuid
import json
import datetime
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Setup production logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("darp-backend")

# Add current directory to python path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from database import engine, Base, get_db
import models
import schemas
import crud

# Import analytical functions
from src.data_loading import load_from_file, get_dataset_metadata
from src.data_cleaning import remove_duplicates, impute_missing_values, handle_outliers
from src.descriptive_statistics import calculate_descriptive_statistics
from src.insight_extraction import extract_insights
from src.univariate_analysis import analyze_numerical_variable, analyze_categorical_variable
from src.bivariate_analysis import (
    analyze_numerical_vs_numerical, analyze_numerical_vs_categorical,
    analyze_categorical_vs_categorical, generate_correlation_matrix
)
from src.feature_engineering import (
    perform_label_encoding, perform_one_hot_encoding, scale_features,
    add_date_features, create_derived_features
)
from src.forecasting import time_series_forecast
from src.geo_mapping import (
    generate_plotly_scatter_map, generate_folium_marker_map,
    generate_folium_heatmap, generate_plotly_choropleth_map
)
from src.predictive_modeling import run_regression_modeling, run_classification_modeling
from src.ab_testing import run_numerical_ab_test, run_conversion_ab_test, run_proportion_z_test
from src.reporting import generate_pdf_report, generate_excel_report, generate_ppt_report

# Initialize database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Dataset Analytics Platform API", version="1.0")

# Setup CORS
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://sun-darp.streamlit.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "uploads")
REPORTS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "pdf"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "excel"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "ppt"), exist_ok=True)

# Mount reports directory as static files
app.mount("/static/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

@app.get("/")
def root():
    return {"status": "ok", "message": "DARP Dataset Analytics Platform API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

def serialize_data(obj):
    """Recursively convert Plotly figures, numpy types, and pandas frames to JSON serializable structures."""
    if isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_data(x) for x in obj)
    elif hasattr(obj, "to_dict"):  # Plotly figures or pandas structures
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def format_dataset_response(db_dataset: models.DatasetModel) -> schemas.DatasetResponse:
    """Format SQLAlchemy database model to Pydantic Response schema."""
    return schemas.DatasetResponse(
        id=db_dataset.id,
        filename=db_dataset.filename,
        shape_rows=db_dataset.shape_rows,
        shape_cols=db_dataset.shape_cols,
        columns_list=json.loads(db_dataset.columns_list),
        missing_counts=json.loads(db_dataset.missing_counts),
        data_types=json.loads(db_dataset.data_types),
        status=db_dataset.status,
        uploaded_at=db_dataset.uploaded_at
    )

def load_dataset_dataframe(filepath: str) -> pd.DataFrame:
    """Helper to load dataframe based on extension."""
    ext = filepath.split(".")[-1].lower()
    if ext == "csv":
        return pd.read_csv(filepath)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(filepath)
    elif ext == "json":
        return pd.read_json(filepath)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def save_dataset_dataframe(df: pd.DataFrame, filepath: str):
    """Helper to save dataframe based on extension."""
    ext = filepath.split(".")[-1].lower()
    if ext == "csv":
        df.to_csv(filepath, index=False)
    elif ext in ["xlsx", "xls"]:
        df.to_excel(filepath, index=False)
    elif ext == "json":
        df.to_json(filepath, orient="records", date_format="iso")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def calculate_and_serialize_metadata(df: pd.DataFrame):
    """Calculate metadata and return serialized JSON fields."""
    meta = get_dataset_metadata(df)
    
    # Calculate missing value counts
    missing_values = df.isnull().sum().to_dict()
    
    # Convert data types to simple strings
    data_types = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            data_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            data_types[col] = "datetime"
        elif isinstance(dtype, pd.CategoricalDtype):
            data_types[col] = "category"
        else:
            data_types[col] = "object"

    return {
        "shape_rows": df.shape[0],
        "shape_cols": df.shape[1],
        "columns_list": json.dumps(df.columns.tolist()),
        "missing_counts": json.dumps(serialize_data(missing_values)),
        "data_types": json.dumps(data_types)
    }


MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB

@app.post("/datasets/upload", response_model=schemas.DatasetResponse)
async def upload_dataset(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Header Validation (Content-Length)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_UPLOAD_SIZE:
                logger.error(f"Upload rejected: Content-Length {size} exceeds limit of {MAX_UPLOAD_SIZE}")
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds the 100MB upload limit. Please upload a smaller file."
                )
        except ValueError:
            pass

    # Create UUID and save locally
    dataset_id = str(uuid.uuid4())
    file_ext = file.filename.split(".")[-1]
    saved_filename = f"{dataset_id}.{file_ext}"
    filepath = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        # Write chunks and validate size dynamically
        total_bytes = 0
        with open(filepath, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunk
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    logger.error(f"Upload rejected: Actual file size read exceeds limit of {MAX_UPLOAD_SIZE}")
                    buffer.close()
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise HTTPException(
                        status_code=413,
                        detail="File size exceeds the 100MB upload limit. Please upload a smaller file."
                    )
                buffer.write(chunk)

        # 2. Disk Size Validation
        if os.path.exists(filepath):
            disk_size = os.path.getsize(filepath)
            if disk_size > MAX_UPLOAD_SIZE:
                logger.error(f"Upload rejected: Disk file size {disk_size} exceeds limit of {MAX_UPLOAD_SIZE}")
                os.remove(filepath)
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds the 100MB upload limit. Please upload a smaller file."
                )

        # Load and verify dataframe
        df = load_from_file(filepath, file_ext)
        meta_fields = calculate_and_serialize_metadata(df)
        
        # Save to database
        db_dataset = schemas.DatasetCreate(
            id=dataset_id,
            filename=file.filename,
            filepath=filepath,
            shape_rows=meta_fields["shape_rows"],
            shape_cols=meta_fields["shape_cols"],
            columns_list=meta_fields["columns_list"],
            missing_counts=meta_fields["missing_counts"],
            data_types=meta_fields["data_types"],
            status="raw"
        )
        
        created = crud.create_dataset(db, db_dataset)
        logger.info(f"Successfully processed upload: {file.filename} (ID: {dataset_id}, Size: {total_bytes} bytes)")
        return format_dataset_response(created)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        # Cleanup file on error
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        logger.error(f"Failed to process dataset: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")


@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    db_datasets = crud.get_datasets(db)
    return [format_dataset_response(d) for d in db_datasets]


@app.get("/datasets/{dataset_id}", response_model=schemas.DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return format_dataset_response(db_dataset)


@app.post("/datasets/{dataset_id}/clean", response_model=schemas.DatasetResponse)
def clean_dataset(dataset_id: str, request: schemas.CleanRequest, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        
        # Apply duplicate removal
        if request.remove_duplicates:
            df = remove_duplicates(df)

        # Apply missing value imputation
        if request.impute_col and request.impute_strategy:
            fill_val = request.fill_value
            if request.impute_strategy in ["mean", "median", "mode"]:
                # Try to parse to float if possible
                try:
                    fill_val = float(fill_val) if fill_val else None
                except:
                    pass
            df = impute_missing_values(df, [request.impute_col], strategy=request.impute_strategy, fill_value=fill_val)

        # Apply outlier treatment
        if request.outlier_col and request.outlier_method and request.outlier_action:
            df = handle_outliers(df, request.outlier_col, method=request.outlier_method, action=request.outlier_action)

        # Save back to CSV/Excel/JSON
        save_dataset_dataframe(df, db_dataset.filepath)

        # Update metadata in DB
        meta_fields = calculate_and_serialize_metadata(df)
        updated = crud.update_dataset_metadata(
            db=db,
            dataset_id=dataset_id,
            shape_rows=meta_fields["shape_rows"],
            shape_cols=meta_fields["shape_cols"],
            columns_list=json.loads(meta_fields["columns_list"]),
            missing_counts=json.loads(meta_fields["missing_counts"]),
            data_types=json.loads(meta_fields["data_types"]),
            status="cleaned"
        )
        return format_dataset_response(updated)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data cleaning failed: {str(e)}")


@app.get("/datasets/{dataset_id}/stats")
def get_descriptive_stats(dataset_id: str, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = load_dataset_dataframe(db_dataset.filepath)
    stats_df = calculate_descriptive_statistics(df)
    return serialize_data(stats_df)


@app.get("/datasets/{dataset_id}/insights")
def get_insights(dataset_id: str, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = load_dataset_dataframe(db_dataset.filepath)
    insights = extract_insights(df)
    return serialize_data(insights)


@app.post("/datasets/{dataset_id}/univariate")
def run_univariate_analysis(dataset_id: str, payload: dict, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    col = payload.get("column")
    if not col:
        raise HTTPException(status_code=400, detail="Column name is required")

    df = load_dataset_dataframe(db_dataset.filepath)
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    
    if is_numeric:
        analysis = analyze_numerical_variable(df, col)
    else:
        analysis = analyze_categorical_variable(df, col)

    return serialize_data(analysis)


@app.post("/datasets/{dataset_id}/bivariate")
def run_bivariate_analysis(dataset_id: str, payload: dict, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    col_x = payload.get("column_x")
    col_y = payload.get("column_y")
    if not col_x or not col_y:
        raise HTTPException(status_code=400, detail="column_x and column_y are required")

    df = load_dataset_dataframe(db_dataset.filepath)
    is_x_num = pd.api.types.is_numeric_dtype(df[col_x])
    is_y_num = pd.api.types.is_numeric_dtype(df[col_y])

    if is_x_num and is_y_num:
        res = analyze_numerical_vs_numerical(df, col_x, col_y)
    elif (is_x_num and not is_y_num) or (not is_x_num and is_y_num):
        num = col_x if is_x_num else col_y
        cat = col_y if is_x_num else col_x
        res = analyze_numerical_vs_categorical(df, num, cat)
    else:
        res = analyze_categorical_vs_categorical(df, col_x, col_y)

    return serialize_data(res)


@app.get("/datasets/{dataset_id}/correlation")
def run_correlation(dataset_id: str, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = load_dataset_dataframe(db_dataset.filepath)
    res = generate_correlation_matrix(df)
    return serialize_data(res)


@app.post("/datasets/{dataset_id}/feature-engineering", response_model=schemas.DatasetResponse)
def feature_engineering(dataset_id: str, request: schemas.FeatureEngineeringRequest, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        
        # 1. Date features
        if request.date_col:
            df = add_date_features(df, request.date_col)
            
        # 2. Scaling
        if request.scale_cols and request.scale_method:
            df, _ = scale_features(df, request.scale_cols, method=request.scale_method)
            
        # 3. Encoding
        if request.encode_cols and request.encode_method:
            if request.encode_method == "Label Encoding":
                df, _ = perform_label_encoding(df, request.encode_cols)
            else:
                df = perform_one_hot_encoding(df, request.encode_cols)
                
        # 4. Derived
        if request.ratio_type:
            df = create_derived_features(
                df,
                feature_type=request.ratio_type,
                num_col=request.num_col,
                den_col=request.den_col
            )

        save_dataset_dataframe(df, db_dataset.filepath)

        # Update metadata in DB
        meta_fields = calculate_and_serialize_metadata(df)
        updated = crud.update_dataset_metadata(
            db=db,
            dataset_id=dataset_id,
            shape_rows=meta_fields["shape_rows"],
            shape_cols=meta_fields["shape_cols"],
            columns_list=json.loads(meta_fields["columns_list"]),
            missing_counts=json.loads(meta_fields["missing_counts"]),
            data_types=json.loads(meta_fields["data_types"]),
            status="cleaned"
        )
        return format_dataset_response(updated)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")


@app.post("/datasets/{dataset_id}/forecast")
def run_forecast(dataset_id: str, request: schemas.ForecastRequest, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        fore_results = time_series_forecast(
            df,
            request.date_col,
            request.val_col,
            forecast_steps=request.steps,
            model_type=request.model_type,
            freq=request.freq
        )
        return serialize_data(fore_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forecasting failed: {str(e)}")


@app.post("/datasets/{dataset_id}/predictive-modeling")
def run_predictive_modeling(dataset_id: str, request: schemas.PredictiveModelingRequest, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        
        # Preprocess features (e.g. Label Encode object fields)
        df_ml = df.copy()
        encoded_features = []
        for col in request.features:
            if col in df_ml.columns:
                if pd.api.types.is_numeric_dtype(df_ml[col]):
                    encoded_features.append(col)
                else:
                    df_ml[col] = df_ml[col].astype(str)
                    df_ml[col] = pd.Categorical(df_ml[col]).codes
                    encoded_features.append(col)

        is_y_numeric = pd.api.types.is_numeric_dtype(df[request.target])
        y_cardinality = len(df[request.target].dropna().unique())
        
        task = "Regression" if (is_y_numeric and y_cardinality > 10) else "Classification"
        if request.task_mode != "Auto Detect":
            task = request.task_mode

        if task == "Regression":
            results = run_regression_modeling(df_ml.dropna(), encoded_features, request.target, model_type=request.algo)
        else:
            results = run_classification_modeling(df_ml.dropna(), encoded_features, request.target, model_type=request.algo)

        return serialize_data(results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predictive modeling failed: {str(e)}")


@app.post("/datasets/{dataset_id}/ab-test")
def run_ab_test(dataset_id: str, request: schemas.ABTestRequest, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        is_metric_num = pd.api.types.is_numeric_dtype(df[request.metric_col])

        if is_metric_num:
            res = run_numerical_ab_test(
                df,
                request.group_col,
                request.metric_col,
                variant_a_label=request.variant_a,
                variant_b_label=request.variant_b
            )
        else:
            if request.test_type == "Proportions Z-Test":
                res = run_proportion_z_test(
                    df,
                    request.group_col,
                    request.metric_col,
                    variant_a_label=request.variant_a,
                    variant_b_label=request.variant_b
                )
            else:
                res = run_conversion_ab_test(
                    df,
                    request.group_col,
                    request.metric_col,
                    variant_a_label=request.variant_a,
                    variant_b_label=request.variant_b
                )

        return serialize_data(res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"A/B testing failed: {str(e)}")


@app.post("/datasets/{dataset_id}/geo-map")
def run_geo_map(dataset_id: str, payload: dict, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    map_style = payload.get("map_style")
    lat_col = payload.get("lat_col")
    lon_col = payload.get("lon_col")
    state_col = payload.get("state_col")
    weight_col = payload.get("weight_col")

    df = load_dataset_dataframe(db_dataset.filepath)

    if map_style == "Choropleth (USA States)":
        fig = generate_plotly_choropleth_map(df, state_col, weight_col)
        return {"plotly_fig": serialize_data(fig)}
    elif map_style == "Scatter Mapbox (Plotly)":
        fig = generate_plotly_scatter_map(df, lat_col, lon_col, size_col=weight_col, color_col=weight_col)
        return {"plotly_fig": serialize_data(fig)}
    elif map_style == "Marker Cluster (Folium)":
        m = generate_folium_marker_map(df, lat_col, lon_col, popup_cols=[weight_col] if weight_col else None)
        html_str = m._repr_html_() if m else ""
        return {"folium_html": html_str}
    else: # Heatmap
        m = generate_folium_heatmap(df, lat_col, lon_col, weight_col=weight_col)
        html_str = m._repr_html_() if m else ""
        return {"folium_html": html_str}


@app.post("/datasets/{dataset_id}/report")
def run_report(dataset_id: str, db: Session = Depends(get_db)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset_dataframe(db_dataset.filepath)
        stats_df = calculate_descriptive_statistics(df)
        insights = extract_insights(df)

        pdf_out = os.path.join(REPORTS_DIR, "pdf", f"{dataset_id}_report.pdf")
        excel_out = os.path.join(REPORTS_DIR, "excel", f"{dataset_id}_sheets.xlsx")
        ppt_out = os.path.join(REPORTS_DIR, "ppt", f"{dataset_id}_slides.pptx")

        generate_pdf_report(stats_df, insights, pdf_out)
        generate_excel_report(df, df, stats_df, excel_out)
        generate_ppt_report(insights, ppt_out)

        return {
            "pdf_url": f"/static/reports/pdf/{dataset_id}_report.pdf",
            "excel_url": f"/static/reports/excel/{dataset_id}_sheets.xlsx",
            "ppt_url": f"/static/reports/ppt/{dataset_id}_slides.pptx"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Report generation failed: {str(e)}")
