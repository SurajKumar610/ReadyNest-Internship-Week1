from sqlalchemy.orm import Session
from models import DatasetModel
from schemas import DatasetCreate
import json

def get_dataset(db: Session, dataset_id: str):
    return db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()

def get_datasets(db: Session):
    return db.query(DatasetModel).all()

def create_dataset(db: Session, dataset: DatasetCreate):
    db_dataset = DatasetModel(
        id=dataset.id,
        filename=dataset.filename,
        filepath=dataset.filepath,
        shape_rows=dataset.shape_rows,
        shape_cols=dataset.shape_cols,
        columns_list=dataset.columns_list,
        missing_counts=dataset.missing_counts,
        data_types=dataset.data_types,
        status=dataset.status
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def update_dataset_metadata(
    db: Session,
    dataset_id: str,
    shape_rows: int,
    shape_cols: int,
    columns_list: list,
    missing_counts: dict,
    data_types: dict,
    status: str
):
    db_dataset = get_dataset(db, dataset_id)
    if db_dataset:
        db_dataset.shape_rows = shape_rows
        db_dataset.shape_cols = shape_cols
        db_dataset.columns_list = json.dumps(columns_list)
        db_dataset.missing_counts = json.dumps(missing_counts)
        db_dataset.data_types = json.dumps(data_types)
        db_dataset.status = status
        db.commit()
        db.refresh(db_dataset)
    return db_dataset
