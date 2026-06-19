import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from database import Base

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    shape_rows = Column(Integer)
    shape_cols = Column(Integer)
    columns_list = Column(Text)       # JSON string list
    missing_counts = Column(Text)     # JSON string dict
    data_types = Column(Text)         # JSON string dict
    status = Column(String, default="raw")
