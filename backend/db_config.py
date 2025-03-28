from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.future import select
import boto3
import os

S3_BUCKET = "your-bucket-name"
S3_REGION = "us-west-1"  # Change as needed

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# FastAPI app initialization
app = FastAPI()

# Database configuration
DATABASE_URL = "postgresql://postgres:20Brain25?!@braindb-instance-1-us-west-1b.cb2oooakoxci.us-west-1.rds.amazonaws.com:5432/braindb"
db = SQLAlchemy()

## Connect to the database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define PatientImage model
class PatientImage(Base):
    __tablename__ = "patient_images"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    scan_date = Column(DateTime, default=datetime.datetime.utcnow)
    s3_url = Column(String)  # S3 link to the MRI scan
    predicted_segmentation_url = Column(String)  # S3 link to segmentation
    tumor_volume = Column(Float)  # Volume of detected tumor
    tumor_growth_rate = Column(Float)  # Growth rate percentage
    tumor_type = Column(String)  # Tumor classification
    annotations = Column(JSON)  # JSON format annotations

# Create tables in the database
Base.metadata.create_all(bind=engine)

print("Database setup complete. Tables created.")