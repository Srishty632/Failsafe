from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class Faculty(Base):
    __tablename__ = "faculty"
    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String)
    email          = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at     = Column(DateTime, default=func.now())

class Prediction(Base):
    __tablename__ = "predictions"
    id          = Column(Integer, primary_key=True, index=True)
    faculty_email = Column(String)
    age         = Column(Integer)
    absences    = Column(Integer)
    failures    = Column(Integer)
    studytime   = Column(Integer)
    at_risk     = Column(Integer)
    probability = Column(Float)
    created_at  = Column(DateTime, default=func.now())
