from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base


class LogEntry(Base):
    __tablename__ = "log_entries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    food = Column(String, nullable=False)
    category = Column(String, nullable=False)
    meal_type = Column(String, nullable=False)
    weight_grams = Column(Float, nullable=False)
    calories = Column(Float, nullable=True)
    protein = Column(Float, nullable=True)
    carbohydrates = Column(Float, nullable=True)
    fat = Column(Float, nullable=True)
    fiber = Column(Float, nullable=True)
    sugars = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    sodium = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
