from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LogEntryCreate(BaseModel):
    food: str
    category: str
    meal_type: str
    weight_grams: float
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbohydrates: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    sugars: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None


class LogEntryResponse(BaseModel):
    id: int
    food: str
    category: str
    meal_type: str
    weight_grams: float
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbohydrates: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    sugars: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    timestamp: datetime

    class Config:
        from_attributes = True
