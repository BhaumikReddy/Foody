from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from model import predict_image, fruits, vegetables
from nutrition import fetch_nutrition, scale_nutrition
from models.log_entry import LogEntry
from schemas.log_entry import LogEntryResponse

router = APIRouter()


@router.post("/predict", response_model=LogEntryResponse)
async def predict(
    file: UploadFile = File(...),
    meal_type: str = Form(...),
    weight_grams: float = Form(...),
    db: Session = Depends(get_db)
):
    """
    Accept an image + meal metadata, classify the food, fetch nutrition,
    save to DB, and return the log entry.
    """
    # 1. Read image bytes and predict food name
    file_bytes = await file.read()
    try:
        food_name = predict_image(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image prediction failed: {str(e)}")

    # 2. Determine category
    category = "Fruit" if food_name in fruits else "Vegetable"

    # 3. Fetch nutrition and scale by weight
    nutrition_per_100g = fetch_nutrition(food_name)
    scaled = scale_nutrition(nutrition_per_100g, weight_grams)

    # 4. Save to database
    entry = LogEntry(
        food=food_name,
        category=category,
        meal_type=meal_type,
        weight_grams=weight_grams,
        calories=scaled.get('calories'),
        protein=scaled.get('protein'),
        carbohydrates=scaled.get('carbohydrates'),
        fat=scaled.get('fat'),
        fiber=scaled.get('fiber'),
        sugars=scaled.get('sugars'),
        cholesterol=scaled.get('cholesterol'),
        sodium=scaled.get('sodium'),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return entry
