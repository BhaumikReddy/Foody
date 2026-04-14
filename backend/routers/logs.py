from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from database import get_db
from models.log_entry import LogEntry
from schemas.log_entry import LogEntryResponse

router = APIRouter()


@router.get("/logs", response_model=dict)
def get_logs(
    meal_type: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    food: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(100),
    offset: int = Query(0),
    db: Session = Depends(get_db)
):
    """Get all log entries with optional filters, pagination, and total count."""
    query = db.query(LogEntry)

    if meal_type:
        query = query.filter(LogEntry.meal_type == meal_type)
    if category:
        query = query.filter(LogEntry.category == category)
    if food:
        query = query.filter(LogEntry.food.ilike(f"%{food}%"))
    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from)
            query = query.filter(LogEntry.timestamp >= dt_from)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid date_from format. Use ISO format.")
    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to)
            query = query.filter(LogEntry.timestamp <= dt_to)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid date_to format. Use ISO format.")

    total = query.count()
    entries = query.order_by(LogEntry.timestamp.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "items": [LogEntryResponse.from_orm(e) for e in entries]
    }


@router.get("/logs/summary")
def get_summary(db: Session = Depends(get_db)):
    """Return aggregated stats for the dashboard."""
    entries = db.query(LogEntry).all()

    if not entries:
        return {
            "total_entries": 0,
            "total_calories": 0.0,
            "avg_calories_per_meal": 0.0,
            "by_meal_type": {},
            "by_category": {},
            "top_foods": [],
            "daily_calories": []
        }

    total_entries = len(entries)
    total_calories = sum(e.calories for e in entries if e.calories is not None)
    avg_calories = round(total_calories / total_entries, 2) if total_entries else 0.0

    by_meal_type: dict = {}
    by_category: dict = {}
    food_counts: dict = {}
    daily_calories: dict = {}

    for e in entries:
        by_meal_type[e.meal_type] = by_meal_type.get(e.meal_type, 0) + 1
        by_category[e.category] = by_category.get(e.category, 0) + 1
        food_counts[e.food] = food_counts.get(e.food, 0) + 1
        day_str = e.timestamp.strftime('%Y-%m-%d') if e.timestamp else 'Unknown'
        daily_calories[day_str] = daily_calories.get(day_str, 0.0) + (e.calories or 0.0)

    top_foods = sorted(
        [{"food": k, "count": v} for k, v in food_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:10]

    daily_calories_list = sorted(
        [{"date": k, "calories": round(v, 2)} for k, v in daily_calories.items()],
        key=lambda x: x["date"]
    )

    return {
        "total_entries": total_entries,
        "total_calories": round(total_calories, 2),
        "avg_calories_per_meal": avg_calories,
        "by_meal_type": by_meal_type,
        "by_category": by_category,
        "top_foods": top_foods,
        "daily_calories": daily_calories_list
    }


@router.delete("/logs/{entry_id}")
def delete_log(entry_id: int, db: Session = Depends(get_db)):
    """Delete a single log entry by ID."""
    entry = db.query(LogEntry).filter(LogEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")
    db.delete(entry)
    db.commit()
    return {"deleted": True}
