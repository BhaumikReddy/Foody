import os
import requests
from dotenv import load_dotenv
load_dotenv()

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def fetch_nutrition(food_name: str) -> dict:
    """Call USDA FoodData Central API and return per-100g nutritional values."""
    api_key = os.getenv("USDA_API_KEY")
    if not api_key:
        print("[nutrition.py] Error: USDA_API_KEY not found in environment.")
        return {k: None for k in ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugars', 'cholesterol', 'sodium']}

    params = {
        "query": food_name,
        "api_key": api_key,
        "dataType": "SR Legacy,Foundation",
        "pageSize": 1
    }

    result_data = {
        'calories': None,
        'protein': None,
        'carbohydrates': None,
        'fat': None,
        'fiber': None,
        'sugars': None,
        'cholesterol': None,
        'sodium': None,
    }

    try:
        response = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('foods') and len(data['foods']) > 0:
            food_item = data['foods'][0]
            nutrients = food_item.get('foodNutrients', [])

            # USDA nutrientNumber values
            nutrient_map = {
                "208": 'calories',
                "203": 'protein',
                "205": 'carbohydrates',
                "204": 'fat',
                "291": 'fiber',
                "269": 'sugars',
                "601": 'cholesterol',
                "307": 'sodium'
            }

            for nutrient in nutrients:
                nutrient_number = str(nutrient.get('nutrientNumber', ''))
                if nutrient_number in nutrient_map:
                    field = nutrient_map[nutrient_number]
                    result_data[field] = nutrient.get('value')

            # Debug — print what was found
            print(f"[nutrition.py] Nutrition fetched for '{food_name}': {result_data}")

        else:
            print(f"[nutrition.py] USDA API returned no foods for query: {food_name}")

        return result_data

    except Exception as e:
        print(f"[nutrition.py] USDA API error: {e}")
        return result_data


def scale_nutrition(nutrition: dict, weight_grams: float) -> dict:
    """Scale per-100g values to the given weight."""
    return {
        k: round(v * weight_grams / 100, 2) if v is not None else None
        for k, v in nutrition.items()
    }