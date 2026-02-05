import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/nutrition_db.json')

def load_nutrition_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_nutrition_by_name(name):
    db = load_nutrition_db()
    for item in db:
        if item['name'] == name:
            return item
    return None

def get_nutrition_for_foods(food_names):
    db = load_nutrition_db()
    result = []
    for name in food_names:
        for item in db:
            if item['name'] == name:
                result.append(item)
                break
    return result
