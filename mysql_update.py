import json
import pymysql
from sqlalchemy import create_engine, text

def load_ingredients_to_db(json_path, db_url):
    engine = create_engine(db_url)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with engine.begin() as conn:
        # Disable FK checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Clear tables
        conn.execute(text("TRUNCATE TABLE foods_ingredient_nutrients"))
        conn.execute(text("TRUNCATE TABLE foods_ingredient"))

        # Enable FK checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        # Insert ingredients
        for item in data:
            conn.execute(
                text("""
                    INSERT INTO foods_ingredient (id, name, image, food_group_id)
                    VALUES (:id, :name, :image, :food_group)
                """),
                {
                    "id": item["id"],
                    "name": item["name"],
                    "image": item["image"],
                    "food_group": item["food_group"]
                }
            )

        # Insert nutrients (junction table)
        for item in data:
            for nutrient_id in item["nutrients"]:
                conn.execute(
                    text("""
                        INSERT INTO foods_ingredient_nutrients (ingredient_id, nutrient_id)
                        VALUES (:ingredient_id, :nutrient_id)
                    """),
                    {
                        "ingredient_id": item["id"],
                        "nutrient_id": nutrient_id
                    }
                )

# UPDATE INGREDIENTS TABLE WITH JSON DATA
# db_url = "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db?charset=utf8mb4"
# json_path = "/home/k503/下載/Spoonfull/rag_backend/ingredients.json"
# print("Loading ingredients into database...")
# load_ingredients_to_db(json_path, db_url)
# print("Ingredients loaded successfully!")



from datetime import datetime

def parse_datetime(dt_str):
    """Convert ISO datetime to MySQL format"""
    return datetime.fromisoformat(dt_str.replace("Z", ""))

def load_meals_to_db(json_path, db_url):
    engine = create_engine(db_url)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with engine.begin() as conn:
        # Disable FK checks
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Clear tables
        conn.execute(text("TRUNCATE TABLE meals_meal_ingredients"))
        conn.execute(text("TRUNCATE TABLE meals_meal"))

        # Re-enable FK
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        # Insert meals
        for item in data:
            conn.execute(
                text("""
                    INSERT INTO meals_meal 
                    (id, created_at, updated_at, meal_time, plate_type, day_cycle, image, meal_name)
                    VALUES 
                    (:id, :created_at, :updated_at, :meal_time, :plate_type, :day_cycle, :image, :meal_name)
                """),
                {
                    "id": item["id"],
                    "created_at": parse_datetime(item["created_at"]),
                    "updated_at": parse_datetime(item["updated_at"]),
                    "meal_time": item["meal_time"],
                    "plate_type": item["plate_type"],
                    "day_cycle": item["day_cycle"],
                    "image": item["image"],
                    "meal_name": item["meal_name"]
                }
            )

        # Insert meal ↔ ingredients
        for item in data:
            for ing_id in item["ingredients"]:
                conn.execute(
                    text("""
                        INSERT INTO meals_meal_ingredients (meal_id, ingredient_id)
                        VALUES (:meal_id, :ingredient_id)
                    """),
                    {
                        "meal_id": item["id"],
                        "ingredient_id": ing_id
                    }
                )

# UPDATE MEALS TABLE WITH JSON DATA
db_url = "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db?charset=utf8mb4"
json_path = "/home/k503/下載/Spoonfull/rag_backend/meals.json"
print("Loading meals into database...")
load_meals_to_db(json_path, db_url)
print("Meals loaded successfully!")