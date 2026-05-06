# ==================================
# Get Meals from Database
# ==================================
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

# Returns a list of meal names available in the meals_meal table in the food_intakes_db database.
def get_list_of_meals():
    try:
        engine = create_engine(
            "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db"
        )

        # print("Running query...")
        query = "SELECT * FROM meals_meal"
        meals_table = pd.read_sql(query, engine)

        # print("Raw data:")
        # print(meals_table)

        meals_list = meals_table[['id', 'meal_name']].to_dict(orient='records')
        meal_names = [meal["meal_name"] for meal in meals_list]

        return meal_names

    except Exception as e:
        print("Error:", e)
        return []




def get_meals_by_nutrients(nutrient_names: list):
    engine = create_engine(
        "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db"
    )

    placeholders = ",".join(["%s"] * len(nutrient_names))

    query = f"""
    SELECT DISTINCT m.id, m.meal_name
    FROM meals_meal m
    JOIN meals_meal_ingredients mi ON m.id = mi.meal_id
    JOIN foods_ingredient i ON mi.ingredient_id = i.id
    JOIN foods_ingredient_nutrients inut ON i.id = inut.ingredient_id
    JOIN foods_nutrient n ON inut.nutrient_id = n.id
    WHERE n.name IN ({placeholders})
    """

    meals_table = pd.read_sql(query, engine, params=tuple(nutrient_names))
    meal_names = meals_table["meal_name"].dropna().tolist()

    return meal_names


# meals = get_meals_by_nutrients(["Fat", "Fats"])
# meals = get_meals_by_nutrients(["Carbohydrate"])
# meals = get_meals_by_nutrients(["Protein"])
# meals = get_meals_by_nutrients(["Total Fiber", "Fiber"])
# print("Meals by nutrients\n", meals)


def get_meals_excluding_nutrients(nutrient_names: list):
    engine = create_engine(
        "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db"
    )

    placeholders = ",".join(["%s"] * len(nutrient_names))

    query = f"""
    SELECT DISTINCT m.meal_name
    FROM meals_meal m
    WHERE m.id NOT IN (
        SELECT m.id
        FROM meals_meal m
        JOIN meals_meal_ingredients mi ON m.id = mi.meal_id
        JOIN foods_ingredient i ON mi.ingredient_id = i.id
        JOIN foods_ingredient_nutrients inut ON i.id = inut.ingredient_id
        JOIN foods_nutrient n ON inut.nutrient_id = n.id
        WHERE n.name IN ({placeholders})
    )
    """

    meals_table = pd.read_sql(query, engine, params=tuple(nutrient_names))
    meal_names = meals_table["meal_name"].dropna().tolist()

    return meal_names


# meals = get_meals_excluding_nutrients(["Fat", "Fats"])
# meals = get_meals_excluding_nutrients(["Carbohydrate"])
# meals = get_meals_excluding_nutrients(["Protein"])
# # meals = get_meals_excluding_nutrients(["Total Fiber", "Fiber"])
# print("Meals excluding nutrients\n", meals)



def get_meals_by_nutrients_with_ingredients(nutrient_names: list):
    engine = create_engine(
        "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db"
    )

    placeholders = ",".join(["%s"] * len(nutrient_names))

    query = f"""
    SELECT 
        m.meal_name,
        i.name AS ingredient_name
    FROM meals_meal m
    JOIN meals_meal_ingredients mi ON m.id = mi.meal_id
    JOIN foods_ingredient i ON mi.ingredient_id = i.id
    JOIN foods_ingredient_nutrients inut ON i.id = inut.ingredient_id
    JOIN foods_nutrient n ON inut.nutrient_id = n.id
    WHERE n.name IN ({placeholders})
    """

    df = pd.read_sql(query, engine, params=tuple(nutrient_names))

    # group ingredients per meal
    meal_map = {}

    for _, row in df.iterrows():
        meal = row["meal_name"]
        ingredient = row["ingredient_name"]

        if meal not in meal_map:
            meal_map[meal] = set()

        meal_map[meal].add(ingredient)

    # format output
    result = [
        f"{meal} ({', '.join(sorted(ingredients))})"
        for meal, ingredients in meal_map.items()
    ]

    return result

# meals = get_meals_by_nutrients_with_ingredients(["Fat", "Fats"])
# meals = get_meals_by_nutrients_with_ingredients(["Carbohydrate"])
# meals = get_meals_by_nutrients_with_ingredients(["Protein"])
meals = get_meals_by_nutrients_with_ingredients(["Total Fiber", "Fiber"])
print("Meals by nutrients with ingredients\n", meals)



def get_meals_excluding_nutrients_with_ingredients(nutrient_names: list):
    engine = create_engine(
        "mysql+pymysql://root:Root%401234@127.0.0.1:3306/food_intakes_db"
    )

    placeholders = ",".join(["%s"] * len(nutrient_names))

    query = f"""
    SELECT 
        m.meal_name,
        i.name AS ingredient_name
    FROM meals_meal m
    JOIN meals_meal_ingredients mi ON m.id = mi.meal_id
    JOIN foods_ingredient i ON mi.ingredient_id = i.id
    WHERE m.id NOT IN (
        SELECT m.id
        FROM meals_meal m
        JOIN meals_meal_ingredients mi ON m.id = mi.meal_id
        JOIN foods_ingredient i ON mi.ingredient_id = i.id
        JOIN foods_ingredient_nutrients inut ON i.id = inut.ingredient_id
        JOIN foods_nutrient n ON inut.nutrient_id = n.id
        WHERE n.name IN ({placeholders})
    )
    """

    df = pd.read_sql(query, engine, params=tuple(nutrient_names))

    # group ingredients per meal
    meal_map = {}

    for _, row in df.iterrows():
        meal = row["meal_name"]
        ingredient = row["ingredient_name"]

        if meal not in meal_map:
            meal_map[meal] = set()

        meal_map[meal].add(ingredient)

    # format output
    result = [
        f"{meal} ({', '.join(sorted(ingredients))})"
        for meal, ingredients in meal_map.items()
    ]

    return result


# meals = get_meals_excluding_nutrients_with_ingredients(["Fat", "Fats"])
# meals = get_meals_excluding_nutrients_with_ingredients(["Carbohydrate"])
# meals = get_meals_excluding_nutrients_with_ingredients(["Protein"])
# meals = get_meals_excluding_nutrients_with_ingredients(["Total Fiber", "Fiber"])
# print("Meals excluding nutrients with ingredients\n", meals)