import pandas as pd


# load excel
food_df = pd.read_excel("/home/k503/下載/Spoonfull/rag_backend/services/food_seg_nutri_content.xlsx")


def calculate_nutritional_content_of_food_item(food_item_name, food_item_volume):
    nutritional_content = {
        'calories_kcal': 0.0,
        'protein_g': 0.0,
        'carbohydrates_g': 0.0,
        'fats_g': 0.0,
        'fiber_g': 0.0
    }

    row = food_df[food_df['food_name'] == food_item_name]

    if row.empty:
        return nutritional_content

    factor = food_item_volume / 100

    nutritional_content['calories_kcal'] = row.iloc[0]['energy_kcal'] * factor
    nutritional_content['protein_g'] = row.iloc[0]['protein_g'] * factor
    nutritional_content['carbohydrates_g'] = row.iloc[0]['carbohydrate_g'] * factor
    nutritional_content['fats_g'] = row.iloc[0]['fat_g'] * factor
    nutritional_content['fiber_g'] = row.iloc[0]['fiber_g'] * factor

    return nutritional_content


def compute_total_nutrition(food_items):
    total = {
        'calories_kcal': 0.0,
        'protein_g': 0.0,
        'carbohydrates_g': 0.0,
        'fats_g': 0.0,
        'fiber_g': 0.0
    }

    if not food_items:
        return total

    for food_name, volume in food_items.items():
        nutrition = calculate_nutritional_content_of_food_item(food_name, volume)

        for key in total:
            total[key] += nutrition.get(key, 0.0)

    return total


# print(compute_total_nutrition({'broccoli': 40.3, 'rice': 130.2, 'chicken': 70}))

