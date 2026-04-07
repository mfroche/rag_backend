import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import re

from rag.services.patient_docs_retriever import get_patient_food_intake


# Get patient info 
def get_patient_info(patient):
    patient_sex = patient.get("sex")
    patient_age = patient.get("age")
    patient_height_cm = patient.get("height_cm")
    patient_weight_kg = patient.get("weight_kg")
    patient_bmi = patient.get("bmi")
    patient_activity_level = patient.get("activity_level")

    patient_info = (
        f"The LTC patient is a {patient_age}-year-old {patient_sex} with a height of {patient_height_cm} cm and a weight of {patient_weight_kg} kg, resulting in a BMI of {patient_bmi}. "
        f"The LTC patient has a {patient_activity_level} physical activity level."
    )
    return patient_info


# Dates
def get_current_month():
    """Return the current month as an integer (1–12)."""
    return datetime.now(ZoneInfo("Asia/Taipei")).month

def get_current_year():
    """Return the current year as a 4-digit integer."""
    return datetime.now(ZoneInfo("Asia/Taipei")).year

def get_current_month_str():
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m")

def get_current_year_str():
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y")

def get_current_month_name():
    curdate = datetime.now(ZoneInfo("Asia/Taipei"))
    return curdate.strftime("%B")

def get_dates_in_current_month():
    """Return a list of all dates in the current month in YYYY-MM-DD format."""
    curdate = datetime.now(ZoneInfo("Asia/Taipei"))
    year = curdate.year
    month = curdate.month

    days_in_month = calendar.monthrange(year, month)[1]

    return [
        f"{year}-{month:02d}-{day:02d}"
        for day in range(1, days_in_month + 1)
    ]

def get_food_intake_results_in_curmonth(dates_in_current_month, pid):
    monthly_results = []

    for date in dates_in_current_month:
        fi_results = get_patient_food_intake(pid, date)

        monthly_results.append({
            "date": date,
            "food_intakes": fi_results if fi_results else "Missing"
        })

    return monthly_results


def create_monthly_food_intake_context(fi_results_in_curmonth_list, date_format="month_day"):
    monthly_context = []

    for day_result in fi_results_in_curmonth_list:
        date_str = day_result["date"]
        date_obj = datetime.fromisoformat(date_str)
        
        # Format date
        if date_format == "month_day":
            formatted_date = date_obj.strftime("%B %-d")  # March 1
        else:
            formatted_date = date_obj.strftime("%Y-%m-%d")  # 2026-03-01
        
        food_intakes = day_result["food_intakes"]
        day_context = f"{formatted_date}:\n"

        if food_intakes == "Missing":
            day_context += "- Missing"
        else:
            for doc, metadata in food_intakes:
                day_context += f"- {doc}\n"

        monthly_context.append(day_context.strip())

    return "\n\n".join(monthly_context)



# ==================================
# Food Intake Calculation
# ==================================

def calculate_food_item_intake(food_intake_docs, debug=False):
    """
    Calculate food intake per item in volume (mL) based on before and after meal data. 
    """

    # 1. Organize food intake ,first, by meal_time (lunch/dinner) and ,second, by meal_phase (before/after)
    meals = {} 

    for doc in food_intake_docs:
        metadata = doc.get("metadata", {})
        meal_time = metadata.get("meal_time_en")
        meal_phase = metadata.get("meal_phase_en")
        food_items = metadata.get("food_items", [])

        if meal_time not in meals:
            meals[meal_time] = {"before": {}, "after": {}}

        for item in food_items:
            food_class_name = item.get("food_class")
            volume_ml = item.get("volume_ml", 0)
            
            # CASE 1: Multiple instances of the same food class (e.g., chicken_1, chicken_2) → aggregate volume under the same food class name (e.g., chicken)
            # [For multi-instance food_class handling] Normalize food_class_name by removing numeric suffixes (e.g., chicken_1 -> chicken)
            normalized_name = re.sub(r'_\d+$', '', food_class_name)

            # Aggregate volume
            if normalized_name in meals[meal_time][meal_phase]:
                meals[meal_time][meal_phase][normalized_name] += volume_ml
            else:
                meals[meal_time][meal_phase][normalized_name] = volume_ml

    if debug:
        print("[STEP 1] Organize data")
        print("Meals structure:", meals)


    # 2. Compute intake per food item (before - after)
    by_meal_result = {}
    total_intake = {}

    for meal_time, meal_phases in meals.items():
        before_items = meal_phases.get("before", {})
        after_items = meal_phases.get("after", {})

        # [For no after meal_phase case handling] If after meal data is missing, assume all before meal volume is consumed (i.e., after meal volume = 0)
        # CASE 2: No AFTER meal phase → everything consumed
        if not after_items:
            after_items = {food: 0 for food in before_items}

        # [For new food items in after meal_phase handling] If after meal has detected food items not present in before meal, assume they were not part of the original meal and thus not consumed. Therefore, disregard new items in after meal for intake calculation)
        # CASE 3: AFTER exists but has new food items not in BEFORE → disregard new items 
        after_items = {
            food: vol for food, vol in after_items.items()
            if food in before_items
        }

        # CASE 4: AFTER exists but missing some foods → assume fully consumed
        for food in before_items:
            if food not in after_items:
                after_items[food] = 0

        all_foods = set(before_items.keys()) # Creates a set of all food items to be processed for intake calculation.
        meal_result = {}

        for food in all_foods:
            before_val = before_items.get(food, 0)
            after_val = after_items.get(food, 0)
            intake = before_val - after_val

            # Prevent negative intake (optional safety)
            intake = max(intake, 0)
            meal_result[food] = intake

            # Aggregate total intake across meals
            total_intake[food] = total_intake.get(food, 0) + intake

        by_meal_result[meal_time] = meal_result

    if debug:
        print("\n[STEP 2] Intake calculation")
        print("By meal:", by_meal_result)
        print("Aggregated total intake:", total_intake)


    # 3. Format total intake per food item across all meals
    total_result = {
        food: round(volume_ml, 1) 
        for food, volume_ml in total_intake.items()
    }

    return {
        "by_meal": by_meal_result,
        "aggregated_total": total_result
    }


# FORMAT AGGREGATED TOTAL INTAKE RESULTS
def format_calculated_intakes(aggregated_total_intake):
    return ", ".join(
        f"{volume} ml of {food}"
        for food, volume in aggregated_total_intake.items()
    )