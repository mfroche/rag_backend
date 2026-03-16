# =========================================
# 1. Connect to Database
# =========================================
from sqlalchemy import create_engine, text
import pymysql

# Replace with your actual credentials
DB_USER = "root"
DB_PASSWORD = "73502634"
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "food_intakes_db"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
) #the data source interface; creates an abstraction over a database connection


# =========================================
# 2. Document Construction
# =========================================
# Conversion of food intake rows into natural language documents. The document semantically establishes relationships between the patient, the patient's meal intake, the  meal, the ingredients of the meal, and the nutrients found in the ingredients.

import pandas as pd
from datetime import date, datetime


def get_current_date():
    selected_date = date.today()
    return selected_date
    

def get_selected_date(year, month, day):
    selected_date = date(year, month, day) 
    # selected_date = date.today()
    return selected_date


def format_datetime(dt):
    """
    Converts a datetime (string, pandas Timestamp, or datetime object) like
    '2026-02-17 12:00:00' or Timestamp('2026-02-17 12:00:00') to
    '2026年2月17日 12:00 下午' in Traditional Chinese.
    """
    if dt is None or pd.isna(dt):
        return ""
    
    # Convert pandas Timestamp to python datetime
    if isinstance(dt, pd.Timestamp):
        dt_obj = dt.to_pydatetime()
    elif isinstance(dt, datetime):
        dt_obj = dt
    else:  # assume string
        dt_obj = datetime.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
    
    # Hour in 12-hour format
    hour_12 = dt_obj.strftime("%I").lstrip("0") or "0"
    minute = dt_obj.strftime("%M")
    am_pm = dt_obj.strftime("%p")
    
    # Map AM/PM to Traditional Chinese
    am_pm_zh = "上午" if am_pm == "AM" else "下午"
    
    # Format as: 2026年2月17日 12:00 下午
    return f"{dt_obj.year}年{dt_obj.month}月{dt_obj.day}日 {hour_12}:{minute} {am_pm_zh}"

    

# =========================================
# =========================================
def sql_query_food_intake(patient_id, selected_date = date.today()):
    # SQL Query food intake & relevant tables based on current date & specific patient referred in foodintake
    query = """
SELECT 
    fi.id AS food_intake_id,

    -- Patient
    p.id AS ltc_patient_id,
    p.room_number,
    p.bed_number,
    p.age,
    p.sex,
    p.activity_level,
    p.height_cm,
    p.weight_kg,

    -- Meal
    m.id AS meal_id,
    m.meal_name,
    m.meal_time,
    m.day_cycle,
    m.plate_type,

    -- Intake
    fi.recorded_at,
    fi.weight_g,
    fi.volume_ml,

    -- Ingredients (aggregated)
    GROUP_CONCAT(DISTINCT ing.name SEPARATOR ', ') AS ingredients,

    -- Nutrients (aggregated)
    GROUP_CONCAT(DISTINCT nut.name SEPARATOR ', ') AS nutrients

FROM meals_foodintake fi

JOIN patients_ltcpatient p
ON fi.ltc_patient_id = p.id

JOIN meals_meal m
ON fi.meal_id = m.id

LEFT JOIN meals_meal_ingredients mi
ON m.id = mi.meal_id

LEFT JOIN foods_ingredient ing
ON mi.ingredient_id = ing.id

LEFT JOIN foods_ingredient_nutrients inut
ON ing.id = inut.ingredient_id

LEFT JOIN foods_nutrient nut
ON inut.nutrient_id = nut.id

WHERE DATE(fi.recorded_at) = %s
AND fi.ltc_patient_id = %s

GROUP BY fi.id
"""

    selected_date = selected_date # format: year/month/day
    patient_id = patient_id 
    
    p_food_intake_df = pd.read_sql(
        query, 
        engine, 
        params=(selected_date, patient_id)
    )

    # Return
    return p_food_intake_df
    
# Helper to replace None/NaN with blank
def clean(val):
    if val is None:
        return ""
    # If it's a pandas NaN
    try:
        import math
        if math.isnan(val):
            return ""
    except:
        pass
    return str(val)



# =========================================
# =========================================
def create_patient_food_intake_doc(df):
    if df.empty:
        return []

    patient_info = df.iloc[0]
    patient_text = (
        f"病人位於房間 {clean(patient_info['room_number'])} 床號 {clean(patient_info['bed_number'])}，"
        f"年齡 {clean(patient_info['age'])} 歲，性別 {clean(patient_info['sex'])}，"
        f"活動量 {clean(patient_info['activity_level'])}，身高 {clean(patient_info['height_cm'])} 公分，"
        f"體重 {clean(patient_info['weight_kg'])} 公斤。"
    )

    full_doc = patient_text + " 病人的餐點如下："

    meal_texts = []
    for i, (_, row) in enumerate(df.iterrows()):
        prefix = "且" if i > 0 else ""
        text = (
            f"{prefix}病人食用了餐點「{clean(row['meal_name'])}」，"
            f"於 {clean(row['meal_time'])} 進食，屬於第 {clean(row['day_cycle'])} 天循環，"
            f"使用餐盤類型 {clean(row['plate_type'])}，"
            f"包含食材 {clean(row['ingredients'])}，提供營養素 {clean(row['nutrients'])}。"
            f"此餐攝取記錄於 {clean(format_datetime(row['recorded_at']))}，"
            f"重量 {clean(row['weight_g'])} 克，體積 {clean(row['volume_ml'])} 毫升。"
        )
        meal_texts.append(text)

    full_doc += " ".join(meal_texts)
    return full_doc.strip()



# =========================================
# =========================================
def create_patient_food_intake_doc_in_english(df):
    if df.empty:
        return []

    # Get patient info from the first row
    patient_info = df.iloc[0]
    patient_text = (
        f"The patient located in room {clean(patient_info['room_number'])} "
        f"bed {clean(patient_info['bed_number'])} is {clean(patient_info['age'])} years old in age, of {clean(patient_info['sex'])} sex, "
        f"has a \"{clean(patient_info['activity_level'])}\" activity level, height of {clean(patient_info['height_cm'])} centimeters (cm), "
        f"and weight of {clean(patient_info['weight_kg'])} kilograms (kg)."
    )

    # Start document with patient text
    full_doc = patient_text + " The patient had the following meal intakes: "

    # Create semantic sentences for each meal intake
    intake_sentences = []
    for _, row in df.iterrows():
        prefix = " And" if i > 0 else ""
        sentence = (
            f"{prefix} The patient consumed the meal \"{clean(row['meal_name'])}\" "
            f"and was eaten on {clean(row['meal_time'])}, on day cycle {clean(row['day_cycle'])}, and served on plate type {clean(row['plate_type'])}, "
            f"containing ingredients {clean(row['ingredients'])} which provide nutrients {clean(row['nutrients'])}. "
            f"The intake was recorded at {clean(format_datetime(row['recorded_at']))} with a weight of {clean(row['weight_g'])} g and a volume of {clean(row['volume_ml'])} mL."
        )
        intake_sentences.append(sentence)

    # Combine all intake sentences
    full_doc += " ".join(intake_sentences)

    return full_doc.strip()