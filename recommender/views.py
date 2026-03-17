from django.shortcuts import render

import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import uuid

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


# import services from rag/services/xxx.py

# HPA RAG Services
from rag.services.hpa_retriever_services import build_prompt, build_rag_context, qdrant_search, retrieve_all, retrieve_text

# Patient Docs RAG Services
from rag.services.patient_docs_sql_ingestor import embed_doc, ingest_patient_food_intake_doc
from rag.services.patient_docs_retriever import get_patient_dietary_targets, get_patient_food_intake, vector_search_patient_docs_chinese, vector_search_patient_docs_english, build_prompt_for_patient_docs, vector_search_patient_docs
from rag.services.generator import ask_groq_llm_with_token_limit, ask_ollama_llm, ask_llm

from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rag.services.patient_docs_sql_ingestor import embed_doc, qd_client

# EMYEEEEL'S FOOD INTAKE BACKEND URL
FOOD_INTAKE_BACKEND_URL = "https://h3vkhzth-8000.asse.devtunnels.ms/api/"

# Create your views here.
# Daily Intake recommender with the ingredients & nutrients
class DailyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # Get relevant patient docs
            query = f"Food intake records of LTC patient in room {room_number} bed {bed_number} on this date {curdate}"
            food_intake_results = vector_search_patient_docs(query, patient_id, top_k=5)
            dri_results = get_patient_dietary_targets(patient_id,)
            patient_info = get_patient_info(patient)
            food_intake_res = get_patient_food_intake(patient_id, curdate,)

            # Hpa context
            hpa_macros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
"""
            hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

            hpa_micros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
"""
            hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

            
            # Chunks
            chunks = [
                {
                    "result": i + 1,
                    "score": score,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata, score) in enumerate(food_intake_results)
            ]

            food_intake_chunks = [
                {
                    "result": i + 1,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata) in enumerate(food_intake_res)
            ]

            hpa_macros_chunks = hpa_macros_results
            hpa_micros_chunks = hpa_micros_results

            # Build context 
            patient_curdate_food_intake_context = "\n\n- ".join([doc for doc, metadata, score in food_intake_results])
            food_intake_context = "\n\n- ".join([doc for doc, metadata in food_intake_res])
            hpa_macros_context = build_rag_context(hpa_macros_results)
            hpa_micros_context = build_rag_context(hpa_micros_results)

            # Build prompt
            prompt = f"""
Based on the following information, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {curdate}.

相關資訊：
Patient Information:
{patient_info}

Patient's specific dietary targets:
{dri_results}

Patient food intake context:
{food_intake_context}

Nutrition guidelines context:
Macronutrients:
{hpa_macros_context}

Micronutrients:
{hpa_micros_context}

回應規則：
- If there is no food intake record for {curdate}, politely explain that dietary recommendations cannot be provided because no intake was recorded for that day.
- Even if there is one meal entry, give recommendations.
- 請言簡意賅。回覆字數應少於400個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
"""

            # Pass context
            # response = ask_llm_with_token_limit(prompt, token_limit=200)
            response = ask_ollama_llm(prompt)
            # response = ask_ollama_llm(prompt)
            return Response(
                {
                    "response": response,
                    "food_intake_chunks": food_intake_chunks,
                    "chunks": chunks,
                    "dri_results": dri_results,
                    "hpa_macros_chunks": hpa_macros_chunks,
                    "hpa_micros_chunks": hpa_micros_chunks,
                    "date": curdate,
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        


# Weekly Intake recommender with the ingredients & nutrients
class WeeklyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 

            # Current date
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()

            # Previous 6 days            
            curdate_str      = curdate.strftime("%Y-%m-%d")
            day_minus_1_str  = (curdate - timedelta(days=1)).strftime("%Y-%m-%d")
            day_minus_2_str  = (curdate - timedelta(days=2)).strftime("%Y-%m-%d")
            day_minus_3_str  = (curdate - timedelta(days=3)).strftime("%Y-%m-%d")
            day_minus_4_str  = (curdate - timedelta(days=4)).strftime("%Y-%m-%d")
            day_minus_5_str  = (curdate - timedelta(days=5)).strftime("%Y-%m-%d")
            day_minus_6_str  = (curdate - timedelta(days=6)).strftime("%Y-%m-%d")


            # Get relevant patient docs
            curdate_results = get_patient_food_intake(patient_id, day_minus_5_str)
            day_minus_1_results = get_patient_food_intake(patient_id, day_minus_1_str)
            day_minus_2_results = get_patient_food_intake(patient_id, day_minus_2_str)
            day_minus_3_results = get_patient_food_intake(patient_id, day_minus_3_str)
            day_minus_4_results = get_patient_food_intake(patient_id, day_minus_4_str)
            day_minus_5_results = get_patient_food_intake(patient_id, day_minus_5_str)
            day_minus_6_results = get_patient_food_intake(patient_id, day_minus_6_str)
            
            dri_results = get_patient_dietary_targets(patient_id,)
            patient_info = get_patient_info(patient)

            # Hpa context
            hpa_macros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
"""
            hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

            hpa_micros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
"""
            hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

            
            # Chunks
            weekly_food_intakes_debug = {
                curdate_str: format_food_intakes_chunks(curdate_results),
                day_minus_1_str: format_food_intakes_chunks(day_minus_1_results),
                day_minus_2_str: format_food_intakes_chunks(day_minus_2_results),
                day_minus_3_str: format_food_intakes_chunks(day_minus_3_results),
                day_minus_4_str: format_food_intakes_chunks(day_minus_4_results),
                day_minus_5_str: format_food_intakes_chunks(day_minus_5_results),
                day_minus_6_str: format_food_intakes_chunks(day_minus_6_results),
            }

            hpa_macros_chunks = hpa_macros_results
            hpa_micros_chunks = hpa_micros_results

            # Build context 
            curdate_context = "\n\n- ".join([doc for doc, metadata in curdate_results]) if curdate_results else "Missing"
            day_minus_1_context = "\n\n- ".join([doc for doc, metadata in day_minus_1_results]) if day_minus_1_results else "Missing"
            day_minus_2_context = "\n\n- ".join([doc for doc, metadata in day_minus_2_results]) if day_minus_2_results else "Missing"
            day_minus_3_context = "\n\n- ".join([doc for doc, metadata in day_minus_3_results]) if day_minus_3_results else "Missing"
            day_minus_4_context = "\n\n- ".join([doc for doc, metadata in day_minus_4_results]) if day_minus_4_results else "Missing"
            day_minus_5_context = "\n\n- ".join([doc for doc, metadata in day_minus_5_results]) if day_minus_5_results else "Missing"
            day_minus_6_context = "\n\n- ".join([doc for doc, metadata in day_minus_6_results]) if day_minus_6_results else "Missing"

            hpa_macros_context = build_rag_context(hpa_macros_results)
            hpa_micros_context = build_rag_context(hpa_micros_results)

            # Build prompt
            prompt = f"""
Based on the following information and food intake records in the last week, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number}.

相關資訊：
Patient Information:
{patient_info}

Patient dietary targets:
{dri_results}

Patient food intake context:
6 Days ago ({day_minus_6_str}): 
{day_minus_6_context}

5 Days ago ({day_minus_5_str}): 
{day_minus_5_context}

4 Days ago ({day_minus_4_str}): 
{day_minus_4_context}

3 Days ago ({day_minus_3_str}): 
{day_minus_3_context}

2 Days ago ({day_minus_2_str}): 
{day_minus_2_context}

Yesterday ({day_minus_1_str}): 
{day_minus_1_context}

Today ({curdate}):
{curdate_context}

Nutrition guidelines context:
Macronutrients:
{hpa_macros_context}

Micronutrients:
{hpa_micros_context}

回應規則：
- Identify any patterns in the patient's food intake.
- If only some days are missing, provide recommendations based on the available records and mention the missing days.
- If all the food intake record is missing for today, yesteday, and 2, 3, 4, 5, and 6 days ago, politely explain that dietary recommendations for the week cannot be provided.
- 請言簡意賅。回覆字數應少於400個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
"""

            # Pass context
            # response = ask_llm_with_token_limit(prompt, token_limit=200)
            response = ask_llm(prompt)
            # response = ask_ollama_llm(prompt)
            return Response(
                {
                    "response": response,
                    "weekly_food_intakes": weekly_food_intakes_debug,
                    "dri_results": dri_results,
                    "hpa_macros_chunks": hpa_macros_chunks,
                    "hpa_micros_chunks": hpa_micros_chunks,
                    "date": curdate,
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        


# Monthly Intake recommender with the ingredients & nutrients
class MonthlyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 

            # Current date, month, year
            curmonth = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m")
            cur_month_name = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%B")

            curyear = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y")

            curdate = datetime.now(ZoneInfo("Asia/Taipei")).date()
            no_of_days_in_curmonth = calendar.monthrange(curdate.year, curdate.month)[1]
            list_of_days = list(range(1, no_of_days_in_curmonth + 1))
            dates_in_curmonth_list = get_dates_in_current_month()


            # Get patient docs on cur month
            fi_results_in_curmonth_list = get_food_intake_results_in_curmonth(dates_in_curmonth_list, patient_id)
            dri_results = get_patient_dietary_targets(patient_id,)
            patient_info = get_patient_info(patient)

            # Hpa context
            hpa_macros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
"""
            hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

            hpa_micros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
"""
            hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

            hpa_gen_query = """
長者的營養建議、指引或飲食建議。
"""
            hpa_gen_results = retrieve_text(hpa_gen_query, top_k=7)

            # chunks
            hpa_macros_chunks = hpa_macros_results
            hpa_micros_chunks = hpa_micros_results

            # Build context 
            curmonth_food_intake_context = create_monthly_food_intake_context(fi_results_in_curmonth_list)
            hpa_macros_context = build_rag_context(hpa_macros_results)
            hpa_micros_context = build_rag_context(hpa_micros_results)
            hpa_gen_context = build_rag_context(hpa_gen_results)

            # Build prompt
            prompt = f"""
Based on the following information and food intake records in this month of {cur_month_name} {curyear}, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number}.

相關資訊：
Patient Information:
{patient_info}

Patient daily dietary targets:
{dri_results}

{cur_month_name} food intake context:
{curmonth_food_intake_context}

Nutrition guidelines context:
General:
{hpa_gen_context}

Macronutrients:
{hpa_macros_context}

Micronutrients:
{hpa_micros_context}

回應規則：
- Clearly state that these are dietary recommendations for {cur_month_name}.
- Identify any patterns in the patient's food intake.
- If some days are missing, provide recommendations based on the available records and briefly mention the missing days or weeks in the month.
- If ALL the food intake records is missing for the {no_of_days_in_curmonth} in {cur_month_name} {curyear}, politely explain that dietary recommendations for the month cannot be provided.
- Give safe general answers
- 請言簡意賅。回覆字數應少於400個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
- 必要時請自行推論。
"""

            # Pass context
            # response = ask_llm_with_token_limit(prompt, token_limit=200)
            response = ask_llm(prompt)
            # response = ask_ollama_llm(prompt)
            return Response(
                {
                    "response": response,
                    "monthly_food_intakes": fi_results_in_curmonth_list,
                    "dri_results": dri_results,
                    "hpa_macros_chunks": hpa_macros_chunks,
                    "hpa_micros_chunks": hpa_micros_chunks,
                    "date": curdate,
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



# General recos based on Macronutrients (Lipids/Fats, Protein, Carbohydrates) and Micronutrients(vitamins and minerals)
class GeneralPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
            
            # Get relevant patient docs
            patient_info = get_patient_info(patient)
            dri_results = get_patient_dietary_targets(patient_id,)

            # Hpa context
            hpa_macros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的巨量營養素攝取建議，包括脂質、蛋白質與碳水化合物的營養需求。
"""
            hpa_macros_results = retrieve_all(hpa_macros_query, top_k=7)

            hpa_micros_query = f"""
高齡{patient.get("age")}歲的{patient.get("sex")}長者的微量營養素攝取建議，包括維生素與礦物質的營養需求。
"""
            hpa_micros_results = retrieve_all(hpa_micros_query, top_k=7)

            hpa_gen_query = """
長者的營養建議、指引或飲食建議。
"""
            hpa_gen_results = retrieve_text(hpa_gen_query, top_k=7)

            
            # Chunks
            hpa_macros_chunks = hpa_macros_results
            hpa_micros_chunks = hpa_micros_results
            hpa_gen_chunks= hpa_gen_results

            # Build context 
            hpa_macros_context = build_rag_context(hpa_macros_results)
            hpa_micros_context = build_rag_context(hpa_micros_results)
            hpa_gen_context = build_rag_context(hpa_gen_results)

            # Build prompt
            prompt = f"""
Based on the following information, please provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number}.

相關資訊：
Patient Information:
{patient_info}

Nutrition guidelines context:
General:
{hpa_gen_context}

Macronutrients:
{hpa_macros_context}

Micronutrients:
{hpa_micros_context}

回應規則：
- 請言簡意賅。回覆字數應少於400個字。
- Give safe general answers, if context not clear.
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
- 必要時請自行推論。
"""

            # Pass context
            # response = ask_llm_with_token_limit(prompt, token_limit=200)
            response = ask_llm(prompt)
            # response = ask_ollama_llm(prompt)
            return Response(
                {
                    "response": response,
                    "hpa_macros_chunks": hpa_macros_chunks,
                    "hpa_micros_chunks": hpa_micros_chunks,
                    "hpa_gen_chunks": hpa_gen_chunks,
                    "date": curdate,
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {
                    "detail": "Error generating response", 
                    "error": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )




# - Frame all recommendations so that they can be applied by the caregiver in the LTC facility.
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


# Get patient info 
def get_patient_info(patient):
    patient_sex = patient.get("sex")
    patient_age = patient.get("age")
    patient_height_cm = patient.get("height_cm")
    patient_weight_kg = patient.get("weight_kg")
    patient_bmi = patient.get("bmi")
    patient_activity_level = patient.get("activity_level")

    patient_info = (
        f"The patient is a {patient_age}-year-old {patient_sex} with a height of {patient_height_cm} cm and a weight of {patient_weight_kg} kg, resulting in a BMI of {patient_bmi}. "
        f"The patient has a {patient_activity_level} physical activity level."
    )

    return patient_info

def format_food_intakes_chunks(results):
    return [
        {
            "result": i + 1,
            "chunk": doc,
            "metadata": metadata
        }
        for i, (doc, metadata) in enumerate(results)
    ]



