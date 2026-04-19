from django.shortcuts import render

import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# HPA RAG Services
from rag.services.hpa_retriever_services import build_prompt, build_rag_context, qdrant_search, retrieve_all, retrieve_text

# Patient Docs RAG Services
from rag.services.patient_docs_retriever import format_food_intakes_docs, get_patient_dietary_targets, get_patient_food_intake, get_patient_profile, get_patient_profile_by_room_and_bed, get_patient_segmented_intake
from rag.services.generator import ask_llm #ask_groq_llm_with_token_limit, ask_ollama_llm, 

# Recommender Services
from recommender.services import calculate_food_item_intake, create_monthly_food_intake_context, format_calculated_intakes, get_dates_in_current_month, get_food_intake_results_in_curmonth, get_list_of_meals, get_nutrition_remarks, get_nutritional_content_in_json, get_patient_info, recommend_meals

# Qdrant Services
from qdrant_client.models import PointStruct, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

# EMYEEEEL'S FOOD INTAKE BACKEND URL
FOOD_INTAKE_BACKEND_URL = "https://h3vkhzth-8000.asse.devtunnels.ms/api/"



# Daily Intake recommender
class DailyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            curdate = datetime.now().strftime("%Y-%m-%d")
            # curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # 1. INFORMATION RETRIEVAL
            # 1.1 Get patient info & patient profile doc
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            age = patient.get("age")
            sex = patient.get("sex")
            # patient_profile_doc = get_patient_profile(pk)[0]
            patient_profile_doc= get_patient_info(patient)

            # 1.2 Get patient dietary target doc
            dri_target_doc = get_patient_dietary_targets(patient_id)[0]

            # 1.3 Get Food Intake docs
                # a. Before and After Segmented Meal Records
            food_intake_res = get_patient_segmented_intake(patient_id, curdate)

                # b. Build food intake context for prompt
            food_intake_context = "\n\n - ".join([doc for doc, metadata in food_intake_res])

                # c. Format food intake docs for response
            food_intake_docs = format_food_intakes_docs(food_intake_res)

                # d. Calculate food intake per item in volume (mL) based on before and after meal data.
            food_item_intakes = calculate_food_item_intake(food_intake_docs)
            aggregated_total_intakes = food_item_intakes.get("aggregated_total", {})
            formatted_aggregated_total_intakes = format_calculated_intakes(aggregated_total_intakes)


            # 1.4 HPA docs retrieval using queries
            if not age and not sex:
                descriptor = "elderly"
            elif age and sex:
                descriptor = f"{age} year old {sex}"
            elif age:
                descriptor = f"{age} year old"
            else:
                descriptor = sex

                # a. Protein-related Query
            query_1 = f"""Calculate the combined protein content of {formatted_aggregated_total_intakes} and determine if it meets the daily protein requirement for a {descriptor} patient?"""
            query_1_results = retrieve_all(query_1, top_k= 5)
            query_1_results_context = build_rag_context(query_1_results)

                # b. Carbohydrate-related Query
            query_2 = f"""Calculate the combined carbohydrate content of {formatted_aggregated_total_intakes} and determine if it meets the daily carbohydrate requirement for a {descriptor} patient?"""
            query_2_results = retrieve_all(query_2, top_k= 5)
            query_2_results_context = build_rag_context(query_2_results)

                # c. Lipid-related Query
            query_3 = f"""Calculate the combined lipid content of {formatted_aggregated_total_intakes} and determine if it meets the daily lipid requirement for a {descriptor} patient?"""
            query_3_results = retrieve_all(query_3, top_k= 5)
            query_3_results_context = build_rag_context(query_3_results)


            # 2. CREATE PROMPT
            prompt = f"""
Based on the following information, answer the following queries to provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {curdate}.

Queries:
- Query 1: {query_1}
- Query 2: {query_2}
- Query 3: {query_3}

相關資訊：
Patient Information:
{patient_profile_doc}

Food intakes for {curdate}:
{food_intake_context}

Query 1 Context:
{query_1_results_context}
Query 2 Context:
{query_2_results_context}
Query 3 Context:
{query_3_results_context}

回應規則：
- If there is no food intake record for {curdate}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day
- Even if there is one meal entry, give recommendations.
- 請言簡意賅。回覆字數應少於500個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
"""

            # 3. GENERATE RESPONSE WITH LLM
            response = ask_llm(prompt)

            return Response(
                {
                    "response": response,
                    "final_prompt": prompt,
                    "patient_profile_doc": patient_profile_doc,
                    "dri_target_doc": dri_target_doc,
                    "food_intake_docs": food_intake_docs,
                    "food_item_intakes": food_item_intakes,
                    "query_1_results": query_1_results,
                    "query_2_results": query_2_results,
                    "query_3_results": query_3_results,
                    "date": curdate,
                    "room_number": room_number,
                    "bed_number": bed_number,
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
        


class DailyRecommenderByPatientView(APIView):    
    def get(self, request, pk):
        try:
            curdate = datetime.now().strftime("%Y-%m-%d")
            # curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # 1. INFORMATION RETRIEVAL
            # 1.1 Get patient info & patient profile doc
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            age = patient.get("age")
            sex = patient.get("sex")
            # patient_profile_doc = get_patient_profile(pk)[0]
            patient_profile_doc= get_patient_info(patient)


            # 1.2 Get patient dietary target doc & dietary target results
                # a. Dietary target doc
            dri_target_doc = get_patient_dietary_targets(patient_id)[0]

                # b. Dietary target results
                # link STILL needs to be changed
            recommended_intakes = requests.get(f"{FOOD_INTAKE_BACKEND_URL}/{pk}").json()


            # 1.3 Get Food Intake docs (TODAY'S RECORD)
            food_intake_res = get_patient_segmented_intake(patient_id, curdate)


            # 1.4 
                # a. In each meal time, get food intake volume per food class
            calculated_intake = calculate_food_item_intake(food_intake_res, debug=False)
                # Lunch 
            lunch_intakes = calculated_intake['by_meal'].get('lunch')
            formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
                # Dinner
            dinner_intakes = calculated_intake['by_meal'].get('dinner')
            formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None

                # b. In each meal time, get nutritional content of food intakes
                # Lunch
            lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            lunch_nutritional_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"],
            } 

                # Dinner
            dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            dinner_nutritional_content = {
                "calories_kcal": dinner_nutri_content["calories_kcal"],
                "protein_g": dinner_nutri_content["protein_g"],
                "fats_g": dinner_nutri_content["fats_g"],
                "carbohydrates_g": dinner_nutri_content["carbohydrates_g"],
            } 

                # c. Add total nutritional content for both lunch & dinner 
            total_nutri_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
            } 

                # d. Get nutritional remarks
            nutrition_remarks = {
                "protein_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "protein_g"),
                "fats_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "fats_g"),
                "carbohydrates_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "carbohydrates_g"),
                "calories_kcal": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "calories_kcal"),
            }        


            # 2. CREATE PROMPT & GENERATE RESPONSE WITH LLM; To give meal recommendations based on remarks
                # a. Get meals from database
            meal_names = get_list_of_meals()

                # b. Get meal recommendations
            meal_recos = recommend_meals(meal_names, nutrition_remarks)

            return Response(
                {
                    "response": meal_recos,
                    "lunch_nutritional_content": lunch_nutritional_content,
                    "dinner_nutritional_content": dinner_nutritional_content,
                    "total_nutritional_content": total_nutri_content,
                    "nutrition_remarks": nutrition_remarks,
                    "patient_profile_doc": patient_profile_doc,
                    "dri_target_doc": dri_target_doc,
                    "food_item_intakes": calculated_intake,
                    "date": curdate,
                    "room_number": room_number,
                    "bed_number": bed_number,
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
        


class DailyRecommenderByDummyPatientView(APIView):    
    def get(self, request):
        try:
            curdate = datetime.now().strftime("%Y-%m-%d")
            # curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # 1. INFORMATION RETRIEVAL
            # 1.1 Get patient info & patient profile doc
            # patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            # patient_id = patient.get("id")
            # room_number = patient.get("room_number")
            # bed_number = patient.get("bed_number") 
            # age = patient.get("age")
            # sex = patient.get("sex")
            # patient_profile_doc = get_patient_profile(pk)[0]
            patient_profile_doc= dummy_patient_profile_doc


            # 1.2 Get patient dietary target doc & dietary target results
                # a. Dietary target doc
            dri_target_doc = dummy_dietary_target_doc

                # b. Dietary target results
            recommended_intakes = dummy_recommended_intakes


            # 1.3 Get Food Intake docs (TODAY'S RECORD)
            food_intake_res = dummy_food_intake_docs


            # 1.4 
                # a. In each meal time, get food intake volume per food class
            calculated_intake = calculate_food_item_intake(food_intake_res, debug=False)
                # Lunch 
            lunch_intakes = calculated_intake['by_meal'].get('lunch')
            formatted_lunch_intakes = format_calculated_intakes(lunch_intakes) if lunch_intakes else None
                # Dinner
            dinner_intakes = calculated_intake['by_meal'].get('dinner')
            formatted_dinner_intakes = format_calculated_intakes(dinner_intakes) if dinner_intakes else None

                # b. In each meal time, get nutritional content of food intakes
                # Lunch
            lunch_nutri_content = get_nutritional_content_in_json(formatted_lunch_intakes)
            lunch_nutritional_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"],
            } 

                # Dinner
            dinner_nutri_content = get_nutritional_content_in_json(formatted_dinner_intakes)
            dinner_nutritional_content = {
                "calories_kcal": dinner_nutri_content["calories_kcal"],
                "protein_g": dinner_nutri_content["protein_g"],
                "fats_g": dinner_nutri_content["fats_g"],
                "carbohydrates_g": dinner_nutri_content["carbohydrates_g"],
            } 

                # c. Add total nutritional content for both lunch & dinner 
            total_nutri_content = {
                "calories_kcal": lunch_nutri_content["calories_kcal"] + dinner_nutri_content["calories_kcal"],
                "protein_g": lunch_nutri_content["protein_g"] + dinner_nutri_content["protein_g"],
                "fats_g": lunch_nutri_content["fats_g"] + dinner_nutri_content["fats_g"],
                "carbohydrates_g": lunch_nutri_content["carbohydrates_g"] + dinner_nutri_content["carbohydrates_g"],
            } 

                # d. Get nutritional remarks
            nutrition_remarks = {
                "protein_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "protein_g"),
                "fats_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "fats_g"),
                "carbohydrates_g": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "carbohydrates_g"),
                "calories_kcal": get_nutrition_remarks(recommended_intakes, total_nutri_content, nutrient = "calories_kcal"),
            }        


            # 2. CREATE PROMPT & GENERATE RESPONSE WITH LLM; To give meal recommendations based on remarks
                # a. Get meals from database
            meal_names = meal_names_list

                # b. Get meal recommendations
            meal_recos = recommend_meals(meal_names, nutrition_remarks)

            return Response(
                {
                    "response": meal_recos,
                    "recommended_intakes": recommended_intakes,
                    "lunch_nutritional_content": lunch_nutritional_content,
                    "dinner_nutritional_content": dinner_nutritional_content,
                    "total_nutritional_content": total_nutri_content,
                    "nutrition_remarks": nutrition_remarks,
                    "patient_profile_doc": patient_profile_doc,
                    "dri_target_doc": dri_target_doc,
                    "food_item_intakes": calculated_intake,
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
- If only some days are missing, provide recommendations based on the available records or general recommendationsand mention the missing days.
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



# Food Intake recommender BY DATE
class PatientFoodIntakeRecommenderByDateView(APIView):    
    def get(self, request, pk, date):
        try:
            # 1. INFORMATION RETRIEVAL
            # 1.1 Get patient info & patient profile doc
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            age = patient.get("age")
            sex = patient.get("sex")
            # patient_profile_doc = get_patient_profile(pk)[0]
            patient_profile_doc= get_patient_info(patient)

            # 1.2 Get patient dietary target doc
            dri_target_doc = get_patient_dietary_targets(patient_id)[0]

            # 1.3 Get Food Intake docs
                # a. Before and After Segmented Meal Records
            food_intake_res = get_patient_segmented_intake(patient_id, date,)

                # b. Build food intake context for prompt
            food_intake_context = "\n\n - ".join([doc for doc, metadata in food_intake_res])

                # c. Format food intake docs for response
            food_intake_docs = format_food_intakes_docs(food_intake_res)

                # d. Calculate food intake per item in volume (mL) based on before and after meal data.
            food_item_intakes = calculate_food_item_intake(food_intake_docs)
            aggregated_total_intakes = food_item_intakes.get("aggregated_total", {})
            formatted_aggregated_total_intakes = format_calculated_intakes(aggregated_total_intakes)


            # 1.4 HPA docs retrieval using queries
            if not age and not sex:
                descriptor = "elderly"
            elif age and sex:
                descriptor = f"{age} year old {sex}"
            elif age:
                descriptor = f"{age} year old"
            else:
                descriptor = sex

                # a. Protein-related Query
            query_1 = f"""Calculate the combined protein content of {formatted_aggregated_total_intakes} and determine if it meets the daily protein requirement for a {descriptor} patient?"""
            query_1_results = retrieve_all(query_1, top_k= 5)
            query_1_results_context = build_rag_context(query_1_results)

                # b. Carbohydrate-related Query
            query_2 = f"""Calculate the combined carbohydrate content of {formatted_aggregated_total_intakes} and determine if it meets the daily carbohydrate requirement for a {descriptor} patient?"""
            query_2_results = retrieve_all(query_2, top_k= 5)
            query_2_results_context = build_rag_context(query_2_results)

                # c. Lipid-related Query
            query_3 = f"""Calculate the combined lipid content of {formatted_aggregated_total_intakes} and determine if it meets the daily lipid requirement for a {descriptor} patient?"""
            query_3_results = retrieve_all(query_3, top_k= 5)
            query_3_results_context = build_rag_context(query_3_results)


            # 2. CREATE PROMPT
            prompt = f"""
Based on the following information, answer the following queries to provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {date}.

Queries:
- Query 1: {query_1}
- Query 2: {query_2}
- Query 3: {query_3}

相關資訊：
Patient Information:
{patient_profile_doc}

Food intakes for {date}:
{food_intake_context}

Query 1 Context:
{query_1_results_context}
Query 2 Context:
{query_2_results_context}
Query 3 Context:
{query_3_results_context}


回應規則：
- If there are no food intake records for {date}, refrain from answering queries and politely explain that dietary recommendations cannot be provided because no intake was recorded for that day
- Even if there is one meal entry, give recommendations.
- Answer in English.
"""

            # 3. GENERATE RESPONSE WITH LLM
            response = ask_llm(prompt)

            return Response(
                {
                    "response": response,
                    "final_prompt": prompt,
                    "patient_profile_doc": patient_profile_doc,
                    "dri_target_doc": dri_target_doc,
                    "food_intake_docs": food_intake_docs,
                    "food_item_intakes": food_item_intakes,
                    "query_1_results": query_1_results,
                    "query_2_results": query_2_results,
                    "query_3_results": query_3_results,
                    "date": date,
                    "room_number": room_number,
                    "bed_number": bed_number,
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





# ============================
# DUMMY DATA
# ============================
dummy_food_intake_docs = [
    {
        "result": 1,
        "document": "During lunch (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 83,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 34.305
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 39.2051
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 319.5655
                }
            ]
        }
    },
    {
        "result": 2,
        "document": "During lunch (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "lunch",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10.4233
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 16.1196
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 100.3211
                }
            ]
        }
    },
    {
        "result": 3,
        "document": "During dinner (晚餐), there was a recorded BEFORE meal (Gross) intake of 378.1g and 373.0756ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 24.305 ml), chicken (volume: 29.2051 ml), rice (volume: 319.5655 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "前",
            "meal_phase_en": "before",
            "total_intake_weight": 378.1,
            "total_intake_volume": 100.0,
            "total_estimated_volume": 373.0756,
            "ltc_patient_id": 9,
            "intake_id": 183,
            "estimation_id": 48,
            "food_items": [
                {
                    "id": 89,
                    "food_class": "broccoli",
                    "volume_ml": 30
                },
                {
                    "id": 90,
                    "food_class": "chicken",
                    "volume_ml": 25
                },
                {
                    "id": 88,
                    "food_class": "rice",
                    "volume_ml": 150
                }
            ]
        }
    },
    {
        "result": 4,
        "document": "During dinner (晚餐), there was a recorded AFTER meal (Leftover/Net) intake of 229.96g and 355.8639ml for broccoli, chicken, rice. This meal consists of food items broccoli (volume: 30.4233 ml), chicken (volume: 36.1196 ml), rice (volume: 289.3211 ml).",
        "metadata": {
            "doc_type": "segmented_intake",
            "room_number": "1005",
            "bed_number": "01",
            "date": "2026-03-28",
            "meal_time": "晚餐",
            "meal_time_en": "dinner",
            "meal_phase": "後",
            "meal_phase_en": "after",
            "total_intake_weight": 229.96,
            "total_intake_volume": 39.18,
            "total_estimated_volume": 355.8639,
            "ltc_patient_id": 9,
            "intake_id": 84,
            "estimation_id": 49,
            "food_items": [
                {
                    "id": 92,
                    "food_class": "broccoli",
                    "volume_ml": 10
                },
                {
                    "id": 93,
                    "food_class": "chicken",
                    "volume_ml": 10
                },
                {
                    "id": 91,
                    "food_class": "rice",
                    "volume_ml": 10
                }
            ]
        }
    }
]


dummy_patient_profile_doc = {
    "page_content": "The LTC Patient in Room 101, Bed 2 is a 75-year-old female. They have a BMI of 23.48 and an activity level classified as Low Active.",
    "metadata":
    {
        "doc_type": "patient_profile",
        "room_number": "101",
        "bed_number": "2",
        "ltc_patient_id": 2
    }
}


dummy_dietary_target_doc = {
    "page_content": "The LTC Patient in Room 101, Bed 2 requires a daily dietary target of: 1956.0 kcal, 40.7g of protein, 269.0g of carbohydrates, and 59.9g of fat. Their total water requirement is 2.71ml.",
    "metadata":
    {
        "doc_type": "dietary_target",
        "room_number": "101",
        "bed_number": "2",
        "ltc_patient_id": 2,
    }
}

dummy_recommended_intakes = {
    "calories_kcal": 1956.0, 
    "protein_g": 40.7, 
    "fats_g": 59.9, 
    "carbohydrates_g": 269.0
}


# From actual db
meal_names_list = ['滷肉排', '馬鈴薯炒肉末', '蒜醬麵腸', '炒時蔬', '筍絲豆皮湯', '滷油干魚', '滷豆支', '四季豆炒香腸', '炒時蔬', '紫菜蛋花湯', '什錦烏龍麵', '炒時蔬', '大白菜豆皮湯', '高麗菜粥', '馬鈴薯燉肉', '芋頭蛋黃球', '滷蘿蔔', '炒時蔬', '味噌蛋花湯', '滷雞排', '炒冬粉', '薑絲海帶根', '炒時蔬', '玉米湯', '洋蔥炒豬柳', '燴咖哩', '炒時蔬', '海帶芽湯', '銀斑魚', '沙茶素腰花', '玉米炒蛋', '炒時蔬', '高麗菜湯', '滷雞腿', '紅蘿蔔滷貢丸', '蒜醬百頁豆腐', '炒時蔬', '紫菜蛋花湯', '紅燒獅子頭', '蘿蔔滷豆輪', '馬鈴薯炒蛋', '炒時蔬', '玉米湯', '滷肉排', '海苔丸', '醬拌豆干', '炒時蔬', '冬菜豆芽湯', '碗粿', '筍絲豆皮湯', '皮蛋鹹粥', '日式豬排', '滷麵筋', '薑絲炒木耳', '炒時蔬', '海帶芽湯', '古早味炒麵', '蘿蔔湯', '絲瓜鹹粥', '滷雞排', '紅蘿蔔炒蛋', '馬鈴薯炒肉末', '炒時蔬', '玉米湯', '滷油干魚', '肉末滷油豆腐', '茄汁炒蛋', '炒時蔬', '筍絲豆皮湯', '香腸', '炒冬粉', '馬鈴薯炒肉末', '炒時蔬', '紫菜蛋花湯', '三杯里肌', '洋蔥炒甜不辣', '紅蘿蔔炒蛋', '炒時蔬', '冬菜豆芽湯', '滷雞腿', '蘿蔔滷豆輪', '炒時蔬', '味噌蛋花湯', '什錦米粉', '炒時蔬', '大白菜豆皮湯', '芋頭鹹粥', '滷銀斑魚', '滷筍絲', '蒜醬百頁豆腐', '炒時蔬', '海帶芽湯', '沙茶腿排', '三杯麵腸', '玉米炒蛋', '炒時蔬', '鳳梨苦瓜湯', '紅燒獅子頭', '滷海帶結', '炒時蔬', '玉米湯', '馬鈴薯燉肉', '滷蘿蔔', '炒冬粉', '炒時蔬', '筍絲豆皮湯', '炸無骨雞排', '燴咖哩', '紅蘿蔔炒蛋', '炒時蔬', '紫菜湯', '米糕', '鳳梨苦瓜湯', '高麗菜鹹粥', '紅燒里肌', '醬拌豆干', '炒時蔬', '玉米湯', '雞肉飯', '筍絲豆皮湯', '絲瓜鹹粥', '滷銀斑魚', '滷豆支', '沙茶玉米炒肉末', '炒時蔬', '大白菜豆皮湯']