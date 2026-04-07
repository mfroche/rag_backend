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
from recommender.services import calculate_food_item_intake, create_monthly_food_intake_context, format_calculated_intakes, get_dates_in_current_month, get_food_intake_results_in_curmonth, get_patient_info

# Qdrant Services
from qdrant_client.models import PointStruct, Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

# EMYEEEEL'S FOOD INTAKE BACKEND URL
FOOD_INTAKE_BACKEND_URL = "https://h3vkhzth-8000.asse.devtunnels.ms/api/"



# Daily Intake recommender
class DailyPatientFoodIntakeRecommenderView(APIView):    
    def get(self, request, pk):
        try:
            # 1. Get current date
            curdate = datetime.now().strftime("%Y-%m-%d")
            # curdate = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")

            # 2. INFORMATION RETRIEVAL
            # 2.1 Get patient docs
            patient = requests.get(f"{FOOD_INTAKE_BACKEND_URL}ltc-patients/{pk}").json()
            patient_id = patient.get("id")
            room_number = patient.get("room_number")
            bed_number = patient.get("bed_number") 
            dri_results = get_patient_dietary_targets(patient_id)

            # 2.2.Get Food Intake docs
            # 1. Before and After Segmented Meal Records
            food_intake_res = get_patient_segmented_intake(patient_id, curdate,)

            # 2. Format food intake docs for response
            food_intake_chunks = [
                {
                    "result": i + 1,
                    "chunk": doc,
                    "metadata": metadata
                }
                for i, (doc, metadata) in enumerate(food_intake_res)
            ]

            # 3. Build food intake context 
            food_intake_context = "\n\n - ".join([doc for doc, metadata in food_intake_res])


            # 2.3. HPA docs
            age = patient.get("age")
            sex = patient.get("sex")

            if not age and not sex:
                descriptor = "elderly patient"
            elif age and sex:
                descriptor = f"{age} year old {sex}"
            elif age:
                descriptor = f"{age} year old"
            else:
                descriptor = sex

            query_1 = f"""What is the combined protein content of 50 ml of chicken, 40 ml of broccoli and does it meet the daily protein requirement for a {descriptor}?"""
            query_1_results = retrieve_all(query_1, top_k=7)


            # Build prompt
            prompt = f"""
Based on the following information, answer the following queries to provide dietary recommendations to the long-term care patient on room {room_number} bed {bed_number} based on food intake record for the day {curdate}.

Queries:
- Query 1: {query_1}

相關資訊：
Patient Information and Dietary Targets:
{dri_results}

Patient food intake context:
{food_intake_context}

Query 1 Context:
{query_1}


回應規則：
- If there is no food intake record for {curdate}, politely explain that dietary recommendations cannot be provided because no intake was recorded for that day.
- Even if there is one meal entry, give recommendations.
- 請言簡意賅。回覆字數應少於500個字。
- 若資訊中無明確答案，請回應「無可用資訊
- 請僅以繁體中文回覆。
"""

            # Pass context to LLM
            response = ask_llm(prompt)

            return Response(
                {
                    "response": response,
                    "final_prompt": prompt,
                    "food_intake_chunks": food_intake_chunks,
                    "dri_results": dri_results,
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




