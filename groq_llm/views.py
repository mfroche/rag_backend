from django.shortcuts import render
import os
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services.groq_services import ask_groq_llm



class GetGroqLLMResponseView(APIView):
    def post(self, request):
        prompt = request.data.get("prompt")

        if not prompt:
            return Response( {"error": "Prompt is required."}, status=400 )
        
        try:
            groq_response = ask_groq_llm(prompt)
            return Response({"response": groq_response})
        except Exception as e:
            return Response( {"error": str(e)}, status=500 )

