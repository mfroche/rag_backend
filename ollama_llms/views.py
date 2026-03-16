from langchain_ollama import ChatOllama
from rest_framework.views import APIView
from rest_framework.response import Response
import requests

# Load local LLM model from Ollama server
# LLM Model: gemma3:4b
OLLAMA_URL = "http://localhost:11434/api/generate"


class GetOllamaLLMResponseView(APIView):
    def post(self, request, model_name):
        prompt = request.data.get("prompt")

        if not prompt:
            return Response(
                {"error": "Prompt is required."},
                status=400
            )
        
        # Load selected LLM model from Ollama server
        if model_name == "deepseek":
            model_name = "deepseek-r1:8b"
        elif model_name == "qwen":
            model_name = "qwen:7b"
        elif model_name == "gemma3":
            model_name = "gemma3:4b"
        else:
            model_name = "gemma3:4b"  # default OR if model_name == None/empty in the urls.py

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            ollama_response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=30
            )
            ollama_response.raise_for_status()

            data = ollama_response.json()
            llm_output = data.get("response", "")

            return Response({"response": llm_output})

        except requests.exceptions.RequestException as e:
            return Response(
                {"error": str(e)},
                status=500
            )
        