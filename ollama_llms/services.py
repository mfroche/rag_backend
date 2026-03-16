import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
REQUEST_TIMEOUT = 120  # Increased from 30s to 120s for complex queries
MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds

#=============================================================
# General Ollama LLM
#=============================================================
def get_ollama_llm_response(prompt: str, model_name=None) -> str:
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
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    


#=============================================================
# Deepseek R1 8B
#=============================================================
def get_deepseek_llm_response(prompt: str) -> str:
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")



#=============================================================
# Qwen 8B
#=============================================================
def get_qwen_llm_response(prompt: str) -> str:
    payload = {
        "model": "qwen:7b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")



#=============================================================
# Gemma 3 4B
#=============================================================
def get_gemma3_llm_response(prompt: str) -> str:
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
