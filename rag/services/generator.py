import ollama


# ==================================
# Ask Local LLM (Ollama)
# ==================================
def ask_ollama_llm(prompt):
    response = ollama.chat(
        model="gemma3:4b", # model="qwen:7b", 
        messages=[
            {"role": "system", "content": "你是一個專業營養助理。"},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"].strip()



# def ask_llm(prompt: str):
#     response = ollama.chat(
#         model="gemma3:4b", # model="qwen:7b", 
#         messages=[
#             {"role": "system", "content": "你是一個專業營養助理。"},
#             {"role": "user", "content": prompt}
#         ],
#         options={
#             "temperature": 0
#         }
#     )

#     return response["message"]["content"].strip()




# ==================================
# Ask Groq LLM
# ==================================
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def ask_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "你是一個專業營養助理。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()



def ask_groq_llm_with_token_limit(prompt: str, token_limit):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "你是一個專業營養助理。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=token_limit,  # Limits the response to determined tokens
    )

    return response.choices[0].message.content.strip()