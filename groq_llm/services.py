# services.py is where functions used in the views are defined. This keeps the views clean and focused on handling HTTP requests and responses, while the services handle the business logic and interactions with external APIs or databases. 

from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def ask_groq_llm(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": "你是一個專業營養助理。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
