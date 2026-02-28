# src/generation/llm_client.py

import os
import requests
from src.config import OPENROUTER_BASE_URL


class LLMClient:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment.")

    def generate(self, prompt: str, temperature: float, max_output_tokens: int) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()