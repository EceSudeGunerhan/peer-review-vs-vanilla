# src/generation/llm_client.py

import os
import time
import logging
import requests
from src.config import OPENROUTER_BASE_URL, MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMClient:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment.")

    def generate(self, prompt: str, temperature: float, max_output_tokens: int) -> str:
        """
        Send a prompt to the LLM via OpenRouter with exponential backoff retry.

        Retries on:
          - HTTP 429 (rate limit)
          - HTTP 500/502/503/504 (server errors)
          - Connection timeouts

        Returns the model's response text.
        """
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

        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=180,  # 3 min timeout for long generations
                )
                latency = time.time() - t0

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()

                    if not content:
                        raise RuntimeError("Empty response from model.")

                    logger.info(
                        f"[{self.model_name}] OK in {latency:.1f}s "
                        f"(~{len(content)} chars)"
                    )
                    return content

                # Retryable error
                if response.status_code in RETRYABLE_STATUS_CODES:
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"[{self.model_name}] HTTP {response.status_code} "
                        f"(attempt {attempt}/{MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable error
                raise RuntimeError(
                    f"OpenRouter error HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            except requests.exceptions.Timeout:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{self.model_name}] Timeout "
                    f"(attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s..."
                )
                last_error = RuntimeError("Request timed out after 180s")
                time.sleep(delay)
                continue

            except requests.exceptions.ConnectionError as e:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{self.model_name}] Connection error "
                    f"(attempt {attempt}/{MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s..."
                )
                last_error = RuntimeError(f"Connection error: {e}")
                time.sleep(delay)
                continue

        # All retries exhausted
        raise last_error or RuntimeError(
            f"Failed after {MAX_RETRIES} retries."
        )