"""
Google Gemini LLM Provider Implementation.
Supports Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash, etc. via REST API.
"""

import json
import time
from typing import Any, Dict, Optional
import requests
import urllib3.util.connection as urllib3_cn

# Optimize DNS lookup on Linux environments by prioritizing IPv4
urllib3_cn.HAS_IPV6 = False

from src.logger import logger
from src.providers.base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Provider implementation for Google Gemini models."""

    DEFAULT_MODEL = "gemini-3-flash-preview"
    FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-3.8-flash", "gemma-4-26b-a4b-it"]
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 45,
        **kwargs: Any
    ):
        super().__init__(api_key, model_name, base_url, timeout_seconds, **kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.base_url = (base_url or self.BASE_URL).rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1800,
        top_p: float = 0.95,
        **kwargs: Any
    ) -> LLMResponse:
        """Generates content via Gemini generateContent API with automatic fallback."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        # Build prioritized list of models to try
        models_to_try = [self.model_name]
        for m in self.FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None

        for idx, current_model in enumerate(models_to_try):
            endpoint = f"{self.base_url}/models/{current_model}:generateContent"
            start_time = time.time()
            logger.info(f"[cyan]Calling Gemini API ({current_model})...[/cyan]")

            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds
                )
                latency = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError(f"Gemini returned no candidates: {data}")

                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    if not parts:
                        raise RuntimeError(f"Gemini candidate has no content parts: {candidate}")

                    generated_text = parts[0].get("text", "").strip()

                    usage = data.get("usageMetadata", {})
                    prompt_tokens = usage.get("promptTokenCount")
                    completion_tokens = usage.get("candidatesTokenCount")
                    total_tokens = usage.get("totalTokenCount")

                    logger.info(f"[green]✓ Gemini response generated in {latency:.2f}s using {current_model} (Tokens: {total_tokens or 'N/A'})[/green]")

                    return LLMResponse(
                        content=generated_text,
                        model=current_model,
                        provider="gemini",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_seconds=latency,
                        raw_response=data
                    )

                # If temporary high demand (503) or rate limit (429), try next fallback
                if response.status_code in (503, 429) and idx < len(models_to_try) - 1:
                    next_model = models_to_try[idx + 1]
                    logger.warning(
                        f"[yellow]Gemini model '{current_model}' busy ({response.status_code}). "
                        f"Automatically failing over to '{next_model}'...[/yellow]"
                    )
                    time.sleep(1)
                    continue

                error_msg = f"Gemini API Error [{response.status_code}]: {response.text}"
                logger.error(f"[red]{error_msg}[/red]")
                raise RuntimeError(error_msg)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if idx < len(models_to_try) - 1:
                    next_model = models_to_try[idx + 1]
                    logger.warning(f"[yellow]Network timeout on '{current_model}'. Failing over to '{next_model}'...[/yellow]")
                    continue
                last_error = e
                logger.error(f"[red]Gemini network request failed: {e}[/red]")
                raise RuntimeError(f"Gemini connection error: {e}") from e

        if last_error:
            raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")

    def validate_credentials(self) -> bool:
        """Validates the Gemini API key with a minimal query."""
        if not self.api_key:
            return False
        try:
            resp = self.generate(prompt="Ping test. Reply with 'OK'.", max_tokens=10)
            return bool(resp.content)
        except Exception as e:
            logger.warning(f"Gemini credential validation failed: {e}")
            return False
