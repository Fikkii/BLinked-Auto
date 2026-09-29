"""
Fireworks AI LLM Provider Implementation.
Supports models like LLaMA 3.3 70B, DeepSeek V3, Qwen 2.5 72B via Fireworks API.
"""

import time
from typing import Any, Dict, Optional
import requests
import urllib3.util.connection as urllib3_cn

urllib3_cn.HAS_IPV6 = False

from src.logger import logger
from src.providers.base import BaseLLMProvider, LLMResponse


class FireworksProvider(BaseLLMProvider):
    """Provider implementation for Fireworks AI serverless inference."""

    DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    BASE_URL = "https://api.fireworks.ai/inference/v1"

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
        """Generates content via Fireworks Chat Completions API."""
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY is not set. Please add it to your .env file.")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        start_time = time.time()
        logger.info(f"[cyan]Calling Fireworks AI API ({self.model_name})...[/cyan]")

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds
            )
            latency = time.time() - start_time

            if response.status_code != 200:
                error_msg = f"Fireworks API Error [{response.status_code}]: {response.text}"
                logger.error(f"[red]{error_msg}[/red]")
                raise RuntimeError(error_msg)

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Fireworks returned no choices: {data}")

            generated_text = choices[0].get("message", {}).get("content", "").strip()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            logger.info(f"[green]✓ Fireworks response generated in {latency:.2f}s (Tokens: {total_tokens or 'N/A'})[/green]")

            return LLMResponse(
                content=generated_text,
                model=self.model_name,
                provider="fireworks",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                raw_response=data
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"[red]Fireworks network request failed: {e}[/red]")
            raise RuntimeError(f"Fireworks connection error: {e}") from e

    def validate_credentials(self) -> bool:
        """Validates the Fireworks API key with a minimal query."""
        if not self.api_key:
            return False
        try:
            resp = self.generate(prompt="Ping test. Reply with 'OK'.", max_tokens=10)
            return bool(resp.content)
        except Exception as e:
            logger.warning(f"Fireworks credential validation failed: {e}")
            return False
