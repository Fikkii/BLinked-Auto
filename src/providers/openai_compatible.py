"""
Generic OpenAI-Compatible LLM Provider.
Enables instant compatibility with OpenRouter, Groq, DeepSeek, Ollama, vLLM, Together, etc.
"""

import time
from typing import Any, Dict, Optional
import requests
import urllib3.util.connection as urllib3_cn

urllib3_cn.HAS_IPV6 = False

from src.logger import logger
from src.providers.base import BaseLLMProvider, LLMResponse


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic OpenAI API compatible client."""

    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL = "https://api.openai.com/v1"

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
        """Generates content via standard /chat/completions endpoint."""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
        logger.info(f"[cyan]Calling OpenAI-Compatible API ({self.model_name} at {self.base_url})...[/cyan]")

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds
            )
            latency = time.time() - start_time

            if response.status_code != 200:
                error_msg = f"API Error [{response.status_code}]: {response.text}"
                logger.error(f"[red]{error_msg}[/red]")
                raise RuntimeError(error_msg)

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"API returned no choices: {data}")

            generated_text = choices[0].get("message", {}).get("content", "").strip()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            logger.info(f"[green]✓ Response generated in {latency:.2f}s (Tokens: {total_tokens or 'N/A'})[/green]")

            return LLMResponse(
                content=generated_text,
                model=self.model_name,
                provider="openai_compatible",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                raw_response=data
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"[red]Network request failed: {e}[/red]")
            raise RuntimeError(f"Connection error: {e}") from e

    def validate_credentials(self) -> bool:
        """Validates credentials with a minimal query."""
        try:
            resp = self.generate(prompt="Ping test. Reply with 'OK'.", max_tokens=10)
            return bool(resp.content)
        except Exception as e:
            logger.warning(f"Credential validation failed: {e}")
            return False
