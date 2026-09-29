"""
Qwen (Alibaba DashScope) LLM Provider Implementation.
Supports Qwen 2.5 72B, Qwen Plus, Qwen Turbo, Qwen Max via OpenAI-compatible DashScope API.
"""

import os
import time
from typing import Any, Dict, Optional
import requests
import urllib3.util.connection as urllib3_cn

urllib3_cn.HAS_IPV6 = False

from src.logger import logger
from src.providers.base import BaseLLMProvider, LLMResponse


class QwenProvider(BaseLLMProvider):
    """Provider implementation for Alibaba Cloud Qwen / DashScope models."""

    DEFAULT_MODEL = "qwen3.6-flash"
    BASE_URL = "https://token-plan.maas.qwencloudapi.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 45,
        **kwargs: Any
    ):
        super().__init__(api_key, model_name, base_url, timeout_seconds, **kwargs)
        raw_key = (api_key or "").strip()
        if raw_key.startswith("k-sp-"):
            raw_key = "s" + raw_key
        self.api_key = raw_key
        self.model_name = model_name or self.DEFAULT_MODEL
        resolved_base_url = (
            base_url
            or os.getenv("QWEN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or self.BASE_URL
        )
        self.base_url = resolved_base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1800,
        top_p: float = 0.95,
        **kwargs: Any
    ) -> LLMResponse:
        """Generates content via Qwen DashScope OpenAI-compatible endpoint."""
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY (or QWEN_API_KEY) is not set. Please add it to your .env file.")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Determine candidate models to try
        candidate_models = [self.model_name]
        if "token-plan" in self.base_url:
            for m in ["qwen3.6-flash", "qwen3.7-plus", "qwen3.8-flash", "qwen3.7-max"]:
                if m not in candidate_models:
                    candidate_models.append(m)
        else:
            for m in ["qwen2.5-72b-instruct", "qwen-plus", "qwen-turbo"]:
                if m not in candidate_models:
                    candidate_models.append(m)

        last_error = None
        for current_model in candidate_models:
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            start_time = time.time()
            logger.info(f"[cyan]Calling Qwen API ({current_model})...[/cyan]")

            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds
                )
                latency = time.time() - start_time

                if response.status_code == 404 and (
                    "model_not_found" in response.text
                    or "Model not exist" in response.text
                ):
                    logger.warning(f"[yellow]Model '{current_model}' not found on {self.base_url}. Trying next model...[/yellow]")
                    continue

                if response.status_code != 200:
                    error_msg = f"Qwen API Error [{response.status_code}]: {response.text}"
                    logger.error(f"[red]{error_msg}[/red]")
                    raise RuntimeError(error_msg)

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"Qwen returned no choices: {data}")

                generated_text = choices[0].get("message", {}).get("content", "").strip()

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

                logger.info(f"[green]✓ Qwen response generated in {latency:.2f}s using {current_model} (Tokens: {total_tokens or 'N/A'})[/green]")

                return LLMResponse(
                    content=generated_text,
                    model=current_model,
                    provider="qwen",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_seconds=latency,
                    raw_response=data
                )

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"[red]Qwen network request failed: {e}[/red]")
                raise RuntimeError(f"Qwen connection error: {e}") from e

        raise RuntimeError(f"All Qwen candidate models exhausted on {self.base_url}.")

    def validate_credentials(self) -> bool:
        """Validates the Qwen API key with a minimal query."""
        if not self.api_key:
            return False
        try:
            resp = self.generate(prompt="Ping test. Reply with 'OK'.", max_tokens=10)
            return bool(resp.content)
        except Exception as e:
            logger.warning(f"Qwen credential validation failed: {e}")
            return False
