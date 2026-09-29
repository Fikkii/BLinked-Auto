"""
Abstract Base Class for LLM Providers.
Defines the unified interface for provider-agnostic content generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers (Gemini, Fireworks, Qwen, OpenAI-compatible).
    Ensures seamless plug-and-play swapping of model backends.
    """

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 45,
        **kwargs: Any
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.extra_kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1800,
        top_p: float = 0.95,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generates text given a prompt and optional system instructions.
        Must return an instance of LLMResponse.
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Quick check or lightweight ping to verify that API key is valid.
        """
        pass

    @property
    def provider_name(self) -> str:
        """Returns the canonical name of this provider."""
        return self.__class__.__name__.replace("Provider", "").lower()
