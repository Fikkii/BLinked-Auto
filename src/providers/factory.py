"""
Provider Factory to dynamically instantiate the desired LLM provider.
"""

from typing import Dict, Optional, Type
from src.config import AppConfig
from src.logger import logger
from src.providers.base import BaseLLMProvider
from src.providers.gemini import GeminiProvider
from src.providers.fireworks import FireworksProvider
from src.providers.qwen import QwenProvider
from src.providers.openai_compatible import OpenAICompatibleProvider


class LLMProviderFactory:
    """Factory registry for LLM providers."""

    _PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
        "gemini": GeminiProvider,
        "fireworks": FireworksProvider,
        "qwen": QwenProvider,
        "openai_compatible": OpenAICompatibleProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Registers a custom provider class."""
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider_name: Optional[str] = None,
        config: Optional[AppConfig] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseLLMProvider:
        """
        Creates and configures an LLM provider instance.
        """
        if config is None:
            from src.config import load_config
            config = load_config()

        name = (provider_name or config.llm.active_provider or "gemini").lower()

        if name not in cls._PROVIDERS:
            available = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(f"Unknown LLM provider '{name}'. Available providers: {available}")

        provider_cls = cls._PROVIDERS[name]

        # Resolve API key
        resolved_api_key = api_key or config.get_api_key_for_provider(name) or ""

        # Resolve Model
        provider_detail = config.llm.providers.get(name)
        default_model = provider_detail.default_model if provider_detail else None
        resolved_model = model_name or config.llm.model or default_model

        # Resolve Base URL
        resolved_base_url = base_url or config.get_base_url_for_provider(name)

        if not resolved_api_key and name != "openai_compatible":
            logger.warning(
                f"[yellow]Warning: No API key found for provider '{name}'. "
                f"Make sure to set the corresponding key in your .env file.[/yellow]"
            )

        return provider_cls(
            api_key=resolved_api_key,
            model_name=resolved_model,
            base_url=resolved_base_url or None,
            timeout_seconds=config.llm.timeout_seconds
        )
