"""
Unit tests for Abstract LLM Providers and Factory.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.providers.base import BaseLLMProvider, LLMResponse
from src.providers.gemini import GeminiProvider
from src.providers.fireworks import FireworksProvider
from src.providers.qwen import QwenProvider
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.providers.factory import LLMProviderFactory


def test_gemini_provider_generation():
    """Test Gemini provider payload formatting and response parsing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Sample LinkedIn post from Gemini"}]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 120,
            "totalTokenCount": 170
        }
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        provider = GeminiProvider(api_key="test_gemini_key", model_name="gemini-2.0-flash")
        result = provider.generate(prompt="Write a post", system_prompt="System prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "Sample LinkedIn post from Gemini"
        assert result.provider == "gemini"
        assert result.model == "gemini-2.0-flash"
        assert result.total_tokens == 170
        
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "x-goog-api-key" in call_kwargs["headers"]
        assert call_kwargs["headers"]["x-goog-api-key"] == "test_gemini_key"
        assert call_kwargs["json"]["systemInstruction"]["parts"][0]["text"] == "System prompt"


def test_fireworks_provider_generation():
    """Test Fireworks provider with OpenAI-compatible chat completions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Sample post from Fireworks Llama"}
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 100,
            "total_tokens": 140
        }
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        provider = FireworksProvider(api_key="test_fw_key")
        result = provider.generate(prompt="Write a post")

        assert result.content == "Sample post from Fireworks Llama"
        assert result.provider == "fireworks"
        assert result.total_tokens == 140
        
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test_fw_key"


def test_qwen_provider_generation():
    """Test Qwen provider via DashScope."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Sample post from Qwen 2.5"}
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 90,
            "total_tokens": 120
        }
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        provider = QwenProvider(api_key="test_qwen_key")
        result = provider.generate(prompt="Write a post")

        assert result.content == "Sample post from Qwen 2.5"
        assert result.provider == "qwen"
        assert result.total_tokens == 120


def test_factory_creation():
    """Test provider factory dynamic resolution."""
    gemini = LLMProviderFactory.create(provider_name="gemini", api_key="dummy")
    assert isinstance(gemini, GeminiProvider)

    fireworks = LLMProviderFactory.create(provider_name="fireworks", api_key="dummy")
    assert isinstance(fireworks, FireworksProvider)

    qwen = LLMProviderFactory.create(provider_name="qwen", api_key="dummy")
    assert isinstance(qwen, QwenProvider)

    openai_comp = LLMProviderFactory.create(provider_name="openai_compatible", api_key="dummy")
    assert isinstance(openai_comp, OpenAICompatibleProvider)

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMProviderFactory.create(provider_name="non_existent_provider")
