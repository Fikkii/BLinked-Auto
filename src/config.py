"""
Configuration loader and validator for the LinkedIn Buffer Automation engine.
Merges environment variables (.env) with YAML configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file automatically
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class BufferSettings(BaseModel):
    api_base_url: str = "https://api.buffer.com"
    access_token: str = Field(default_factory=lambda: os.getenv("BUFFER_ACCESS_TOKEN", ""))
    profile_id: Optional[str] = Field(default_factory=lambda: os.getenv("BUFFER_PROFILE_ID", "") or None)
    default_service: str = "linkedin"
    draft_mode: bool = True
    timeout_seconds: int = 30


class PipelineSettings(BaseModel):
    state_file: str = "data/state.json"
    history_file: str = "data/history.jsonl"
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    auto_resume: bool = True


class ProviderDetail(BaseModel):
    default_model: str
    fallback_model: Optional[str] = None
    base_url: str
    api_key_env: str
    alt_api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None


class LLMSettings(BaseModel):
    active_provider: str = Field(default_factory=lambda: os.getenv("ACTIVE_LLM_PROVIDER", "gemini"))
    model: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_MODEL", "") or None)
    temperature: float = 0.7
    max_tokens: int = 1800
    top_p: float = 0.95
    timeout_seconds: int = 45
    providers: Dict[str, ProviderDetail] = Field(default_factory=dict)


class AppSettings(BaseModel):
    name: str = "Buffer LinkedIn Content Automation"
    version: str = "1.0.0"
    dry_run: bool = Field(default_factory=lambda: os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


class ImageGenerationSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.getenv("ENABLE_IMAGE_GENERATION", "true").lower() in ("true", "1", "yes"))
    provider: str = "qwen"
    model: str = Field(default_factory=lambda: os.getenv("QWEN_IMAGE_MODEL", "qwen-image-3.0"))
    fallback_model: Optional[str] = "qwen-image-3.0-pro"
    standby_photo_path: str = Field(default_factory=lambda: os.getenv("STANDBY_PHOTO_PATH", "data/standby/personal_photo.jpg"))
    output_dir: str = "data/generated_images"
    api_base_url: str = Field(
        default_factory=lambda: os.getenv("QWEN_IMAGE_BASE_URL") or os.getenv("QWEN_BASE_URL") or "https://token-plan.maas.qwencloudapi.com"
    )
    timeout_seconds: int = 60
    prompt_extend: bool = True


class CloudinarySettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: os.getenv("ENABLE_CLOUDINARY", "true").lower() in ("true", "1", "yes"))
    cloud_name: str = Field(default_factory=lambda: os.getenv("CLOUDINARY_CLOUD_NAME", ""))
    api_key: str = Field(default_factory=lambda: os.getenv("CLOUDINARY_API_KEY", ""))
    api_secret: str = Field(default_factory=lambda: os.getenv("CLOUDINARY_API_SECRET", ""))
    url: str = Field(default_factory=lambda: os.getenv("CLOUDINARY_URL", ""))
    folder: str = "linkedin_automation"


class AppConfig(BaseModel):
    app: AppSettings
    pipeline: PipelineSettings
    buffer: BufferSettings
    llm: LLMSettings
    image_generation: ImageGenerationSettings = Field(default_factory=ImageGenerationSettings)
    cloudinary: CloudinarySettings = Field(default_factory=CloudinarySettings)
    persona: Dict[str, Any] = Field(default_factory=dict)
    prompts: Dict[str, Any] = Field(default_factory=dict)


    def get_api_key_for_provider(self, provider_name: str) -> Optional[str]:
        """Retrieves API key from environment for a given provider."""
        provider_name = provider_name.lower()
        if provider_name == "gemini":
            return os.getenv("GEMINI_API_KEY")
        elif provider_name == "fireworks":
            return os.getenv("FIREWORKS_API_KEY")
        elif provider_name == "qwen":
            return os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        elif provider_name == "openai_compatible":
            return os.getenv("OPENAI_API_KEY")
        return None

    def get_base_url_for_provider(self, provider_name: str) -> str:
        """Retrieves base URL for provider, with custom env override support."""
        provider_name = provider_name.lower()
        if provider_name == "openai_compatible":
            custom_url = os.getenv("CUSTOM_LLM_BASE_URL")
            if custom_url:
                return custom_url
        elif provider_name == "qwen":
            qwen_url = os.getenv("QWEN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
            if qwen_url:
                return qwen_url
        provider_cfg = self.llm.providers.get(provider_name)
        if provider_cfg and provider_cfg.base_url:
            return provider_cfg.base_url
        return ""


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Safely loads a YAML file if it exists."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config() -> AppConfig:
    """Loads and merges all YAML configs and environment variables."""
    config_yaml_path = CONFIGS_DIR / "config.yaml"
    persona_yaml_path = CONFIGS_DIR / "persona.yaml"
    prompts_yaml_path = CONFIGS_DIR / "prompts.yaml"

    raw_config = load_yaml_file(config_yaml_path)
    persona_data = load_yaml_file(persona_yaml_path)
    prompts_data = load_yaml_file(prompts_yaml_path)

    app_data = raw_config.get("app", {})
    pipeline_data = raw_config.get("pipeline", {})
    buffer_data = raw_config.get("buffer", {})
    llm_data = raw_config.get("llm", {})
    img_data = raw_config.get("image_generation", {})
    cloudinary_data = raw_config.get("cloudinary", {})

    app_settings = AppSettings(**app_data)
    pipeline_settings = PipelineSettings(**pipeline_data)
    buffer_settings = BufferSettings(**buffer_data)
    image_generation_settings = ImageGenerationSettings(**img_data)
    cloudinary_settings = CloudinarySettings(**cloudinary_data)

    # Process providers dict in LLM
    providers_dict = {}
    for p_name, p_data in llm_data.get("providers", {}).items():
        providers_dict[p_name] = ProviderDetail(**p_data)
    
    llm_settings = LLMSettings(
        active_provider=os.getenv("ACTIVE_LLM_PROVIDER", llm_data.get("active_provider", "gemini")),
        model=os.getenv("LLM_MODEL") or llm_data.get("model"),
        temperature=llm_data.get("temperature", 0.7),
        max_tokens=llm_data.get("max_tokens", 1800),
        top_p=llm_data.get("top_p", 0.95),
        timeout_seconds=llm_data.get("timeout_seconds", 45),
        providers=providers_dict,
    )

    return AppConfig(
        app=app_settings,
        pipeline=pipeline_settings,
        buffer=buffer_settings,
        llm=llm_settings,
        image_generation=image_generation_settings,
        cloudinary=cloudinary_settings,
        persona=persona_data,
        prompts=prompts_data,
    )

