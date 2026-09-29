"""
Unit tests for Cloudinary Uploader and CDN Asset Hosting.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.config import AppConfig, CloudinarySettings, load_config
from src.image_generator.cloudinary_uploader import CloudinaryUploader


@pytest.fixture(autouse=True)
def clean_cloudinary_env(monkeypatch):
    """Ensure local environment variables do not bleed into unit tests."""
    monkeypatch.delenv("CLOUDINARY_URL", raising=False)
    monkeypatch.delenv("CLOUDINARY_CLOUD_NAME", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_SECRET", raising=False)


def test_cloudinary_not_configured_when_empty():
    """Test that Cloudinary reports unconfigured when keys are empty."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        cloud_name="",
        api_key="",
        api_secret="",
        url=""
    )
    uploader = CloudinaryUploader(config=config)
    assert uploader.is_configured() is False


def test_cloudinary_configured_with_url():
    """Test Cloudinary configuration via CLOUDINARY_URL."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        url="cloudinary://123456789:abcdef_secret@my-cloud-name"
    )
    uploader = CloudinaryUploader(config=config)
    assert uploader.is_configured() is True
    assert uploader.cloud_name == "my-cloud-name"


def test_cloudinary_configured_with_individual_keys():
    """Test Cloudinary configuration via cloud_name, api_key, api_secret."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        cloud_name="my-personal-cloud",
        api_key="987654321",
        api_secret="my_super_secret"
    )
    uploader = CloudinaryUploader(config=config)
    assert uploader.is_configured() is True
    assert uploader.cloud_name == "my-personal-cloud"


def test_cloudinary_upload_simulated_when_dry_run():
    """Test simulated upload during dry-run."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        cloud_name="my-cloud",
        api_key="123",
        api_secret="abc"
    )
    uploader = CloudinaryUploader(config=config)
    
    result = uploader.upload_image(
        image_source="data/generated_images/mock_123.png",
        cycle_id="cycle_999",
        dry_run=True
    )
    
    assert result["status"] == "simulated"
    assert "res.cloudinary.com/my-cloud" in result["secure_url"]
    assert "linkedin_post_cycle_999" in result["secure_url"]


def test_cloudinary_upload_live_mocked():
    """Test successful Cloudinary upload with mocked SDK response."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        cloud_name="production-cloud",
        api_key="valid_key",
        api_secret="valid_secret"
    )
    uploader = CloudinaryUploader(config=config)

    mock_upload_resp = {
        "secure_url": "https://res.cloudinary.com/production-cloud/image/upload/v1720000000/linkedin_automation/post_abc.png",
        "url": "http://res.cloudinary.com/production-cloud/image/upload/v1720000000/linkedin_automation/post_abc.png",
        "public_id": "linkedin_automation/post_abc",
        "format": "png",
        "bytes": 204850,
        "width": 1024,
        "height": 1024,
        "created_at": "2026-09-29T15:00:00Z"
    }

    with patch("cloudinary.uploader.upload", return_value=mock_upload_resp) as mock_upload:
        result = uploader.upload_image(
            image_source="data/generated_images/test.png",
            cycle_id="cycle_abc",
            dry_run=False
        )

        assert result["status"] == "uploaded"
        assert result["secure_url"] == "https://res.cloudinary.com/production-cloud/image/upload/v1720000000/linkedin_automation/post_abc.png"
        assert result["public_id"] == "linkedin_automation/post_abc"
        assert result["bytes"] == 204850
        mock_upload.assert_called_once()


def test_cloudinary_fallback_on_remote_url_error():
    """Test that failed upload of remote image falls back to original URL gracefully."""
    config = load_config()
    config.cloudinary = CloudinarySettings(
        enabled=True,
        cloud_name="prod-cloud",
        api_key="key",
        api_secret="secret"
    )
    uploader = CloudinaryUploader(config=config)

    with patch("cloudinary.uploader.upload", side_effect=Exception("Cloudinary API Timeout")):
        result = uploader.upload_image(
            image_source="https://dashscope-cdn.aliyuncs.com/temp_image.png",
            cycle_id="cycle_err",
            dry_run=False
        )
        assert result["status"] == "fallback"
        assert result["secure_url"] == "https://dashscope-cdn.aliyuncs.com/temp_image.png"
