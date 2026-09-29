"""
Unit tests for Qwen-Image-3.0 Image Generator and Standby Reference Photo handling.
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import requests

from src.config import load_config
from src.image_generator.qwen_image import GeneratedImage, QwenImageGenerator
from src.generator.prompt_builder import PromptBuilder


def _make_valid_image(path: Path, width=300, height=300) -> Path:
    """Creates a valid readable image file for testing image validators."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    img.save(path, format=fmt)
    return path


def test_resolve_standby_photo_explicit(tmp_path):
    """Test resolving standby photo via explicit path."""
    config = load_config()
    photo_file = tmp_path / "custom_headshot.jpg"
    _make_valid_image(photo_file)
    
    generator = QwenImageGenerator(config=config)
    resolved = generator.resolve_standby_photo(custom_path=str(photo_file))
    
    assert resolved == photo_file
    assert resolved.exists()


def test_resolve_standby_photo_configured_and_fallback(tmp_path):
    """Test resolving photo configured in settings and directory auto-discovery."""
    config = load_config()
    standby_dir = tmp_path / "standby"
    standby_dir.mkdir()
    
    candidate = standby_dir / "my_portrait.png"
    _make_valid_image(candidate)
    
    config.image_generation.standby_photo_path = str(standby_dir / "personal_photo.jpg")
    
    generator = QwenImageGenerator(config=config)
    # Configured file does not exist, but auto-discovery in standby_dir finds candidate
    resolved = generator.resolve_standby_photo()
    assert resolved == candidate


def test_resolve_standby_photo_none_when_empty(tmp_path):
    """Test when no standby photo exists in configured directory."""
    config = load_config()
    empty_dir = tmp_path / "empty_standby"
    empty_dir.mkdir()
    config.image_generation.standby_photo_path = str(empty_dir / "personal_photo.jpg")
    
    generator = QwenImageGenerator(config=config)
    resolved = generator.resolve_standby_photo()
    assert resolved is None


def test_encode_image_to_data_uri(tmp_path):
    """Test converting an image file into a base64 Data URI."""
    config = load_config()
    generator = QwenImageGenerator(config=config)
    
    test_img = tmp_path / "sample.png"
    _make_valid_image(test_img, 300, 300)
    
    data_uri = generator.encode_image_to_data_uri(test_img)
    assert data_uri.startswith("data:image/png;base64,")


def test_generate_image_dry_run(tmp_path):
    """Test dry-run simulation generates placeholder SVG and returns GeneratedImage."""
    config = load_config()
    output_dir = tmp_path / "gen_images"
    config.image_generation.output_dir = str(output_dir)
    
    generator = QwenImageGenerator(config=config)
    result = generator.generate_image(
        prompt="A software engineer working late on Python distributed backend architecture",
        dry_run=True,
        cycle_id="test_cycle_001"
    )
    
    assert isinstance(result, GeneratedImage)
    assert result.status == "simulated"
    assert "test_cycle_001" in result.image_url
    assert result.local_path is not None
    assert Path(result.local_path).exists()
    assert Path(result.local_path).suffix == ".svg"
    
    # Check that SVG contains prompt text
    with open(result.local_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    assert "software engineer" in svg_content


def test_generate_image_live_mocked_success(tmp_path):
    """Test successful live DashScope call with mocked HTTP responses."""
    config = load_config()
    output_dir = tmp_path / "gen_images"
    standby_dir = tmp_path / "standby"
    standby_dir.mkdir(parents=True)
    
    photo = standby_dir / "standby_test.jpg"
    _make_valid_image(photo)
    
    config.image_generation.output_dir = str(output_dir)
    config.image_generation.standby_photo_path = str(photo)
    config.image_generation.api_base_url = "https://dashscope.aliyuncs.com"
    config.image_generation.model = "qwen-image-3.0"
    
    generator = QwenImageGenerator(
        config=config,
        api_key="sk-test-mock-key-valid",
        base_url="https://dashscope.aliyuncs.com",
        model="qwen-image-3.0"
    )
    
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.json.return_value = {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"image": "https://dashscope-cdn.aliyuncs.com/qwen/output_test_123.png"}
                        ]
                    }
                }
            ]
        }
    }

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b"\x89PNG\r\n\x1a\nFakeImageData"

    with patch("requests.post", return_value=mock_post_resp) as mock_post, \
         patch("requests.get", return_value=mock_get_resp) as mock_get:
        
        result = generator.generate_image(
            prompt="Professional photo of an engineer in a glass office with monitor display",
            dry_run=False,
            cycle_id="cycle_live_test"
        )
        
        assert result.status == "success"
        assert result.image_url == "https://dashscope-cdn.aliyuncs.com/qwen/output_test_123.png"
        assert result.model == "qwen-image-3.0"
        assert result.standby_photo_used == str(photo)
        assert result.local_path is not None
        assert Path(result.local_path).exists()
        
        # Verify payload sent to DashScope
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["model"] == "qwen-image-3.0"
        assert payload["parameters"]["prompt_extend"] is True
        
        messages = payload["input"]["messages"]
        assert len(messages) == 1
        content_items = messages[0]["content"]
        # Standby photo should be first item, prompt second
        assert len(content_items) == 2
        assert "image" in content_items[0]
        assert content_items[0]["image"].startswith("data:image/jpeg;base64,")
        assert "text" in content_items[1]
        assert "Professional photo" in content_items[1]["text"]


def test_generate_image_api_error_handling(tmp_path):
    """Test that DashScope API non-200 responses raise RuntimeError."""
    config = load_config()
    generator = QwenImageGenerator(config=config, api_key="sk-test-key")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"code": "InvalidParameter", "message": "Model not available"}'
    
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError) as exc_info:
            generator.generate_image(prompt="Sample prompt", dry_run=False)
        assert "API Error [400]" in str(exc_info.value)


def test_generate_image_network_exception(tmp_path):
    """Test network connection error raises RuntimeError."""
    config = load_config()
    generator = QwenImageGenerator(config=config, api_key="sk-test-key")
    
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection timed out")):
        with pytest.raises(RuntimeError) as exc_info:
            generator.generate_image(prompt="Sample prompt", dry_run=False)
        assert "network failure" in str(exc_info.value).lower()


def test_prompt_builder_build_image_prompt():
    """Test PromptBuilder.build_image_prompt synthesizes appropriate instructions."""
    config = load_config()
    builder = PromptBuilder(config)
    
    post = "5 years into backend engineering, the best code is the code you never have to write."
    system_prompt, user_prompt = builder.build_image_prompt(
        post_content=post,
        topic="Minimalist backend systems",
        pillar_name="Backend Architecture",
        story_anchor="Refactoring a legacy service",
        user_mind="Keep code clean and maintainable"
    )
    
    assert "visual" in system_prompt.lower() or "qwen" in system_prompt.lower()
    assert "Minimalist backend systems" in user_prompt
    assert "Keep code clean" in user_prompt
    assert "5 years into backend engineering" in user_prompt
