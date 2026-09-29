"""
Unit tests for Dashboard Web API endpoints.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from src.web.app import app
from src.providers.base import LLMResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    """Test that dashboard home page renders HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Buffer LinkedIn Studio" in response.text
    assert "Creative Studio" in response.text
    assert "Client Hub" in response.text
    assert "Admin Center" in response.text


def test_get_config(client):
    """Test /api/config returns configuration details."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "persona" in data
    assert "prompts" in data
    assert "llm" in data
    assert "buffer" in data


def test_get_status(client):
    """Test /api/status endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "has_pending" in data


def test_generate_api(client):
    """Test /api/generate endpoint with mocked LLM."""
    mock_llm_resp = LLMResponse(
        content="Generated via Web API #Fullstack #Python",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=120
    )

    with patch("src.providers.gemini.GeminiProvider.generate", return_value=mock_llm_resp):
        response = client.post("/api/generate", json={
            "user_mind": "Web test thought",
            "provider": "gemini",
            "generate_image": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Generated via Web API" in data["content"]
        assert data["cycle_id"] is not None


def test_publish_draft_api_dry_run(client):
    """Test /api/publish-draft in dry-run mode."""
    response = client.post("/api/publish-draft", json={
        "text": "Dry run post from web dashboard #WebDev",
        "dry_run": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dry_run"] is True


def test_standby_photo_endpoints(client):
    """Test GET and POST /api/standby-photo."""
    # Test GET
    get_res = client.get("/api/standby-photo")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert "exists" in get_data

    # Test POST upload
    dummy_jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    upload_res = client.post(
        "/api/standby-photo",
        files={"file": ("personal_photo.jpg", dummy_jpg, "image/jpeg")}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["success"] is True
    assert "preview_url" in upload_data


def test_generate_image_endpoint(client):
    """Test POST /api/generate-image endpoint."""
    from src.image_generator.qwen_image import GeneratedImage
    mock_gen_img = GeneratedImage(
        prompt="Software engineer coding on laptop with coffee",
        image_url="https://mock.cdn/img.png",
        local_path="data/generated_images/simulated_web_test_123.svg",
        model="wan2.7-image",
        status="simulated"
    )
    with patch("src.image_generator.qwen_image.QwenImageGenerator.generate_image", return_value=mock_gen_img):
        res = client.post("/api/generate-image", json={
            "prompt": "Software engineer coding on laptop with coffee",
            "cycle_id": "web_test_123"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "image_data" in data
        assert data["image_data"]["prompt"] == "Software engineer coding on laptop with coffee"
        assert "preview_url" in data["image_data"]


def test_cloudinary_status_endpoint(client):
    """Test GET /api/cloudinary endpoint."""
    res = client.get("/api/cloudinary")
    assert res.status_code == 200
    data = res.json()
    assert "configured" in data
    assert "folder" in data
    assert "message" in data

