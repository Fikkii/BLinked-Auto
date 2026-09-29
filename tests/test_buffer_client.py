"""
Unit tests for Buffer GraphQL API Client.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.config import BufferSettings
from src.buffer.client import BufferClient, BufferProfile


def test_get_profiles_graphql():
    settings = BufferSettings(access_token="valid_token")
    client = BufferClient(settings=settings)

    # 1. First call returns account organizations
    orgs_resp = MagicMock()
    orgs_resp.status_code = 200
    orgs_resp.json.return_value = {
        "data": {
            "account": {
                "id": "acc_123",
                "email": "test@example.com",
                "organizations": [{"id": "org_456", "name": "Test Org"}]
            }
        }
    }

    # 2. Second call returns channels for org
    channels_resp = MagicMock()
    channels_resp.status_code = 200
    channels_resp.json.return_value = {
        "data": {
            "channels": [
                {
                    "id": "channel_li_1",
                    "name": "fikkii",
                    "displayName": "Oluwafikayo Ajala",
                    "service": "linkedin"
                },
                {
                    "id": "channel_tw_2",
                    "name": "fikki_aj",
                    "displayName": "fikki_aj",
                    "service": "twitter"
                }
            ]
        }
    }

    with patch("requests.post", side_effect=[orgs_resp, channels_resp, orgs_resp, channels_resp]):
        profiles = client.get_profiles()
        assert len(profiles) == 2
        assert profiles[0].id == "channel_li_1"
        assert profiles[0].service == "linkedin"
        assert profiles[0].display_name == "Oluwafikayo Ajala"
        
        li_id = client.get_linkedin_profile_id()
        assert li_id == "channel_li_1"


def test_create_draft_dry_run():
    settings = BufferSettings(access_token="valid_token")
    client = BufferClient(settings=settings)

    res = client.create_draft(text="Draft post test", profile_ids=["channel_li_1"], dry_run=True)
    assert res["success"] is True
    assert res["draft"] is True
    assert res["simulated"] is True


def test_create_draft_live_graphql():
    settings = BufferSettings(access_token="valid_token", draft_mode=True)
    client = BufferClient(settings=settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "createPost": {
                "post": {
                    "id": "post_789",
                    "text": "Draft content",
                    "status": "draft"
                }
            }
        }
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client.create_draft(text="Draft content", profile_ids=["channel_li_1"])
        assert res["success"] is True
        assert res["id"] == "post_789"
        assert res["status"] == "draft"

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer valid_token"
        assert call_kwargs["json"]["variables"]["input"]["saveToDraft"] is True
        assert call_kwargs["json"]["variables"]["input"]["channelId"] == "channel_li_1"
