"""
Buffer GraphQL API Client.
Official integration with Buffer GraphQL API (https://api.buffer.com).
Handles organization and channel (profile) discovery, draft creation, retries, and error handling.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests
import urllib3.util.connection as urllib3_cn

# Optimize DNS lookup on Linux environments by prioritizing IPv4
urllib3_cn.HAS_IPV6 = False

from src.config import BufferSettings
from src.logger import logger


@dataclass
class BufferProfile:
    id: str
    service: str
    service_username: str
    formatted_service: str
    display_name: str
    organization_id: str
    is_default: bool = False
    raw_data: Dict[str, Any] = None


class BufferClient:
    """Client for interacting with the official Buffer GraphQL API."""

    GRAPHQL_URL = "https://api.buffer.com"

    def __init__(self, settings: Optional[BufferSettings] = None, access_token: Optional[str] = None):
        self.settings = settings or BufferSettings()
        self.access_token = (access_token or self.settings.access_token or "").strip()
        self.endpoint = (self.settings.api_base_url or self.GRAPHQL_URL).rstrip("/")
        self.timeout = self.settings.timeout_seconds

    def _ensure_access_token(self) -> None:
        if not self.access_token:
            raise ValueError(
                "BUFFER_ACCESS_TOKEN is missing in your .env file.\n"
                "Please generate a Public API Token from your Buffer Account Settings: "
                "https://account.buffer.com/channels or Developer Portal."
            )

    def _post_graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper to post a GraphQL query/mutation to Buffer API."""
        self._ensure_access_token()
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code == 401 or response.status_code == 403:
            raise PermissionError(
                f"Buffer authentication failed [{response.status_code}]: "
                f"Invalid or expired BUFFER_ACCESS_TOKEN. Please verify your token in .env."
            )
        elif response.status_code != 200:
            raise RuntimeError(f"Buffer GraphQL HTTP Error [{response.status_code}]: {response.text}")

        resp_json = response.json()
        if "errors" in resp_json and resp_json["errors"]:
            error_msgs = "; ".join([e.get("message", str(e)) for e in resp_json["errors"]])
            raise RuntimeError(f"Buffer GraphQL Error: {error_msgs}")

        return resp_json.get("data", {})

    def get_profiles(self) -> List[BufferProfile]:
        """
        Fetches all connected channels/profiles across all organizations.
        """
        self._ensure_access_token()
        logger.info("[cyan]Fetching connected Buffer channels/profiles...[/cyan]")

        # 1. Fetch Account & Organizations
        orgs_query = """
        query GetOrganizations {
          account {
            id
            email
            organizations {
              id
              name
            }
          }
        }
        """
        data = self._post_graphql(orgs_query)
        account = data.get("account") or {}
        organizations = account.get("organizations") or []

        if not organizations:
            logger.warning("[yellow]No organizations found for this Buffer account.[/yellow]")
            return []

        all_profiles: List[BufferProfile] = []

        # 2. Query Channels for each organization
        channels_query = """
        query GetChannels($input: ChannelsInput!) {
          channels(input: $input) {
            id
            name
            displayName
            service
          }
        }
        """

        for org in organizations:
            org_id = org.get("id")
            if not org_id:
                continue

            channels_data = self._post_graphql(channels_query, variables={"input": {"organizationId": org_id}})
            channels = channels_data.get("channels") or []

            for ch in channels:
                service_name = (ch.get("service") or "").lower()
                profile = BufferProfile(
                    id=ch.get("id", ""),
                    service=service_name,
                    service_username=ch.get("name", ""),
                    formatted_service=service_name.capitalize(),
                    display_name=ch.get("displayName") or ch.get("name", ""),
                    organization_id=org_id,
                    raw_data=ch
                )
                all_profiles.append(profile)

        logger.info(f"[green]✓ Found {len(all_profiles)} connected Buffer channel(s).[/green]")
        return all_profiles

    def get_linkedin_profile_id(self, preferred_id: Optional[str] = None) -> str:
        """
        Resolves a LinkedIn channel ID. If preferred_id is provided, verifies it.
        Otherwise, auto-discovers the first available LinkedIn channel.
        """
        profiles = self.get_profiles()
        linkedin_profiles = [p for p in profiles if "linkedin" in p.service.lower()]

        if not linkedin_profiles:
            raise RuntimeError(
                "No LinkedIn channel found connected to your Buffer account.\n"
                "Please connect your LinkedIn account inside Buffer (https://publish.buffer.com)."
            )

        if preferred_id:
            for p in linkedin_profiles:
                if p.id == preferred_id:
                    return p.id
            logger.warning(
                f"[yellow]Preferred profile ID '{preferred_id}' not found in active LinkedIn channels. "
                f"Falling back to '{linkedin_profiles[0].display_name} ({linkedin_profiles[0].id})'.[/yellow]"
            )

        return linkedin_profiles[0].id

    def create_draft(
        self,
        text: str,
        profile_ids: Optional[List[str]] = None,
        image_url: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Creates a draft post in Buffer using the createPost mutation.
        Optionally attaches an image asset if image_url is provided.
        """
        if not text or not text.strip():
            raise ValueError("Cannot create empty draft in Buffer.")

        if dry_run:
            logger.info("[yellow][DRY-RUN] Simulating Buffer draft creation...[/yellow]")
            return {
                "success": True,
                "draft": True,
                "simulated": True,
                "message": "Draft creation simulated successfully",
                "profile_ids": profile_ids or ["dry_run_linkedin_channel"],
                "text_snippet": text[:120] + ("..." if len(text) > 120 else ""),
                "image_url": image_url,
                "has_image": bool(image_url),
                "id": "dry_run_draft_id_12345"
            }

        self._ensure_access_token()

        # Resolve target channel ID if not provided
        target_channel_id = (profile_ids[0] if profile_ids and profile_ids[0] else None) or self.settings.profile_id
        if not target_channel_id:
            target_channel_id = self.get_linkedin_profile_id()

        mutation = """
        mutation CreateDraftPost($input: CreatePostInput!) {
          createPost(input: $input) {
            ... on PostActionSuccess {
              post {
                id
                text
                status
              }
            }
            ... on MutationError {
              message
            }
          }
        }
        """

        input_payload: Dict[str, Any] = {
            "channelId": target_channel_id,
            "text": text,
            "schedulingType": "automatic",
            "mode": "addToQueue",
            "saveToDraft": True if self.settings.draft_mode else False
        }

        # Attach image asset if public URL is provided
        if image_url and (image_url.startswith("http://") or image_url.startswith("https://")):
            input_payload["assets"] = [
                {
                    "image": {
                        "url": image_url
                    }
                }
            ]
            logger.info(f"[cyan]Attaching image asset to Buffer draft: {image_url[:60]}...[/cyan]")

        variables = {
            "input": input_payload
        }


        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[cyan]Posting draft to Buffer GraphQL (Attempt {attempt}/{max_retries}, Channel: {target_channel_id})...[/cyan]")
                data = self._post_graphql(mutation, variables=variables)
                create_post_res = data.get("createPost") or {}

                # Check for union types in GraphQL response
                if "message" in create_post_res:
                    raise RuntimeError(f"Buffer GraphQL Mutation Error: {create_post_res['message']}")

                post_info = create_post_res.get("post") or {}
                post_id = post_info.get("id") or "unknown_id"
                post_status = post_info.get("status") or "draft"

                logger.info(f"[green]✓ Successfully created Buffer draft (ID: {post_id}, Status: {post_status})![/green]")
                return {
                    "success": True,
                    "id": post_id,
                    "status": post_status,
                    "post": post_info,
                    "channel_id": target_channel_id
                }

            except (requests.exceptions.RequestException, RuntimeError, PermissionError) as e:
                if isinstance(e, PermissionError):
                    # Do not retry on auth error
                    raise e

                last_error = e
                sleep_time = backoff_factor ** attempt
                logger.warning(
                    f"[yellow]Error during Buffer draft creation: {e}. "
                    f"Retrying in {sleep_time:.1f}s (Attempt {attempt}/{max_retries})...[/yellow]"
                )
                time.sleep(sleep_time)

        raise RuntimeError(f"Failed to create Buffer draft after {max_retries} attempts: {last_error}")
