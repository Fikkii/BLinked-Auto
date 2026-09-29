"""
Cloudinary Image Uploader Service.
Uploads generated visual assets to Cloudinary to provide clean, permanent,
CDN-backed HTTPS URLs for Buffer drafts and social publishing.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cloudinary
import cloudinary.uploader
import cloudinary.api

from src.config import AppConfig, load_config
from src.logger import logger


class CloudinaryUploader:
    """
    Handles uploading generated visual assets to Cloudinary.
    Provides permanent, high-performance CDN URLs for Buffer GraphQL integration.
    """

    DEFAULT_FOLDER = "linkedin_automation"

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        c_cfg = getattr(self.config, "cloudinary", None)

        self.url = (os.getenv("CLOUDINARY_URL") or (c_cfg.url if c_cfg else "")).strip()
        self.cloud_name = (os.getenv("CLOUDINARY_CLOUD_NAME") or (c_cfg.cloud_name if c_cfg else "")).strip()
        self.api_key = (os.getenv("CLOUDINARY_API_KEY") or (c_cfg.api_key if c_cfg else "")).strip()
        self.api_secret = (os.getenv("CLOUDINARY_API_SECRET") or (c_cfg.api_secret if c_cfg else "")).strip()
        self.folder = (c_cfg.folder if c_cfg and c_cfg.folder else self.DEFAULT_FOLDER)
        self.enabled = (c_cfg.enabled if c_cfg is not None else True)

        # Parse CLOUDINARY_URL if provided
        if self.url and not self.cloud_name:
            try:
                # cloudinary://<api_key>:<api_secret>@<cloud_name>
                if "@" in self.url:
                    parts = self.url.split("@")
                    self.cloud_name = parts[-1].split("?")[0].strip()
            except Exception:
                pass

        self._configure()

    def _configure(self) -> None:
        """Initializes Cloudinary SDK configuration."""
        if self.url:
            cloudinary.config(cloudinary_url=self.url)
        elif self.cloud_name and self.api_key and self.api_secret:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True
            )

    def is_configured(self) -> bool:
        """Checks if valid Cloudinary credentials have been provided."""
        if not self.enabled:
            return False

        if self.url:
            return not ("your_cloudinary" in self.url.lower() or "your_key" in self.url.lower())

        if self.cloud_name and self.api_key and self.api_secret:
            has_placeholder = any(
                "your_" in val.lower()
                for val in (self.cloud_name, self.api_key, self.api_secret)
            )
            return not has_placeholder

        return False

    def validate_credentials(self) -> bool:
        """Validates Cloudinary credentials with a lightweight ping test."""
        if not self.is_configured():
            return False
        try:
            res = cloudinary.api.ping()
            return res.get("status") == "ok"
        except Exception as e:
            logger.warning(f"Cloudinary validation ping failed: {e}")
            return False

    def upload_image(
        self,
        image_source: Union[str, Path],
        cycle_id: Optional[str] = None,
        public_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Uploads an image (local file path or remote HTTP URL) to Cloudinary.
        Returns a dictionary containing the permanent `secure_url` and asset metadata.
        """
        source_str = str(image_source)
        cid = cycle_id or "latest"
        target_public_id = public_id or f"linkedin_post_{cid}"

        if dry_run or not self.is_configured():
            mode = "DRY-RUN mode" if dry_run else "Cloudinary not configured"
            simulated_url = (
                source_str
                if source_str.startswith("http")
                else f"https://res.cloudinary.com/{self.cloud_name or 'buffer-studio'}/image/upload/{self.folder}/{target_public_id}.png"
            )
            logger.info(f"[yellow]Cloudinary upload simulated ({mode}). CDN URL: {simulated_url[:60]}...[/yellow]")
            return {
                "secure_url": simulated_url,
                "public_id": f"{self.folder}/{target_public_id}",
                "status": "simulated",
                "cloud_name": self.cloud_name or "simulated",
                "reason": mode,
            }

        logger.info(f"[bold cyan]Uploading visual asset to Cloudinary (Folder: '{self.folder}')...[/bold cyan]")
        try:
            upload_result = cloudinary.uploader.upload(
                source_str,
                folder=self.folder,
                public_id=target_public_id,
                overwrite=True,
                resource_type="image",
                use_filename=True,
                unique_filename=False,
            )

            secure_url = upload_result.get("secure_url") or upload_result.get("url")
            pub_id = upload_result.get("public_id")
            fmt = upload_result.get("format")
            bytes_size = upload_result.get("bytes", 0)

            logger.info(
                f"[green]✓ Uploaded to Cloudinary successfully! "
                f"URL: {secure_url} ({bytes_size} bytes, format: {fmt})[/green]"
            )

            return {
                "secure_url": secure_url,
                "public_id": pub_id,
                "format": fmt,
                "bytes": bytes_size,
                "width": upload_result.get("width"),
                "height": upload_result.get("height"),
                "created_at": upload_result.get("created_at"),
                "cloud_name": self.cloud_name,
                "status": "uploaded",
                "raw_response": upload_result,
            }

        except Exception as e:
            logger.error(f"[red]Cloudinary upload failed: {e}[/red]")
            # Fall back gracefully to original source if remote URL, else report failure without crashing
            if source_str.startswith("http"):
                logger.warning("[yellow]Falling back to original remote URL for Buffer draft.[/yellow]")
                return {
                    "secure_url": source_str,
                    "public_id": None,
                    "status": "fallback",
                    "error": str(e),
                }
            return {
                "secure_url": None,
                "public_id": None,
                "status": "failed",
                "error": str(e),
            }
