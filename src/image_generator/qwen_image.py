"""
Qwen-Image-3.0 Image Generation Engine.
Integrates with Alibaba DashScope Multimodal Generation API.
Supports reference-based image generation using a standby personal photo
and context-driven LinkedIn visual prompts.
"""

import base64
import mimetypes
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from src.config import AppConfig, load_config
from src.logger import logger


@dataclass
class GeneratedImage:
    """Represents the output of the Qwen-Image-3.0 generation process."""
    prompt: str
    image_url: str
    local_path: Optional[str] = None
    model: str = "qwen-image-3.0"
    status: str = "success"  # "success" | "simulated" | "fallback"
    standby_photo_used: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_seconds: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class QwenImageGenerator:
    """
    Client for generating images using Alibaba Cloud Qwen-Image-3.0 via DashScope.
    Supports feeding a standby personal photo as a visual reference alongside
    context-aware prompts derived from LinkedIn posts.
    """

    DEFAULT_MODEL = "qwen-image-3.0"
    FALLBACK_MODEL = "qwen-image-3.0-pro"
    DEFAULT_BASE_URL = "https://token-plan.maas.qwencloudapi.com"
    API_ENDPOINT = "/api/v1/services/aigc/multimodal-generation/generation"

    SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.config = config or load_config()
        raw_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or self.config.get_api_key_for_provider("qwen")
            or ""
        ).strip()
        if raw_key.startswith("k-sp-"):
            raw_key = "s" + raw_key
        self.api_key = raw_key

        img_cfg = getattr(self.config, "image_generation", None)
        self.model = model or (img_cfg.model if img_cfg else self.DEFAULT_MODEL)
        
        raw_base_url = (
            base_url
            or os.getenv("QWEN_IMAGE_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or (img_cfg.api_base_url if img_cfg else self.DEFAULT_BASE_URL)
        ).rstrip("/")
        # If user passed OpenAI-compatible URL ending in /compatible-mode/v1, strip it for native multimodal endpoint
        if raw_base_url.endswith("/compatible-mode/v1"):
            self.base_url = raw_base_url[:-len("/compatible-mode/v1")].rstrip("/")
        else:
            self.base_url = raw_base_url

        self.timeout = timeout_seconds or (img_cfg.timeout_seconds if img_cfg else 60)
        self.output_dir = Path(img_cfg.output_dir if img_cfg else "data/generated_images")
        self.standby_configured_path = Path(img_cfg.standby_photo_path if img_cfg else "data/standby/personal_photo.jpg")

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.standby_configured_path.parent.mkdir(parents=True, exist_ok=True)

    def is_api_key_valid(self) -> bool:
        """Checks if a non-placeholder API key is configured."""
        if not self.api_key:
            return False
        if "your_dashscope" in self.api_key.lower() or "your_key" in self.api_key.lower():
            return False
        return True

    def _is_valid_image(self, path: Path) -> bool:
        """
        Validates that a file is an actual readable image with dimensions >= 240x240,
        as strictly required by Wan 2.7 and DashScope image reference APIs.
        """
        if not path or not path.exists() or not path.is_file():
            return False
        if path.stat().st_size < 500:  # Ignore corrupted stubs or empty files
            return False
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
                return w >= 240 and h >= 240
        except Exception:
            return False

    def resolve_standby_photo(self, custom_path: Optional[str] = None) -> Optional[Path]:
        """
        Locates and validates the standby personal photo to use as a reference.
        Checks:
        1. Explicit custom_path if provided.
        2. Configured standby photo path (e.g. data/standby/personal_photo.jpg).
        3. Project root directory (personal_photo.jpg, personal_photo.png, 1781602010575.png).
        4. Any supported image in data/standby/ directory.
        Automatically syncs valid images from project root into data/standby/.
        """
        import shutil

        # 1. Custom path
        if custom_path:
            p = Path(custom_path)
            if self._is_valid_image(p):
                logger.info(f"[cyan]Using specified standby photo:[/cyan] {p}")
                return p
            logger.warning(f"[yellow]Specified standby photo not found or invalid at: {custom_path}[/yellow]")

        # 2. Configured path (if valid)
        if self._is_valid_image(self.standby_configured_path):
            logger.info(f"[cyan]Using configured standby photo:[/cyan] {self.standby_configured_path}")
            return self.standby_configured_path

        # 3. Check project root directory for personal photo (only when using default standby path)
        is_default_path = str(self.standby_configured_path).replace("\\", "/").endswith("data/standby/personal_photo.jpg")
        if is_default_path:
            root_candidates = [
                Path("personal_photo.jpg"),
                Path("personal_photo.png"),
                Path("personal_photo.jpeg"),
                Path("1781602010575.png"),
            ]
            for rc in root_candidates:
                if self._is_valid_image(rc):
                    logger.info(f"[cyan]Found valid personal photo in project root:[/cyan] {rc}")
                    try:
                        self.standby_configured_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(rc, self.standby_configured_path)
                        logger.info(f"[green]✓ Synchronized standby photo to:[/green] {self.standby_configured_path}")
                        return self.standby_configured_path
                    except Exception as sync_err:
                        logger.warning(f"[yellow]Could not copy root photo to standby dir: {sync_err}[/yellow]")
                        return rc

        # 4. Search directory containing configured path for any valid image
        search_dir = self.standby_configured_path.parent
        if search_dir.exists():
            for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                for candidate in search_dir.glob(f"*{ext}"):
                    if self._is_valid_image(candidate):
                        logger.info(f"[cyan]Auto-discovered valid standby photo in {search_dir}:[/cyan] {candidate.name}")
                        return candidate

        logger.warning(
            f"[yellow]No valid standby photo found (>= 240x240 px). "
            f"Place your photo at '{self.standby_configured_path}' or 'personal_photo.jpg' to enable personalized reference-based generation.[/yellow]"
        )
        return None

    def encode_image_to_data_uri(self, image_path: Path) -> str:
        """
        Reads and optimizes an image file, converting it into a base64 Data URI string.
        Ensures proper RGB color mode and prevents payload bloat.
        """
        import io
        from PIL import Image

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/jpeg"

        try:
            with Image.open(image_path) as im:
                # Convert palettes or CMYK to RGB
                if im.mode in ("RGBA", "P", "CMYK"):
                    if "png" in mime_type.lower():
                        im = im.convert("RGBA")
                    else:
                        im = im.convert("RGB")
                        mime_type = "image/jpeg"
                
                # Downscale if massive (> 2048px) to speed up upload while keeping high quality
                max_dim = 2048
                if max(im.size) > max_dim:
                    im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                save_fmt = "PNG" if "png" in mime_type.lower() else "JPEG"
                if save_fmt == "JPEG":
                    im.save(buf, format=save_fmt, quality=92, optimize=True)
                else:
                    im.save(buf, format=save_fmt, optimize=True)
                
                encoded_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_bytes}"
        except Exception as e:
            logger.warning(f"[yellow]Pillow processing skipped for {image_path}: {e}. Reading raw bytes.[/yellow]")
            with open(image_path, "rb") as f:
                encoded_bytes = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_bytes}"

    def _generate_placeholder_image(self, prompt: str, cycle_id: Optional[str] = None) -> Path:
        """
        Creates a clean mock SVG/PNG image on disk for simulations or dry runs
        so that local previews and testing work seamlessly without network calls.
        """
        cid = cycle_id or str(int(time.time()))
        output_file = self.output_dir / f"simulated_{cid}.svg"
        
        safe_prompt = (prompt[:140] + "...") if len(prompt) > 140 else prompt
        # Escape XML entities
        safe_prompt = safe_prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', '&quot;')

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="50%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#090d16"/>
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#818cf8"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <circle cx="512" cy="380" r="160" fill="#334155" stroke="url(#accent)" stroke-width="4"/>
  <!-- User Silhouette -->
  <circle cx="512" cy="330" r="60" fill="#94a3b8"/>
  <path d="M432 460 C432 400, 592 400, 592 460 Z" fill="#94a3b8"/>
  <!-- Badge -->
  <rect x="362" y="580" width="300" height="40" rx="20" fill="url(#accent)"/>
  <text x="512" y="606" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">
    QWEN-IMAGE-3.0 • STANDBY REFERENCE
  </text>
  <!-- Prompt Card -->
  <rect x="112" y="650" width="800" height="240" rx="16" fill="#1e293b" stroke="#334155" stroke-width="2"/>
  <text x="142" y="695" font-family="system-ui, -apple-system, sans-serif" font-size="18" font-weight="bold" fill="#38bdf8">
    Prompt &amp; Story Anchor Context:
  </text>
  <foreignObject x="142" y="715" width="740" height="150">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: system-ui, sans-serif; font-size: 15px; color: #cbd5e1; line-height: 1.5;">
      {safe_prompt}
    </div>
  </foreignObject>
  <text x="512" y="940" font-family="system-ui, sans-serif" font-size="13" fill="#64748b" text-anchor="middle">
    Automated LinkedIn Visual Generation • Cycle #{cid}
  </text>
</svg>"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return output_file

    def generate_image(
        self,
        prompt: str,
        standby_photo_path: Optional[str] = None,
        dry_run: bool = False,
        cycle_id: Optional[str] = None,
    ) -> GeneratedImage:
        """
        Generates an image via Qwen-Image-3.0 using the prompt and standby reference photo.
        In dry-run mode or if DASHSCOPE_API_KEY is not set, cleanly simulates generation.
        """
        resolved_photo = self.resolve_standby_photo(standby_photo_path)
        photo_name = resolved_photo.name if resolved_photo else None

        # Check simulation condition
        if dry_run or not self.is_api_key_valid():
            mode_note = "DRY-RUN mode" if dry_run else "DASHSCOPE_API_KEY not configured"
            logger.info(f"[yellow]Simulating Qwen-Image-3.0 generation ({mode_note})...[/yellow]")
            local_mock_path = self._generate_placeholder_image(prompt, cycle_id=cycle_id)
            
            simulated_url = f"https://dashscope-simulated.aliyuncs.com/qwen-image-3.0/draft_{cycle_id or 'mock'}.png"
            return GeneratedImage(
                prompt=prompt,
                image_url=simulated_url,
                local_path=str(local_mock_path),
                model=self.model,
                status="simulated",
                standby_photo_used=str(resolved_photo) if resolved_photo else None,
                latency_seconds=0.1,
                raw_response={"simulated": True, "reason": mode_note}
            )

        # Live Qwen-Image-3.0 API Call
        endpoint = f"{self.base_url}{self.API_ENDPOINT}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build message contents: if standby photo exists, include it first
        user_content: List[Dict[str, Any]] = []
        if resolved_photo:
            try:
                data_uri = self.encode_image_to_data_uri(resolved_photo)
                user_content.append({"image": data_uri})
                logger.info(f"[cyan]Attaching standby photo '{resolved_photo.name}' as visual reference...[/cyan]")
            except Exception as e:
                logger.warning(f"[yellow]Failed to encode standby photo: {e}. Proceeding with text-to-image.[/yellow]")

        user_content.append({"text": prompt})

        # Determine candidate models and endpoints to try
        is_token_plan = "token-plan" in self.base_url.lower()
        if is_token_plan:
            # On Token Plan, Wan 2.7 is the active production model family
            candidate_models = ["wan2.7-image", "wan2.7-image-pro"]
            if self.model and self.model not in candidate_models and self.model != "qwen-image-3.0":
                candidate_models.insert(0, self.model)
        else:
            candidate_models = [self.model]
            if self.FALLBACK_MODEL and self.FALLBACK_MODEL not in candidate_models:
                candidate_models.append(self.FALLBACK_MODEL)
            if "wan2.7-image" not in candidate_models:
                candidate_models.append("wan2.7-image")

        payload = {
            "model": candidate_models[0],
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            },
            "parameters": {
                "prompt_extend": True
            }
        }

        logger.info(f"[cyan]• Image Prompt:[/cyan] {prompt[:100]}...")

        resp = None
        resp_json = {}
        actual_model_used = candidate_models[0]
        latency = 0.0

        for current_model in candidate_models:
            payload["model"] = current_model
            actual_model_used = current_model
            start_time = time.time()

            # Select multimodal generation endpoint for Wan 2.7 or Token Plan
            if "wan2" in current_model.lower() or is_token_plan:
                endpoint = f"{self.base_url}/api/v1/services/aigc/multimodal-generation/generation"
            else:
                endpoint = f"{self.base_url}{self.API_ENDPOINT}"

            logger.info(f"[bold cyan]Calling Qwen Image Generator ({current_model}) on {self.base_url}...[/bold cyan]")

            try:
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                latency = time.time() - start_time

                # If model does not exist on this endpoint, try next candidate
                if resp.status_code == 404 and ("Model not exist" in resp.text or "not found" in resp.text.lower()):
                    logger.warning(
                        f"[yellow]Model '{current_model}' not found on {self.base_url}. Trying next candidate...[/yellow]"
                    )
                    continue

                # If reference photo is below 240x240 minimum resolution or unreadable, retry with text-only
                if resp.status_code == 400 and (
                    "resolution must be at least" in resp.text
                    or "Error validating image" in resp.text
                    or "can not read image" in resp.text
                ):
                    logger.warning(
                        "[yellow]Standby photo rejected by reference API. Retrying with text-only generation...[/yellow]"
                    )
                    payload["input"]["messages"][0]["content"] = [{"text": prompt}]
                    resp = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    latency = time.time() - start_time

                if resp.status_code == 200:
                    resp_json = resp.json()
                    break

                err_text = resp.text
                logger.error(f"[red]Qwen Image API Error [{resp.status_code}]: {err_text}[/red]")
                raise RuntimeError(f"Qwen-Image API Error [{resp.status_code}]: {err_text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"[red]Network error calling Qwen-Image ({current_model}): {e}[/red]")
                raise RuntimeError(f"Qwen-Image network failure: {e}") from e

        if not resp or resp.status_code != 200:
            raise RuntimeError(f"All image generation candidate models exhausted on {self.base_url}.")

        output = resp_json.get("output", {})
        choices = output.get("choices", [])
        image_url = None

        if choices:
            message_content = choices[0].get("message", {}).get("content", [])
            if isinstance(message_content, list):
                for item in message_content:
                    if isinstance(item, dict) and "image" in item:
                        image_url = item["image"]
                        break
            elif isinstance(message_content, dict) and "image" in message_content:
                image_url = message_content["image"]

        if not image_url and "results" in output:
            results = output.get("results", [])
            if results and isinstance(results[0], dict):
                image_url = results[0].get("url")

        if not image_url:
            raise RuntimeError(f"Image URL not found in Qwen-Image response: {resp_json}")

        logger.info(f"[green]✓ Image generated in {latency:.2f}s using {actual_model_used}! URL: {image_url[:60]}...[/green]")

        # Download and cache locally
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        cid = cycle_id or timestamp_str
        local_filename = f"qwen_image_{cid}.png"
        local_file_path = self.output_dir / local_filename

        try:
            img_res = requests.get(image_url, timeout=30)
            if img_res.status_code == 200:
                with open(local_file_path, "wb") as f:
                    f.write(img_res.content)
                logger.info(f"[green]✓ Saved generated image locally to: {local_file_path}[/green]")
        except Exception as dl_err:
            logger.warning(f"[yellow]Could not download image locally: {dl_err}. URL remains valid.[/yellow]")
            local_file_path = None

        return GeneratedImage(
            prompt=prompt,
            image_url=image_url,
            local_path=str(local_file_path) if local_file_path and local_file_path.exists() else None,
            model=actual_model_used,
            status="success",
            standby_photo_used=str(resolved_photo) if resolved_photo else None,
            latency_seconds=latency,
            raw_response=resp_json
        )
