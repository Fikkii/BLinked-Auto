"""
FastAPI Backend Application for the LinkedIn & Buffer Automation Dashboard.
Provides REST APIs for Creative Studio, Client Review Hub, and Admin Control Center.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import load_config, AppConfig
from src.state_manager import StateManager
from src.providers.factory import LLMProviderFactory
from src.generator.content_generator import ContentGenerator, GeneratedPost
from src.buffer.client import BufferClient
from src.image_generator.qwen_image import QwenImageGenerator, GeneratedImage
from src.image_generator.cloudinary_uploader import CloudinaryUploader
from src.logger import logger

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

app = FastAPI(
    title="Buffer LinkedIn Content Automation Studio",
    description="Enterprise Multi-Role Dashboard for Creatives, Clients, and Admins",
    version="2.0.0"
)

# Mount static assets and generated media
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if DATA_DIR.exists():
    app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    user_mind: Optional[str] = None
    topic: Optional[str] = None
    pillar_id: Optional[str] = None
    hook_style: Optional[str] = None
    story_anchor: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    generate_image: bool = True
    image_prompt: Optional[str] = None
    standby_photo: Optional[str] = None


class GenerateImageRequest(BaseModel):
    prompt: Optional[str] = None
    post_content: Optional[str] = None
    topic: Optional[str] = None
    pillar_name: Optional[str] = None
    story_anchor: Optional[str] = None
    user_mind: Optional[str] = None
    standby_photo: Optional[str] = None
    cycle_id: Optional[str] = None
    dry_run: bool = False


class PublishDraftRequest(BaseModel):
    text: str
    channel_id: Optional[str] = None
    image_url: Optional[str] = None
    dry_run: bool = False
    metadata: Optional[Dict[str, Any]] = None


class SaveCheckpointRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    image_data: Optional[Dict[str, Any]] = None


class TestProviderRequest(BaseModel):
    provider: str



# ------------------------------------------------------------------------------
# Page Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renders the single-page dashboard."""
    config = load_config()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "persona": config.persona,
            "active_provider": config.llm.active_provider,
            "version": config.app.version
        }
    )


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/config")
async def get_configuration():
    """Returns app configuration, persona data, content pillars, hook styles, and image settings."""
    config = load_config()
    img_gen = QwenImageGenerator(config=config)
    standby_photo = img_gen.resolve_standby_photo()

    return {
        "persona": config.persona,
        "prompts": config.prompts,
        "llm": {
            "active_provider": config.llm.active_provider,
            "model": config.llm.model,
            "providers": {k: v.model_dump() for k, v in config.llm.providers.items()}
        },
        "image_generation": {
            "enabled": config.image_generation.enabled,
            "model": config.image_generation.model,
            "standby_photo_path": config.image_generation.standby_photo_path,
            "has_standby_photo": bool(standby_photo and standby_photo.exists()),
            "standby_photo_name": standby_photo.name if standby_photo else None,
            "standby_preview_url": f"/media/standby/{standby_photo.name}" if standby_photo else None,
            "has_api_key": img_gen.is_api_key_valid()
        },
        "buffer": {
            "api_base_url": config.buffer.api_base_url,
            "profile_id": config.buffer.profile_id,
            "has_token": bool(config.buffer.access_token)
        },
        "cloudinary": {
            "configured": CloudinaryUploader(config=config).is_configured(),
            "cloud_name": CloudinaryUploader(config=config).cloud_name or None,
            "folder": CloudinaryUploader(config=config).folder
        }
    }


@app.get("/api/cloudinary")
async def get_cloudinary_status():
    """Returns current Cloudinary configuration and connectivity status."""
    config = load_config()
    uploader = CloudinaryUploader(config=config)
    is_conf = uploader.is_configured()
    return {
        "configured": is_conf,
        "cloud_name": uploader.cloud_name if is_conf else None,
        "folder": uploader.folder,
        "message": "Cloudinary CDN active and ready" if is_conf else "Cloudinary not configured. Set CLOUDINARY_URL or CLOUDINARY_CLOUD_NAME in .env"
    }


@app.get("/api/standby-photo")
async def get_standby_photo_status():
    """Returns the current standby personal photo status and preview URL."""
    config = load_config()
    img_gen = QwenImageGenerator(config=config)
    photo = img_gen.resolve_standby_photo()

    if photo and photo.exists():
        return {
            "exists": True,
            "filename": photo.name,
            "size_bytes": photo.stat().st_size,
            "preview_url": f"/media/standby/{photo.name}",
            "full_path": str(photo)
        }
    return {
        "exists": False,
        "message": "No standby personal photo found. Upload a photo or place one in data/standby/personal_photo.jpg"
    }


@app.post("/api/standby-photo")
async def upload_standby_photo(file: UploadFile = File(...)):
    """Uploads a personal standby reference photo for Qwen-Image-3.0."""
    try:
        standby_dir = DATA_DIR / "standby"
        standby_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as personal_photo.jpg or preserve extension
        ext = Path(file.filename or "photo.jpg").suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        target_path = standby_dir / f"personal_photo{ext}"

        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"[green]✓ Uploaded new standby photo to: {target_path}[/green]")
        return {
            "success": True,
            "filename": target_path.name,
            "preview_url": f"/media/standby/{target_path.name}",
            "message": "Personal standby photo uploaded successfully!"
        }
    except Exception as e:
        logger.error(f"Error uploading standby photo: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/status")
async def get_pipeline_status():
    """Returns current state checkpoint and recent history log."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    state = state_mgr.get_state()
    history = state_mgr.get_recent_history(limit=10)
    return {
        "state": state,
        "has_pending": state_mgr.has_pending_generation(),
        "recent_history": history
    }


@app.get("/api/profiles")
async def get_buffer_profiles():
    """Fetches connected Buffer channels / profiles."""
    config = load_config()
    client = BufferClient(settings=config.buffer)
    try:
        profiles = client.get_profiles()
        return {
            "success": True,
            "profiles": [
                {
                    "id": p.id,
                    "service": p.service,
                    "formatted_service": p.formatted_service,
                    "display_name": p.display_name,
                    "username": p.service_username,
                    "is_linkedin": "linkedin" in p.service.lower()
                }
                for p in profiles
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching Buffer profiles: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/generate")
async def generate_post(req: GenerateRequest):
    """Generates a new LinkedIn post with the selected LLM provider and optional Qwen-Image-3.0 visual."""
    config = load_config()
    provider_name = req.provider or config.llm.active_provider
    
    try:
        provider = LLMProviderFactory.create(
            provider_name=provider_name,
            config=config,
            model_name=req.model
        )
        generator = ContentGenerator(provider=provider, config=config)

        generated: GeneratedPost = generator.generate_post(
            topic=req.topic,
            pillar_id=req.pillar_id,
            hook_style=req.hook_style,
            story_anchor=req.story_anchor,
            user_mind=req.user_mind
        )

        state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)

        image_data = None
        if req.generate_image:
            img_gen = QwenImageGenerator(config=config)
            img_prompt = req.image_prompt or generator.generate_image_prompt(
                post_content=generated.content,
                topic=req.topic,
                pillar_name=generated.metadata.get("pillar_name"),
                story_anchor=req.story_anchor,
                user_mind=req.user_mind
            )
            gen_img: GeneratedImage = img_gen.generate_image(
                prompt=img_prompt,
                standby_photo_path=req.standby_photo,
                dry_run=config.app.dry_run
            )
            
            # Format preview url for web
            local_preview_url = None
            if gen_img.local_path:
                local_preview_url = f"/media/generated_images/{Path(gen_img.local_path).name}"

            image_data = {
                "prompt": gen_img.prompt,
                "image_url": gen_img.image_url,
                "local_path": gen_img.local_path,
                "preview_url": local_preview_url or gen_img.image_url,
                "model": gen_img.model,
                "status": gen_img.status,
                "standby_photo_used": gen_img.standby_photo_used,
                "generated_at": gen_img.generated_at
            }

            # Upload to Cloudinary if configured
            c_uploader = CloudinaryUploader(config=config)
            if c_uploader.is_configured():
                c_upload = c_uploader.upload_image(
                    image_source=gen_img.local_path or gen_img.image_url,
                    dry_run=config.app.dry_run
                )
                if c_upload.get("secure_url"):
                    image_data["cloudinary_url"] = c_upload["secure_url"]
                    image_data["image_url"] = c_upload["secure_url"]
                    image_data["preview_url"] = c_upload["secure_url"]
                    image_data["cloudinary_public_id"] = c_upload.get("public_id")

        cycle_id = state_mgr.save_generation(
            post_content=generated.content,
            metadata=generated.metadata,
            image_data=image_data
        )

        return {
            "success": True,
            "cycle_id": cycle_id,
            "content": generated.content,
            "metadata": generated.metadata,
            "image_data": image_data,
            "character_count": len(generated.content),
            "estimated_reading_time_mins": max(1, round(len(generated.content.split()) / 200, 1))
        }
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/generate-image")
async def generate_image_endpoint(req: GenerateImageRequest):
    """Generates or regenerates an image with Qwen-Image-3.0 using the prompt & standby photo."""
    config = load_config()
    try:
        img_gen = QwenImageGenerator(config=config)

        active_prompt = req.prompt
        if not active_prompt and req.post_content:
            provider = LLMProviderFactory.create(
                provider_name=config.llm.active_provider,
                config=config
            )
            generator = ContentGenerator(provider=provider, config=config)
            active_prompt = generator.generate_image_prompt(
                post_content=req.post_content,
                topic=req.topic,
                pillar_name=req.pillar_name,
                story_anchor=req.story_anchor,
                user_mind=req.user_mind
            )

        if not active_prompt:
            active_prompt = (
                "Featuring the person from the reference photo, depict them naturally at a modern developer workstation "
                "with multiple monitors displaying clean Python and JavaScript code, warm atmospheric lighting, focused expression. "
                "Photorealistic 35mm lens photography, natural office depth of field."
            )

        gen_img: GeneratedImage = img_gen.generate_image(
            prompt=active_prompt,
            standby_photo_path=req.standby_photo,
            dry_run=req.dry_run or config.app.dry_run,
            cycle_id=req.cycle_id
        )

        local_preview_url = None
        if gen_img.local_path:
            local_preview_url = f"/media/generated_images/{Path(gen_img.local_path).name}"

        image_data = {
            "prompt": gen_img.prompt,
            "image_url": gen_img.image_url,
            "local_path": gen_img.local_path,
            "preview_url": local_preview_url or gen_img.image_url,
            "model": gen_img.model,
            "status": gen_img.status,
            "standby_photo_used": gen_img.standby_photo_used,
            "generated_at": gen_img.generated_at
        }

        # Upload to Cloudinary if configured
        try:
            c_uploader = CloudinaryUploader(config=config)
            if c_uploader.is_configured():
                c_upload = c_uploader.upload_image(
                    image_source=gen_img.local_path or gen_img.image_url,
                    cycle_id=req.cycle_id,
                    dry_run=req.dry_run or config.app.dry_run
                )
                if c_upload.get("secure_url"):
                    image_data["cloudinary_url"] = c_upload["secure_url"]
                    image_data["image_url"] = c_upload["secure_url"]
                    image_data["preview_url"] = c_upload["secure_url"]
                    image_data["cloudinary_public_id"] = c_upload.get("public_id")
        except Exception as c_err:
            logger.warning(f"[yellow]Cloudinary upload skipped or failed in web studio: {c_err}[/yellow]")

        # If a checkpoint cycle exists, update it with the new image
        if req.cycle_id and req.post_content:
            state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
            state_mgr.save_generation(
                post_content=req.post_content,
                metadata={"topic": req.topic, "provider": config.llm.active_provider},
                image_data=image_data,
                cycle_id=req.cycle_id
            )

        return {
            "success": True,
            "image_data": image_data
        }
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/save-checkpoint")
async def save_checkpoint(req: SaveCheckpointRequest):
    """Updates the state checkpoint with edited content and image data without publishing."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    cycle_id = state_mgr.save_generation(
        post_content=req.content,
        metadata=req.metadata or {},
        image_data=req.image_data
    )
    return {"success": True, "cycle_id": cycle_id, "message": "Saved to checkpoint"}


@app.post("/api/publish-draft")
async def publish_draft(req: PublishDraftRequest):
    """Publishes the post as a draft to the Buffer GraphQL API, attaching image asset if provided."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    client = BufferClient(settings=config.buffer)

    target_channel_id = req.channel_id or config.buffer.profile_id
    if not target_channel_id and not req.dry_run:
        try:
            target_channel_id = client.get_linkedin_profile_id()
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "error": f"LinkedIn channel resolution failed: {e}"})

    try:
        buffer_resp = client.create_draft(
            text=req.text,
            profile_ids=[target_channel_id] if target_channel_id else None,
            image_url=req.image_url,
            max_retries=config.pipeline.max_retries,
            backoff_factor=config.pipeline.retry_backoff_seconds,
            dry_run=req.dry_run
        )
        # Mark completed and archive
        state_mgr.mark_completed(buffer_resp, profile_id=target_channel_id)
        return {
            "success": True,
            "dry_run": req.dry_run,
            "channel_id": target_channel_id,
            "image_url": req.image_url,
            "buffer_response": buffer_resp
        }
    except Exception as e:
        state_mgr.record_failure(str(e), stage="buffer_draft")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/resume")

async def resume_pipeline(req: Dict[str, Any] = {}):
    """Resumes drafting the pending post in checkpoint to save API credits."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    pending = state_mgr.get_pending_generation()

    if not pending:
        return JSONResponse(status_code=400, content={"success": False, "error": "No pending draft in checkpoint"})

    client = BufferClient(settings=config.buffer)
    target_channel_id = config.buffer.profile_id
    if not target_channel_id:
        try:
            target_channel_id = client.get_linkedin_profile_id()
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

    image_data = pending.get("image_data") or {}
    image_url = image_data.get("image_url")

    try:
        buffer_resp = client.create_draft(
            text=pending.get("content", ""),
            profile_ids=[target_channel_id],
            image_url=image_url,
            max_retries=config.pipeline.max_retries,
            backoff_factor=config.pipeline.retry_backoff_seconds,
            dry_run=req.get("dry_run", False)
        )
        state_mgr.mark_completed(buffer_resp, profile_id=target_channel_id)

        return {
            "success": True,
            "cycle_id": pending.get("cycle_id"),
            "buffer_response": buffer_resp
        }
    except Exception as e:
        state_mgr.record_failure(str(e), stage="buffer_draft")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/clear-state")
async def clear_state():
    """Resets the state checkpoint to IDLE."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    state_mgr.clear_state()
    return {"success": True, "message": "State checkpoint reset to IDLE"}


@app.post("/api/test-provider")
async def test_provider(req: TestProviderRequest):
    """Tests connectivity to a specific LLM provider."""
    config = load_config()
    try:
        provider = LLMProviderFactory.create(
            provider_name=req.provider,
            config=config
        )
        test_prompt = "Say 'Provider connection OK' in 5 words."
        resp = provider.generate(prompt=test_prompt, max_tokens=20)
        return {
            "success": True,
            "provider": req.provider,
            "model": resp.model,
            "latency_seconds": resp.latency_seconds,
            "tokens": resp.total_tokens,
            "response": resp.content
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "provider": req.provider, "error": str(e)})


@app.get("/api/history")
async def get_full_history(limit: int = 50):
    """Returns the full published history log."""
    config = load_config()
    state_mgr = StateManager(config.pipeline.state_file, config.pipeline.history_file)
    history = state_mgr.get_recent_history(limit=limit)
    return {"success": True, "count": len(history), "history": history}
