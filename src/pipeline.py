"""
Master Pipeline Orchestrator.
Coordinates StateManager, ContentGenerator, and BufferClient with complete
fault tolerance, resume capabilities, and interactive Human-in-the-Loop (HITL) approval.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from rich.panel import Panel


from src.config import AppConfig, load_config
from src.logger import console, logger
from src.state_manager import StateManager, StateStatus
from src.providers.factory import LLMProviderFactory
from src.generator.content_generator import ContentGenerator, GeneratedPost
from src.buffer.client import BufferClient
from src.image_generator.qwen_image import QwenImageGenerator, GeneratedImage
from src.image_generator.cloudinary_uploader import CloudinaryUploader


class LinkedInBufferPipeline:
    """End-to-end fault tolerant LinkedIn content generation and Buffer drafting pipeline with Human-in-the-Loop."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        provider_name: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ):
        self.config = config or load_config()
        self.dry_run = dry_run if dry_run is not None else self.config.app.dry_run
        
        self.state_manager = StateManager(
            state_file_path=self.config.pipeline.state_file,
            history_file_path=self.config.pipeline.history_file
        )
        
        self.provider = LLMProviderFactory.create(
            provider_name=provider_name or self.config.llm.active_provider,
            config=self.config
        )
        
        self.generator = ContentGenerator(
            provider=self.provider,
            config=self.config
        )
        
        self.buffer_client = BufferClient(
            settings=self.config.buffer
        )

        self.image_generator = QwenImageGenerator(
            config=self.config
        )

        self.cloudinary_uploader = CloudinaryUploader(
            config=self.config
        )

    def _upload_to_cloudinary_if_configured(
        self,
        image_data: Optional[Dict[str, Any]],
        cycle_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Uploads generated image to Cloudinary if credentials are configured."""
        if not image_data or not self.cloudinary_uploader.is_configured():
            return image_data
        try:
            source = image_data.get("local_path") or image_data.get("image_url")
            if source:
                c_res = self.cloudinary_uploader.upload_image(
                    image_source=source,
                    cycle_id=cycle_id,
                    dry_run=self.dry_run
                )
                if c_res.get("secure_url"):
                    image_data["cloudinary_url"] = c_res["secure_url"]
                    image_data["image_url"] = c_res["secure_url"]
                    image_data["cloudinary_public_id"] = c_res.get("public_id")
        except Exception as e:
            logger.warning(f"Could not upload to Cloudinary: {e}. Keeping existing image URL.")
        return image_data


    def _prompt_user_action(self) -> str:
        """
        Interactive Human-in-the-Loop menu.
        Returns: 'publish', 'edit', 'image', 'regenerate', 'save', or 'cancel'
        """
        console.print("\n[bold cyan]👤 Human-In-The-Loop Review:[/bold cyan]")
        console.print("  [bold green][P][/bold green] Publish Draft to Buffer (with Image Asset if attached)")
        console.print("  [bold yellow][E][/bold yellow] Edit text before publishing")
        console.print("  [bold magenta][I][/bold magenta] Regenerate Image only (Qwen-Image-3.0)")
        console.print("  [bold blue][R][/bold blue] Regenerate with new angle / refinements")
        console.print("  [bold magenta][S][/bold magenta] Save to checkpoint only (Exit without drafting)")
        console.print("  [bold red][C][/bold red] Cancel & Discard")

        try:
            choice = input("\nWould you like to save this draft to Buffer? [P/e/i/r/s/c] (default: P): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            return "cancel"

        if not choice or choice in ("p", "publish", "yes", "y"):
            return "publish"
        elif choice in ("e", "edit"):
            return "edit"
        elif choice in ("i", "image", "img"):
            return "image"
        elif choice in ("r", "regenerate", "regen"):
            return "regenerate"
        elif choice in ("s", "save"):
            return "save"
        elif choice in ("c", "cancel", "q", "quit", "n", "no"):
            return "cancel"
        return "publish"

    def _edit_content_interactive(self, current_text: str) -> str:
        """Opens editor or terminal prompt to allow user to tweak generated text."""
        editor = os.getenv("EDITOR") or ("nano" if shutil.which("nano") else ("vim" if shutil.which("vim") else None))
        if editor:
            console.print(f"[cyan]Opening system editor ({editor}) for post refinement...[/cyan]")
            with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(current_text)
                tf_path = tf.name
            try:
                subprocess.call([editor, tf_path])
                with open(tf_path, "r", encoding="utf-8") as f:
                    edited = f.read().strip()
                if edited:
                    logger.info("[green]✓ Post updated from editor.[/green]")
                    return edited
            except Exception as e:
                logger.warning(f"Could not launch editor: {e}")
            finally:
                if os.path.exists(tf_path):
                    os.remove(tf_path)

        # Fallback text input
        console.print("[cyan]Enter modified text (Type your changes, press Ctrl+D or Ctrl+Z when done):[/cyan]")
        try:
            lines = sys.stdin.read().strip()
            if lines:
                return lines
        except Exception:
            pass
        return current_text

    def run(
        self,
        topic: Optional[str] = None,
        pillar_id: Optional[str] = None,
        hook_style: Optional[str] = None,
        story_anchor: Optional[str] = None,
        user_mind: Optional[str] = None,
        force_regenerate: bool = False,
        interactive: bool = True,
        generate_image: Optional[bool] = None,
        standby_photo: Optional[str] = None,
        image_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the content generation, Qwen-Image-3.0 image synthesis,
        Human-in-the-loop review, and Buffer drafting pipeline.
        """
        console.rule("[bold cyan]LinkedIn Content Automation Pipeline[/bold cyan]")

        should_gen_image = generate_image if generate_image is not None else getattr(self.config.image_generation, "enabled", True)

        # ----------------------------------------------------------------------
        # STAGE 1: Check Checkpoint / State or Generate Content
        # ----------------------------------------------------------------------
        pending_post = self.state_manager.get_pending_generation()
        cycle_id = None
        post_content = ""
        post_metadata: Dict[str, Any] = {}
        image_data: Optional[Dict[str, Any]] = None

        if pending_post and not force_regenerate:
            logger.info(
                f"[bold yellow]⚡ Resuming previous cycle ({pending_post.get('cycle_id')}) from checkpoint! "
                f"Skipping LLM & Image generation to conserve API credits.[/bold yellow]"
            )
            post_content = pending_post.get("content", "")
            post_metadata = pending_post.get("metadata", {})
            image_data = pending_post.get("image_data")
            cycle_id = pending_post.get("cycle_id")
        else:
            if force_regenerate and pending_post:
                logger.info("[yellow]Force regenerate requested. Discarding cached checkpoint post.[/yellow]")
                self.state_manager.clear_state()

            # 1. Generate new text post via LLM
            step_total = "3" if should_gen_image else "2"
            logger.info(f"[bold blue]Step 1/{step_total}: Generating Content using {self.provider.provider_name.upper()}...[/bold blue]")
            generated: GeneratedPost = self.generator.generate_post(
                topic=topic,
                pillar_id=pillar_id,
                hook_style=hook_style,
                story_anchor=story_anchor,
                user_mind=user_mind
            )
            post_content = generated.content
            post_metadata = generated.metadata

            # Save initial checkpoint
            cycle_id = self.state_manager.save_generation(
                post_content=post_content,
                metadata=post_metadata
            )

            # 2. Generate context-aware image via Qwen-Image-3.0 if enabled
            if should_gen_image:
                logger.info(f"[bold blue]Step 2/{step_total}: Synthesizing Visual Asset with QWEN-IMAGE-3.0...[/bold blue]")
                try:
                    img_prompt = image_prompt_override or self.generator.generate_image_prompt(
                        post_content=post_content,
                        topic=topic,
                        pillar_name=post_metadata.get("pillar_name"),
                        story_anchor=story_anchor,
                        user_mind=user_mind
                    )
                    gen_img: GeneratedImage = self.image_generator.generate_image(
                        prompt=img_prompt,
                        standby_photo_path=standby_photo,
                        dry_run=self.dry_run,
                        cycle_id=cycle_id
                    )
                    image_data = {
                        "prompt": gen_img.prompt,
                        "image_url": gen_img.image_url,
                        "local_path": gen_img.local_path,
                        "model": gen_img.model,
                        "status": gen_img.status,
                        "standby_photo_used": gen_img.standby_photo_used,
                        "generated_at": gen_img.generated_at
                    }
                    # Upload to Cloudinary if configured
                    image_data = self._upload_to_cloudinary_if_configured(image_data, cycle_id=cycle_id)
                    # Update checkpoint with image info
                    self.state_manager.save_generation(
                        post_content=post_content,
                        metadata=post_metadata,
                        image_data=image_data,
                        cycle_id=cycle_id
                    )
                except Exception as img_err:
                    logger.error(f"[red]Image generation encountered an error: {img_err}. Continuing with text draft.[/red]")

        # ----------------------------------------------------------------------
        # HUMAN-IN-THE-LOOP (HITL) REVIEW LOOP
        # ----------------------------------------------------------------------
        while True:
            # Display post preview in console
            console.print(
                Panel(
                    post_content,
                    title=f"[bold green]Generated LinkedIn Draft (Cycle: {cycle_id})[/bold green]",
                    subtitle=f"Characters: {len(post_content)} | Provider: {post_metadata.get('provider', 'N/A')}",
                    border_style="green",
                    padding=(1, 2)
                )
            )

            # Display image asset panel if present
            if image_data:
                standby_name = Path(image_data['standby_photo_used']).name if image_data.get('standby_photo_used') else "None (Text-to-Image)"
                c_url = image_data.get('cloudinary_url')
                cdn_line = f"[bold cyan]• Cloudinary CDN:[/bold cyan] [green]{c_url}[/green]\n" if c_url else ""
                img_summary = (
                    f"[bold cyan]• Visual Prompt:[/bold cyan] {image_data.get('prompt')}\n"
                    f"[bold cyan]• Standby Photo Used:[/bold cyan] {standby_name}\n"
                    f"[bold cyan]• Local Image File:[/bold cyan] {image_data.get('local_path') or 'Cached'}\n"
                    f"{cdn_line}"
                    f"[bold cyan]• Image Asset URL:[/bold cyan] {image_data.get('image_url')}"
                )
                console.print(
                    Panel(
                        img_summary,
                        title="[bold magenta]🎨 Generated Image Asset (Qwen-Image-3.0 / Cloudinary)[/bold magenta]",
                        subtitle=f"Model: {image_data.get('model', 'qwen-image-3.0')} | Status: {image_data.get('status', 'OK').upper()}",
                        border_style="magenta",
                        padding=(1, 2)
                    )
                )

            if not interactive:
                # In non-interactive / CI / script mode, automatically proceed
                break

            action = self._prompt_user_action()

            if action == "publish":
                break

            elif action == "edit":
                post_content = self._edit_content_interactive(post_content)
                self.state_manager.save_generation(post_content, post_metadata, image_data=image_data, cycle_id=cycle_id)
                continue

            elif action == "image":
                refine_prompt = input("\n🎨 Enter custom prompt for Qwen-Image-3.0 (Press Enter to regenerate from post context): ").strip()
                active_prompt = refine_prompt if refine_prompt else self.generator.generate_image_prompt(
                    post_content=post_content,
                    topic=topic,
                    pillar_name=post_metadata.get("pillar_name"),
                    story_anchor=story_anchor,
                    user_mind=user_mind
                )
                logger.info("[cyan]Regenerating image with Qwen-Image-3.0...[/cyan]")
                gen_img = self.image_generator.generate_image(
                    prompt=active_prompt,
                    standby_photo_path=standby_photo,
                    dry_run=self.dry_run,
                    cycle_id=cycle_id
                )
                image_data = {
                    "prompt": gen_img.prompt,
                    "image_url": gen_img.image_url,
                    "local_path": gen_img.local_path,
                    "model": gen_img.model,
                    "status": gen_img.status,
                    "standby_photo_used": gen_img.standby_photo_used,
                    "generated_at": gen_img.generated_at
                }
                image_data = self._upload_to_cloudinary_if_configured(image_data, cycle_id=cycle_id)
                self.state_manager.save_generation(post_content, post_metadata, image_data=image_data, cycle_id=cycle_id)
                continue

            elif action == "regenerate":
                refinement = input("\n💭 Any specific thoughts or adjustments for the new draft? (Press Enter to keep current): ").strip()
                active_mind = refinement if refinement else user_mind
                logger.info("[cyan]Regenerating draft with LLM...[/cyan]")
                generated = self.generator.generate_post(
                    topic=topic,
                    pillar_id=pillar_id,
                    hook_style=hook_style,
                    story_anchor=story_anchor,
                    user_mind=active_mind
                )
                post_content = generated.content
                post_metadata = generated.metadata

                if should_gen_image:
                    new_img_prompt = self.generator.generate_image_prompt(
                        post_content=post_content,
                        topic=topic,
                        pillar_name=post_metadata.get("pillar_name"),
                        story_anchor=story_anchor,
                        user_mind=active_mind
                    )
                    gen_img = self.image_generator.generate_image(
                        prompt=new_img_prompt,
                        standby_photo_path=standby_photo,
                        dry_run=self.dry_run,
                        cycle_id=cycle_id
                    )
                    image_data = {
                        "prompt": gen_img.prompt,
                        "image_url": gen_img.image_url,
                        "local_path": gen_img.local_path,
                        "model": gen_img.model,
                        "status": gen_img.status,
                        "standby_photo_used": gen_img.standby_photo_used,
                        "generated_at": gen_img.generated_at
                    }
                    image_data = self._upload_to_cloudinary_if_configured(image_data, cycle_id=cycle_id)

                cycle_id = self.state_manager.save_generation(post_content, post_metadata, image_data=image_data, cycle_id=cycle_id)
                continue

            elif action == "save":
                logger.info(f"[yellow]💾 Post and image saved to state checkpoint ({cycle_id}). Exiting without publishing to Buffer.[/yellow]")
                logger.info("Run `uv run python main.py resume` anytime to push this draft to Buffer.")
                return {
                    "success": True,
                    "cycle_id": cycle_id,
                    "status": "saved_to_checkpoint",
                    "content": post_content,
                    "metadata": post_metadata,
                    "image_data": image_data
                }

            elif action == "cancel":
                logger.info("[yellow]Draft discarded.[/yellow]")
                self.state_manager.clear_state()
                return {
                    "success": False,
                    "status": "cancelled",
                    "cycle_id": cycle_id
                }

        # ----------------------------------------------------------------------
        # STAGE 2: Publish Draft to Buffer API
        # ----------------------------------------------------------------------
        step_final = "Step 3/3" if should_gen_image else "Step 2/2"
        logger.info(f"[bold blue]{step_final}: Publishing Draft to Buffer API...[/bold blue]")
        
        try:
            target_profile_id = self.config.buffer.profile_id
            
            # In live mode without explicit profile_id, resolve LinkedIn profile
            if not self.dry_run and not target_profile_id:
                try:
                    target_profile_id = self.buffer_client.get_linkedin_profile_id()
                except Exception as e:
                    logger.warning(f"[yellow]Could not auto-resolve LinkedIn profile: {e}. Will attempt default.[/yellow]")
            elif self.dry_run and not target_profile_id:
                target_profile_id = "dry_run_linkedin_profile"

            image_url_to_draft = image_data.get("image_url") if image_data else None

            buffer_response = self.buffer_client.create_draft(
                text=post_content,
                profile_ids=[target_profile_id] if target_profile_id else None,
                image_url=image_url_to_draft,
                max_retries=self.config.pipeline.max_retries,
                backoff_factor=self.config.pipeline.retry_backoff_seconds,
                dry_run=self.dry_run
            )

            # Record success and archive
            self.state_manager.mark_completed(
                buffer_response=buffer_response,
                profile_id=target_profile_id
            )

            mode_str = "[DRY-RUN SIMULATION]" if self.dry_run else "[BUFFER LIVE DRAFT]"
            logger.info(f"[bold green]✨ Pipeline completed successfully! {mode_str}[/bold green]")
            return {
                "success": True,
                "cycle_id": cycle_id,
                "dry_run": self.dry_run,
                "content": post_content,
                "metadata": post_metadata,
                "image_data": image_data,
                "buffer_response": buffer_response
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[bold red]❌ Failed during Buffer draft creation: {error_msg}[/bold red]")
            self.state_manager.record_failure(error_msg, stage="buffer_draft")
            logger.info(
                "[bold yellow]💡 Tip: Your generated post and image are safely saved in state checkpoint! "
                "Fix your Buffer token/settings and run `uv run python main.py resume` to push it without wasting LLM or image credits.[/bold yellow]"
            )
            return {
                "success": False,
                "cycle_id": cycle_id,
                "error": error_msg,
                "content": post_content,
                "metadata": post_metadata,
                "image_data": image_data
            }

