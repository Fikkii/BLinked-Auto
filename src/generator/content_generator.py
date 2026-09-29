"""
Content Generator Engine.
Coordinates prompt creation, LLM inference, output sanitization, and quality validation.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from src.config import AppConfig
from src.logger import logger
from src.generator.prompt_builder import PromptBuilder, PromptContext
from src.providers.base import BaseLLMProvider, LLMResponse


@dataclass
class GeneratedPost:
    content: str
    metadata: Dict[str, Any]
    response_info: LLMResponse


class ContentGenerator:
    """Orchestrates generation of high-quality LinkedIn posts."""

    def __init__(self, provider: BaseLLMProvider, config: AppConfig):
        self.provider = provider
        self.config = config
        self.prompt_builder = PromptBuilder(config)

    def _sanitize_output(self, raw_text: str) -> str:
        """Removes markdown code fences, preambles, and cleans whitespace."""
        text = raw_text.strip()

        # Strip markdown ``` or ```markdown wrappers if LLM returned them
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove leading "Here is a LinkedIn post..." or "Sure! Here is the post:"
        intro_patterns = [
            r"^(?:Here is|Here's|Sure,? here is|Certainly,? here is)\s+.*?(?:post|content|draft):\s*\n+",
            r"^#+ (?:LinkedIn Post|Draft Post|Post)\s*\n+"
        ]
        for pat in intro_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

        return text

    def _validate_post(self, content: str) -> bool:
        """Validates that the post meets LinkedIn best-practice requirements."""
        if len(content) < 100:
            logger.warning("[yellow]Generated post is too short (< 100 chars).[/yellow]")
            return False

        if len(content) > 3000:
            logger.warning("[yellow]Generated post exceeds LinkedIn character limit (3000 chars).[/yellow]")
            return False

        # Check for hashtags
        if "#" not in content:
            logger.warning("[yellow]Post does not contain hashtags.[/yellow]")

        return True

    def generate_post(
        self,
        topic: Optional[str] = None,
        pillar_id: Optional[str] = None,
        hook_style: Optional[str] = None,
        story_anchor: Optional[str] = None,
        target_takeaway: Optional[str] = None,
        user_mind: Optional[str] = None
    ) -> GeneratedPost:
        """
        Generates and validates a complete LinkedIn post.
        """
        ctx: PromptContext = self.prompt_builder.build_prompt(
            topic=topic,
            pillar_id=pillar_id,
            hook_style_key=hook_style,
            story_anchor=story_anchor,
            target_takeaway=target_takeaway,
            user_mind=user_mind
        )

        logger.info(f"[bold cyan]Generating LinkedIn Post[/bold cyan]")
        logger.info(f"[cyan]• Pillar:[/cyan] {ctx.pillar_name}")
        logger.info(f"[cyan]• Topic:[/cyan] {ctx.topic}")
        logger.info(f"[cyan]• Hook Style:[/cyan] {ctx.hook_style}")
        if user_mind:
            logger.info(f"[cyan]• User Mind:[/cyan] {user_mind[:80]}...")
        logger.info(f"[cyan]• Story Anchor:[/cyan] {ctx.story_anchor[:80]}...")

        # Invoke LLM
        response: LLMResponse = self.provider.generate(
            prompt=ctx.user_prompt,
            system_prompt=ctx.system_prompt,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            top_p=self.config.llm.top_p
        )

        sanitized_content = self._sanitize_output(response.content)
        self._validate_post(sanitized_content)

        metadata = {
            "pillar_id": ctx.pillar_id,
            "pillar_name": ctx.pillar_name,
            "topic": ctx.topic,
            "hook_style": ctx.hook_style,
            "story_anchor": ctx.story_anchor,
            "target_takeaway": ctx.target_takeaway,
            "provider": response.provider,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "latency_seconds": response.latency_seconds,
        }

        return GeneratedPost(
            content=sanitized_content,
            metadata=metadata,
            response_info=response
        )

    def generate_image_prompt(
        self,
        post_content: str,
        topic: Optional[str] = None,
        pillar_name: Optional[str] = None,
        story_anchor: Optional[str] = None,
        user_mind: Optional[str] = None,
    ) -> str:
        """
        Generates a context-aware image generation prompt tailored for Qwen-Image-3.0
        using the active LLM provider.
        """
        sys_prompt, usr_prompt = self.prompt_builder.build_image_prompt(
            post_content=post_content,
            topic=topic,
            pillar_name=pillar_name,
            story_anchor=story_anchor,
            user_mind=user_mind,
        )

        logger.info("[bold cyan]Synthesizing Qwen-Image-3.0 Visual Prompt...[/bold cyan]")
        try:
            resp = self.provider.generate(
                prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=0.6,
                max_tokens=300,
            )
            raw_prompt = self._sanitize_output(resp.content)
            # Remove any leading "Prompt:" or quotes
            clean_prompt = re.sub(r'^(?:Prompt|Image Prompt):\s*', '', raw_prompt, flags=re.IGNORECASE).strip(' "\'')
            logger.info(f"[green]✓ Image prompt crafted:[/green] {clean_prompt[:90]}...")
            return clean_prompt
        except Exception as e:
            logger.warning(f"[yellow]Failed to generate image prompt via LLM: {e}. Using intelligent fallback.[/yellow]")
            topic_str = topic or "Fullstack Software Engineering & Backend Tutoring"
            return (
                f"Featuring the person from the reference photo, depict them naturally at a modern developer workstation "
                f"with multiple monitors displaying clean Python code and architectural diagrams, warm atmospheric lighting, "
                f"focused and thoughtful posture, illustrating the concept of {topic_str}. "
                f"Photorealistic 35mm lens photography, sharp focus, natural office depth of field."
            )

