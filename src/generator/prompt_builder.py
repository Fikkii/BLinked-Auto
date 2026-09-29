"""
Prompt Builder for LinkedIn Content Generation.
Combines creator persona, tutoring anecdotes, fullstack & UI/UX experience,
and high-converting LinkedIn hook frameworks.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.config import AppConfig


@dataclass
class PromptContext:
    system_prompt: str
    user_prompt: str
    pillar_id: str
    pillar_name: str
    topic: str
    hook_style: str
    story_anchor: str
    target_takeaway: str


class PromptBuilder:
    """Builds highly structured, persona-infused prompts for LinkedIn posts."""

    # Rich storytelling anchors based on 5-year fullstack exp, backend tutoring, UI/UX, AI automations
    STORY_ANCHORS = [
        "A university student in my backend tutoring session spent 2 hours debugging a 500 error that turned out to be an unhandled promise rejection.",
        "During office hours yesterday, a student asked me: 'Why do we need database indexes if our query already returns in 50ms locally?'",
        "When I started adding UI/UX design (Figma) after 5 years of writing Python APIs, I realized our frontend engineers weren't slow—our backend responses lacked UX feedback states.",
        "A junior dev in my tutoring group asked why their async loop in Python was blocking the entire FastAPI server.",
        "Last month we refactored an AI automation script that was burning $40/day in unnecessary LLM token calls by introducing state checkpointing.",
        "5 years ago, I thought a 'good backend' meant raw throughput. Today, after tutoring undergrads and designing UI/UX flows, I know clarity and predictability beat cleverness.",
        "Teaching a group of 20 university students how database connection pools work forced me to eliminate all buzzwords and explain it with a restaurant kitchen analogy.",
        "When designing our new app interface in Figma, I caught 3 database schema flaws before writing a single line of backend code."
    ]

    TARGET_TAKEAWAYS = [
        "How understanding system fundamentals and communication makes you a resilient fullstack engineer.",
        "The exact mental model for designing developer-friendly APIs and intuitive UI/UX states.",
        "Actionable steps to build fault-tolerant AI automations with state checkpointing and token efficiency.",
        "Why the ability to explain complex technical concepts simply is the most underrated senior developer superpower.",
        "A practical 4-step framework for bridging backend data architecture with frontend user experience."
    ]

    def __init__(self, config: AppConfig):
        self.config = config
        self.persona = config.persona
        self.prompts = config.prompts

    def get_pillars(self) -> List[Dict[str, Any]]:
        """Returns the list of content pillars from persona configuration."""
        return self.persona.get("content_pillars", [])

    def build_prompt(
        self,
        topic: Optional[str] = None,
        pillar_id: Optional[str] = None,
        hook_style_key: Optional[str] = None,
        story_anchor: Optional[str] = None,
        target_takeaway: Optional[str] = None,
        user_mind: Optional[str] = None
    ) -> PromptContext:
        """
        Builds a comprehensive prompt context. If user_mind is provided,
        it anchors the post directly around the user's specific thought/experience.
        """
        pillars = self.get_pillars()
        if not pillars:
            # Fallback default pillar if config missing
            selected_pillar = {
                "id": "backend_mastery",
                "name": "Backend Architecture & Fullstack Engineering",
                "sample_topics": ["Lessons from 5 years of Python & JavaScript in production"]
            }
        elif pillar_id:
            selected_pillar = next((p for p in pillars if p.get("id") == pillar_id), pillars[0])
        else:
            selected_pillar = random.choice(pillars)

        # Select topic
        if user_mind and not topic:
            topic = f"Insight / Reflection: {user_mind[:90]}..."
        elif not topic:
            sample_topics = selected_pillar.get("sample_topics", ["Fullstack & AI Engineering Insights"])
            topic = random.choice(sample_topics)

        # Select hook style
        hook_styles = self.prompts.get("hook_styles", {})
        if hook_style_key and hook_style_key in hook_styles:
            hook_data = hook_styles[hook_style_key]
            hook_name = hook_data.get("name", hook_style_key)
            hook_pattern = hook_data.get("pattern", "")
        else:
            hook_key, hook_data = random.choice(list(hook_styles.items())) if hook_styles else ("contrarian", {"name": "Contrarian Take", "pattern": "Debunk a common dev myth"})
            hook_name = hook_data.get("name", hook_key)
            hook_pattern = hook_data.get("pattern", "")

        # Select story anchor
        if user_mind:
            story = f"What's on my mind: \"{user_mind}\""
            if story_anchor:
                story = f"{story}\nAdditional context: {story_anchor}"
        else:
            story = story_anchor or random.choice(self.STORY_ANCHORS)
            
        takeaway = target_takeaway or random.choice(self.TARGET_TAKEAWAYS)

        # Build system prompt
        system_prompt = self.prompts.get("system_prompt", "You are an expert technical content strategist.")

        # Build user prompt
        template = self.prompts.get("user_prompt_template", "")
        if template:
            user_prompt = template.format(
                pillar_name=selected_pillar.get("name", "Engineering"),
                topic=topic,
                hook_style_name=hook_name,
                hook_style_pattern=hook_pattern,
                story_anchor=story,
                target_takeaway=takeaway
            )
        else:
            user_prompt = (
                f"Write a LinkedIn post about '{topic}' within the pillar '{selected_pillar.get('name')}'. "
                f"Hook: {hook_name} ({hook_pattern}). Story anchor: {story}. Takeaway: {takeaway}."
            )

        if user_mind:
            user_prompt += f"\n\nIMPORTANT: Deeply incorporate my core reflection/thought: '{user_mind}' into the story and technical lessons."

        return PromptContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pillar_id=selected_pillar.get("id", "general"),
            pillar_name=selected_pillar.get("name", "General"),
            topic=topic,
            hook_style=hook_name,
            story_anchor=story,
            target_takeaway=takeaway
        )

    def build_image_prompt(
        self,
        post_content: str,
        topic: Optional[str] = None,
        pillar_name: Optional[str] = None,
        story_anchor: Optional[str] = None,
        user_mind: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Builds the system and user prompts to instruct the LLM to generate
        a photorealistic, context-relevant image generation prompt for Qwen-Image-3.0.
        """
        system_prompt = self.prompts.get(
            "image_prompt_system",
            "You are an expert AI visual director creating a photorealistic scene prompt for Qwen-Image-3.0 based on a technical LinkedIn post."
        )

        template = self.prompts.get(
            "image_prompt_template",
            "Create a Qwen-Image-3.0 prompt for a LinkedIn post about {topic}.\nStory: {story_anchor}\nKey lines:\n{post_summary}"
        )

        # Extract first 3-4 lines or summary of post
        post_lines = [l.strip() for l in post_content.split("\n") if l.strip()]
        post_summary = "\n".join(post_lines[:4]) if post_lines else post_content[:200]

        story = story_anchor or (f"What's on my mind: {user_mind}" if user_mind else "Engineering problem solving")

        user_prompt = template.format(
            topic=topic or "Fullstack Engineering & Tutoring",
            pillar_name=pillar_name or "Engineering & Mentorship",
            story_anchor=story,
            post_summary=post_summary
        )

        if user_mind and user_mind not in story:
            user_prompt += f"\nCore Reflection / Thought: {user_mind}\n"

        return system_prompt, user_prompt

