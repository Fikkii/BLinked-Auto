"""
Unit tests for PromptBuilder and Persona Context.
"""

from src.config import load_config
from src.generator.prompt_builder import PromptBuilder, PromptContext


def test_prompt_builder_structure():
    config = load_config()
    builder = PromptBuilder(config)
    
    pillars = builder.get_pillars()
    assert len(pillars) >= 4
    
    # Test specific topic and pillar
    ctx = builder.build_prompt(
        topic="Database indexing bottlenecks",
        pillar_id="backend_mastery",
        hook_style_key="tutoring_story"
    )
    
    assert isinstance(ctx, PromptContext)
    assert "Database indexing bottlenecks" in ctx.user_prompt
    assert "tutoring" in ctx.system_prompt.lower() or "backend" in ctx.system_prompt.lower()
    assert ctx.pillar_id == "backend_mastery"
    assert len(ctx.story_anchor) > 0


def test_prompt_builder_random_selection():
    config = load_config()
    builder = PromptBuilder(config)
    
    # Without arguments, it should successfully pick randomly
    ctx = builder.build_prompt()
    assert ctx.topic is not None
    assert ctx.pillar_name is not None
    assert ctx.system_prompt is not None
    assert ctx.user_prompt is not None


def test_prompt_builder_with_user_mind():
    config = load_config()
    builder = PromptBuilder(config)
    
    thought = "I spent 4 hours rewriting an API rate limiter using Redis token buckets while tutoring undergrads"
    ctx = builder.build_prompt(user_mind=thought)
    assert thought in ctx.user_prompt
    assert thought in ctx.story_anchor

