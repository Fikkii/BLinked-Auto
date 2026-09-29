"""
Unit tests for End-to-End Pipeline, Human-in-the-Loop Review, and Token-Saving Resume Workflow.
"""

from unittest.mock import MagicMock, patch
from src.config import load_config
from src.pipeline import LinkedInBufferPipeline
from src.providers.base import LLMResponse
from src.state_manager import StateManager


def test_pipeline_generate_and_draft_success(tmp_path):
    config = load_config()
    config.pipeline.state_file = str(tmp_path / "state.json")
    config.pipeline.history_file = str(tmp_path / "history.jsonl")
    
    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    mock_llm_resp = LLMResponse(
        content="5 years of Python taught me one thing: simplicity beats complexity.\n\n#Python #WebDev",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=150
    )

    with patch.object(pipeline.provider, "generate", return_value=mock_llm_resp) as mock_gen:
        result = pipeline.run(topic="Simplicity in backend architecture", interactive=False, generate_image=False)
        assert result["success"] is True
        mock_gen.assert_called_once()
        assert result.get("image_data") is None
        
        # State should now be idle since it finished successfully
        assert not pipeline.state_manager.has_pending_generation()


def test_pipeline_with_user_mind(tmp_path):
    config = load_config()
    config.pipeline.state_file = str(tmp_path / "state.json")
    config.pipeline.history_file = str(tmp_path / "history.jsonl")
    
    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    mock_llm_resp = LLMResponse(
        content="During office hours, an undergrad spent 3 hours debugging database deadlocks.\n\n#Python #Databases",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=180
    )

    with patch.object(pipeline.provider, "generate", return_value=mock_llm_resp) as mock_gen:
        result = pipeline.run(user_mind="Debugging deadlocks during tutoring", interactive=False, generate_image=False)
        assert result["success"] is True
        mock_gen.assert_called_once()
        call_prompt = mock_gen.call_args[1]["prompt"]
        assert "Debugging deadlocks during tutoring" in call_prompt


def test_pipeline_generate_with_image_success(tmp_path):
    config = load_config()
    config.pipeline.state_file = str(tmp_path / "state.json")
    config.pipeline.history_file = str(tmp_path / "history.jsonl")
    
    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    mock_post_resp = LLMResponse(
        content="5 years of Python taught me one thing: simplicity beats complexity.\n\n#Python #WebDev",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=150
    )
    mock_img_prompt_resp = LLMResponse(
        content="Photorealistic editorial portrait of software engineer at clean workspace with dual screens displaying Python code.",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=60
    )

    with patch.object(pipeline.provider, "generate", side_effect=[mock_post_resp, mock_img_prompt_resp]) as mock_gen:
        result = pipeline.run(topic="Simplicity in backend architecture", interactive=False, generate_image=True)
        assert result["success"] is True
        assert mock_gen.call_count == 2
        assert result["image_data"] is not None
        assert "Photorealistic editorial portrait" in result["image_data"]["prompt"]
        assert result["image_data"]["local_path"] is not None
        assert not pipeline.state_manager.has_pending_generation()


def test_pipeline_human_in_the_loop_approval(tmp_path):
    config = load_config()
    config.pipeline.state_file = str(tmp_path / "state.json")
    config.pipeline.history_file = str(tmp_path / "history.jsonl")
    
    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    mock_llm_resp = LLMResponse(
        content="Testing Human in the loop post.\n\n#SoftwareEngineering",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=100
    )

    with patch.object(pipeline.provider, "generate", return_value=mock_llm_resp):
        with patch.object(pipeline, "_prompt_user_action", return_value="publish"):
            result = pipeline.run(interactive=True)
            assert result["success"] is True


def test_pipeline_human_in_the_loop_save_only(tmp_path):
    config = load_config()
    config.pipeline.state_file = str(tmp_path / "state.json")
    config.pipeline.history_file = str(tmp_path / "history.jsonl")
    
    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    mock_llm_resp = LLMResponse(
        content="Post to save only.\n\n#SoftwareEngineering",
        model="gemini-3.6-flash",
        provider="gemini",
        total_tokens=100
    )

    with patch.object(pipeline.provider, "generate", return_value=mock_llm_resp):
        with patch.object(pipeline, "_prompt_user_action", return_value="save"):
            result = pipeline.run(interactive=True)
            assert result["status"] == "saved_to_checkpoint"
            # State must have pending post!
            assert pipeline.state_manager.has_pending_generation()


def test_pipeline_resume_saves_llm_call(tmp_path):
    config = load_config()
    state_file = str(tmp_path / "state.json")
    history_file = str(tmp_path / "history.jsonl")
    config.pipeline.state_file = state_file
    config.pipeline.history_file = history_file
    
    # Pre-populate state with a generated post that failed at buffer draft stage
    state_mgr = StateManager(state_file, history_file)
    state_mgr.save_generation(
        post_content="Already generated post waiting in state! #Tech",
        metadata={"pillar": "backend_mastery", "provider": "gemini"}
    )
    state_mgr.record_failure("Buffer network timeout", stage="buffer_draft")

    pipeline = LinkedInBufferPipeline(config=config, provider_name="gemini", dry_run=True)
    
    # Verify that pipeline runs and completes WITHOUT calling provider.generate!
    with patch.object(pipeline.provider, "generate") as mock_gen:
        result = pipeline.run(force_regenerate=False, interactive=False)
        assert result["success"] is True
        assert result["content"] == "Already generated post waiting in state! #Tech"
        
        # LLM MUST NOT BE CALLED!
        mock_gen.assert_not_called()
        
        # Checkpoint is cleared and archived to history
        assert not pipeline.state_manager.has_pending_generation()
        history = pipeline.state_manager.get_recent_history(limit=1)
        assert len(history) == 1
        assert "Already generated post" in history[0]["post_content"]
