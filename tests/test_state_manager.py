"""
Unit tests for StateManager and Fault-Tolerance Checkpointing.
"""

import json
import pytest
from src.state_manager import StateManager, StateStatus


@pytest.fixture
def temp_state_manager(tmp_path):
    state_file = tmp_path / "state.json"
    history_file = tmp_path / "history.jsonl"
    return StateManager(str(state_file), str(history_file))


def test_initial_state(temp_state_manager):
    assert not temp_state_manager.has_pending_generation()
    assert temp_state_manager.get_pending_generation() is None
    state = temp_state_manager.get_state()
    assert state["status"] == StateStatus.IDLE


def test_save_generation_and_recovery(temp_state_manager):
    content = "Why most junior backend devs struggle with async loops..."
    metadata = {"pillar": "backend_mastery", "provider": "gemini"}
    
    cycle_id = temp_state_manager.save_generation(content, metadata)
    assert cycle_id is not None
    assert temp_state_manager.has_pending_generation()
    
    pending = temp_state_manager.get_pending_generation()
    assert pending is not None
    assert pending["content"] == content
    assert pending["metadata"]["pillar"] == "backend_mastery"
    assert pending["cycle_id"] == cycle_id


def test_record_failure_preserves_content(temp_state_manager):
    content = "Important post content"
    temp_state_manager.save_generation(content, {"test": 123})
    
    temp_state_manager.record_failure("Buffer API Connection Timeout", stage="buffer_draft")
    
    state = temp_state_manager.get_state()
    assert state["status"] == StateStatus.FAILED
    assert state["retry_count"] == 1
    assert "Timeout" in state["last_error"]
    
    # Crucial fault-tolerance test: Content must still exist to be resumed
    assert temp_state_manager.has_pending_generation()
    pending = temp_state_manager.get_pending_generation()
    assert pending["content"] == content


def test_mark_completed_archives_and_resets(temp_state_manager):
    content = "Post to be completed"
    cycle_id = temp_state_manager.save_generation(content, {"topic": "FastAPI"})
    
    buffer_resp = {"id": "buf_update_999", "status": "draft"}
    temp_state_manager.mark_completed(buffer_resp, profile_id="prof_linkedin_123")
    
    # State should be reset to IDLE
    assert not temp_state_manager.has_pending_generation()
    state = temp_state_manager.get_state()
    assert state["status"] == StateStatus.IDLE
    assert state["pending_post"] is None
    
    # History should contain the archived post
    history = temp_state_manager.get_recent_history(limit=5)
    assert len(history) == 1
    assert history[0]["cycle_id"] == cycle_id
    assert history[0]["post_content"] == content
    assert history[0]["buffer_update_id"] == "buf_update_999"
