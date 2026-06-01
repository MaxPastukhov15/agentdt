import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.utils.long_term_mem import load_chat_memory, save_chat_memory


@pytest.fixture
def mock_settings(tmp_path):
    with patch("app.utils.long_term_mem.settings") as mock:
        mock.long_term_memory = tmp_path
        yield mock


def test_load_chat_memory_exists(mock_settings, tmp_path):
    thread_id = "test-thread-123"
    summary_data = {"summary": "Test summary content"}
    file_path = tmp_path / f"{thread_id}_summary.json"
    file_path.write_text(json.dumps(summary_data, ensure_ascii=False), encoding="utf-8")

    result = load_chat_memory(thread_id)

    assert result == "Test summary content"


def test_load_chat_memory_not_exists(mock_settings, tmp_path):
    result = load_chat_memory("nonexistent-thread")
    assert result == ""


def test_save_chat_memory(mock_settings, tmp_path):
    thread_id = "test-thread-456"
    summary = "New summary content"

    save_chat_memory(thread_id, summary)

    file_path = tmp_path / f"{thread_id}_summary.json"
    assert file_path.exists()
    saved = json.loads(file_path.read_text(encoding="utf-8"))
    assert saved["summary"] == summary


def test_save_chat_memory_overwrites(mock_settings, tmp_path):
    thread_id = "test-thread-789"
    file_path = tmp_path / f"{thread_id}_summary.json"
    file_path.write_text(json.dumps({"summary": "old"}, ensure_ascii=False), encoding="utf-8")

    save_chat_memory(thread_id, "new")
    saved = json.loads(file_path.read_text(encoding="utf-8"))
    assert saved["summary"] == "new"


def test_load_chat_memory_corrupted_json(mock_settings, tmp_path):
    thread_id = "corrupted-thread"
    file_path = tmp_path / f"{thread_id}_summary.json"
    file_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_chat_memory(thread_id)
