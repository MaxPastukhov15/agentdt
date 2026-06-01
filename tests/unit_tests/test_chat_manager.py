import asyncio
import os
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
import pytest_asyncio

from app.core.chat_manager import ChatManager


@pytest_asyncio.fixture
async def chat_manager(tmp_path):
    db_path = str(tmp_path / "test.db")

    with patch("app.core.chat_manager.AsyncSqliteSaver") as mock_saver_class:
        mock_saver_inst = AsyncMock()
        mock_saver_class.return_value = mock_saver_inst

        manager = ChatManager(path=db_path)
        await manager.initialize()

        manager.checkpointer = mock_saver_inst

        yield manager
        await manager.close()


@pytest.mark.asyncio
async def test_create_chat(chat_manager):
    title = "Тестовый чат"
    thread_id = await chat_manager.create_chat(title)

    assert isinstance(thread_id, str)
    chats = await chat_manager.list_chats()
    assert len(chats) == 1
    assert chats[0]["title"] == title
    assert chats[0]["thread_id"] == thread_id


@pytest.mark.asyncio
async def test_rename_chat(chat_manager):
    thread_id = await chat_manager.create_chat("Старое название")
    new_title = "Новое название"

    await chat_manager.rename_chat(thread_id, new_title)

    chats = await chat_manager.list_chats()
    assert chats[0]["title"] == new_title


@pytest.mark.asyncio
async def test_delete_chat(chat_manager):
    thread_id = await chat_manager.create_chat("На удаление")

    await chat_manager.delete_chat(thread_id)

    chats = await chat_manager.list_chats()
    assert len(chats) == 0


@pytest.mark.asyncio
async def test_list_chats_ordering(chat_manager):
    await chat_manager.create_chat("Первый")
    await asyncio.sleep(1.1)
    await chat_manager.create_chat("Второй")

    chats = await chat_manager.list_chats()

    assert len(chats) == 2
    assert chats[0]["title"] == "Второй"


@pytest.mark.asyncio
async def test_directory_creation(tmp_path):
    nested_path = tmp_path / "subdir" / "chat.db"
    manager = ChatManager(path=str(nested_path))

    assert os.path.exists(os.path.dirname(nested_path))


@pytest.mark.asyncio
async def test_delete_chat_calls_checkpointer(chat_manager):
    thread_id = await chat_manager.create_chat("Test Chat")

    await chat_manager.delete_chat(thread_id)

    chat_manager.checkpointer.adelete_thread.assert_called_once_with(thread_id=thread_id)


@pytest.mark.asyncio
async def test_initialize_sets_up_checkpointer(tmp_path):
    db_path = str(tmp_path / "init_test.db")

    with patch("app.core.chat_manager.AsyncSqliteSaver") as mock_saver_class:
        manager = ChatManager(path=db_path)
        await manager.initialize()

        assert mock_saver_class.called
        args, kwargs = mock_saver_class.call_args
        assert isinstance(args[0], aiosqlite.Connection)

        await manager.close()


@pytest.mark.asyncio
async def test_create_chat_title_truncated(chat_manager):
    long_title = "A" * 100
    thread_id = await chat_manager.create_chat(long_title)

    chats = await chat_manager.list_chats()
    assert len(chats[0]["title"]) == 50


@pytest.mark.asyncio
async def test_create_chat_empty_title(chat_manager):
    thread_id = await chat_manager.create_chat("")

    chats = await chat_manager.list_chats()
    assert chats[0]["title"] == ""


@pytest.mark.asyncio
async def test_list_chats_empty(chat_manager):
    chats = await chat_manager.list_chats()
    assert chats == []


@pytest.mark.asyncio
async def test_delete_chat_nonexistent(chat_manager):
    await chat_manager.delete_chat("nonexistent-id")
    chats = await chat_manager.list_chats()
    assert chats == []


@pytest.mark.asyncio
async def test_rename_chat_nonexistent(chat_manager):
    await chat_manager.rename_chat("nonexistent-id", "New Title")
    chats = await chat_manager.list_chats()
    assert chats == []
