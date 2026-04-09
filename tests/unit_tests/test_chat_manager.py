import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.core.chat_manager import ChatManager


@pytest_asyncio.fixture
async def chat_manager(tmp_path):
    # Используем путь в памяти для sqlite, чтобы тест летал
    db_path = str(tmp_path / "test.db")

    # Патчим AsyncSqliteSaver, чтобы он не лез в БД реально во время инициализации
    with patch("core.chat_manager.AsyncSqliteSaver") as mock_saver_class:
        # Создаем мок-экземпляр, который будет возвращен при вызове AsyncSqliteSaver()
        mock_saver_inst = AsyncMock()
        mock_saver_class.return_value = mock_saver_inst

        manager = ChatManager(path=db_path)
        await manager.initialize()

        # Сохраняем ссылку на мок внутри объекта для проверок в тестах
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
    # Создаем два чата
    await chat_manager.create_chat("Первый")
    await asyncio.sleep(1.1)
    await chat_manager.create_chat("Второй")

    chats = await chat_manager.list_chats()

    assert len(chats) == 2
    print(chats)
    # Проверяем сортировку (последний созданный должен быть первым)
    assert chats[0]["title"] == "Второй"


@pytest.mark.asyncio
async def test_directory_creation(tmp_path):
    # Проверка, что менеджер создает папку, если её нет
    nested_path = tmp_path / "subdir" / "chat.db"
    manager = ChatManager(path=str(nested_path))

    assert os.path.exists(os.path.dirname(nested_path))


@pytest.mark.asyncio
async def test_delete_chat_calls_checkpointer(chat_manager):
    # 1. Подготовка: создаем чат
    thread_id = await chat_manager.create_chat("Test Chat")

    # 2. Действие: удаляем чат
    await chat_manager.delete_chat(thread_id)

    # 3. Проверка: убеждаемся, что метод adelete_thread у checkpointer был вызван
    # именно с этим thread_id
    chat_manager.checkpointer.adelete_thread.assert_called_once_with(
        thread_id=thread_id
    )


@pytest.mark.asyncio
async def test_initialize_sets_up_checkpointer(tmp_path):
    db_path = str(tmp_path / "init_test.db")

    with patch("core.chat_manager.AsyncSqliteSaver") as mock_saver_class:
        manager = ChatManager(path=db_path)
        await manager.initialize()

        # Проверяем, что AsyncSqliteSaver был инициализирован с объектом соединения
        assert mock_saver_class.called
        # Первый аргумент вызова должен быть объектом aiosqlite.Connection
        args, kwargs = mock_saver_class.call_args
        import aiosqlite

        assert isinstance(args[0], aiosqlite.Connection)

        await manager.close()
