import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Must be applied BEFORE any test module imports flet
if "flet" not in sys.modules:
    sys.modules["flet"] = MagicMock()


@pytest.fixture
def mock_ui_classes():
    """Patch UI classes used inside presenters."""
    with (
        patch("app.presenters.chat_presenter.ChatHistoryView") as MockHistoryView,
        patch("app.presenters.chat_presenter.ChatInput") as MockInput,
        patch("app.presenters.main_presenter.SidebarView") as MockSidebar,
        patch("app.presenters.main_presenter.ChatPresenter") as MockChatPresenter,
    ):
        hview = MagicMock()
        inview = MagicMock()
        sidebar = MagicMock()
        chat_ptr = MagicMock()
        chat_ptr.load_chat = AsyncMock()

        MockHistoryView.return_value = hview
        MockInput.return_value = inview
        MockSidebar.return_value = sidebar
        MockChatPresenter.return_value = chat_ptr

        yield {
            "ChatHistoryView": MockHistoryView,
            "ChatInput": MockInput,
            "SidebarView": MockSidebar,
            "ChatPresenter": MockChatPresenter,
            "history_view": hview,
            "input_view": inview,
            "sidebar": sidebar,
            "chat_ptr": chat_ptr,
        }


@pytest.fixture
def mock_model_and_chat_manager():
    agent = MagicMock()
    agent.app = MagicMock()
    chat_manager = MagicMock()
    chat_manager.list_chats = AsyncMock()
    chat_manager.create_chat = AsyncMock()
    chat_manager.delete_chat = AsyncMock()
    chat_manager.rename_chat = AsyncMock()
    return agent, chat_manager
