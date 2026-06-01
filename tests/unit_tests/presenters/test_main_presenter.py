from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.presenters.main_presenter import MainPresenter


@pytest.fixture
def page():
    p = MagicMock()
    p.run_task = AsyncMock()
    return p


@pytest.fixture
def main_presenter(mock_ui_classes, mock_model_and_chat_manager, page):
    agent, chat_manager = mock_model_and_chat_manager
    presenter = MainPresenter(page, agent, chat_manager)
    return presenter


class TestMainPresenterInit:
    def test_stores_page(self, main_presenter, page):
        assert main_presenter.page is page

    def test_stores_model_core(self, main_presenter, mock_model_and_chat_manager):
        agent, _ = mock_model_and_chat_manager
        assert main_presenter.model_core is agent

    def test_creates_file_picker(self, main_presenter):
        assert main_presenter.file_picker is not None

    def test_adds_docs_dialog_to_overlay(self, main_presenter, page):
        page.overlay.append.assert_called_once_with(main_presenter.docs_dialog)

    def test_creates_sidebar(self, main_presenter, mock_ui_classes):
        mock_ui_classes["SidebarView"].assert_called_once()
        assert main_presenter.sidebar is mock_ui_classes["sidebar"]

    def test_creates_chat_presenter(self, main_presenter, mock_ui_classes):
        mock_ui_classes["ChatPresenter"].assert_called_once()

    def test_creates_layout(self, main_presenter):
        assert main_presenter.layout is not None

    def test_runs_refresh_sidebar_on_init(self, main_presenter, page):
        page.run_task.assert_called_once()


class TestGetView:
    def test_returns_layout(self, main_presenter):
        assert main_presenter.get_view() is main_presenter.layout


class TestRefreshSidebar:
    @pytest.mark.asyncio
    async def test_updates_sidebar_with_chats(self, main_presenter):
        chats = [{"thread_id": "1", "title": "Chat 1"}]
        main_presenter.chat_manager.list_chats = AsyncMock(return_value=chats)

        await main_presenter.refresh_sidebar()

        main_presenter.sidebar.update_chats.assert_called_once_with(chats)


class TestCreateNewChat:
    @pytest.mark.asyncio
    async def test_creates_chat(self, main_presenter):
        main_presenter.chat_manager.create_chat = AsyncMock(return_value="new-id")

        await main_presenter.create_new_chat(None)

        main_presenter.chat_manager.create_chat.assert_called_once_with("Новый чат")

    @pytest.mark.asyncio
    async def test_loads_newly_created_chat(self, main_presenter):
        main_presenter.chat_manager.create_chat = AsyncMock(return_value="new-id")

        await main_presenter.create_new_chat(None)

        main_presenter.chat_ptr.load_chat.assert_called_once_with("new-id")


class TestDeleteChat:
    @pytest.mark.asyncio
    async def test_deletes_chat(self, main_presenter):
        main_presenter.chat_manager.delete_chat = AsyncMock()

        await main_presenter.delete_chat("thread-1")

        main_presenter.chat_manager.delete_chat.assert_called_once_with("thread-1")

    @pytest.mark.asyncio
    async def test_clears_history_when_current_chat_deleted(self, main_presenter):
        main_presenter.chat_manager.delete_chat = AsyncMock()
        main_presenter.chat_ptr.current_thread_id = "thread-1"

        await main_presenter.delete_chat("thread-1")

        assert main_presenter.chat_ptr.current_thread_id is None
        main_presenter.chat_ptr.history_view.clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_clear_history_for_different_chat(self, main_presenter):
        main_presenter.chat_manager.delete_chat = AsyncMock()
        main_presenter.chat_ptr.current_thread_id = "other-thread"

        await main_presenter.delete_chat("thread-1")

        assert main_presenter.chat_ptr.current_thread_id == "other-thread"


class TestRenameChat:
    @pytest.mark.asyncio
    async def test_renames_chat(self, main_presenter):
        main_presenter.chat_manager.rename_chat = AsyncMock()

        await main_presenter.rename_chat("thread-1", "New Title")

        main_presenter.chat_manager.rename_chat.assert_called_once_with("thread-1", "New Title")


class TestPickFilesClicked:
    @pytest.mark.asyncio
    async def test_disables_and_enables_button(self, main_presenter):
        main_presenter.file_picker.pick_files = AsyncMock(return_value=[])
        e = MagicMock()
        e.control = MagicMock()

        await main_presenter.pick_files_clicked(e)

        assert e.control.disabled is False
        assert main_presenter.page.update.call_count >= 2

    @pytest.mark.asyncio
    async def test_calls_pick_files(self, main_presenter):
        main_presenter.file_picker.pick_files = AsyncMock(return_value=[])
        e = MagicMock()
        e.control = MagicMock()

        await main_presenter.pick_files_clicked(e)

        main_presenter.file_picker.pick_files.assert_called_once_with(
            allow_multiple=True, allowed_extensions=["pdf"]
        )

    @pytest.mark.asyncio
    async def test_processes_returned_files(self, main_presenter):
        fake_file = MagicMock()
        fake_file.path = "C:\\test\\doc.pdf"
        main_presenter.file_picker.pick_files = AsyncMock(return_value=[fake_file])

        with patch.object(main_presenter, "_process_files") as mock_process:
            await main_presenter.pick_files_clicked(MagicMock())
            mock_process.assert_called_once_with([fake_file])

    @pytest.mark.asyncio
    async def test_skips_processing_when_no_files(self, main_presenter):
        main_presenter.file_picker.pick_files = AsyncMock(return_value=[])

        with patch.object(main_presenter, "_process_files") as mock_process:
            await main_presenter.pick_files_clicked(MagicMock())
            mock_process.assert_not_called()


class TestProcessFiles:
    @pytest.mark.asyncio
    async def test_copies_file_and_ingests(self, main_presenter):
        fake_file = MagicMock()
        fake_file.path = "C:\\source\\doc.pdf"

        with (
            patch("shutil.copy2") as mock_copy,
            patch.object(main_presenter, "refresh_files_list") as mock_refresh,
            patch("app.presenters.main_presenter.settings") as mock_settings,
            patch("app.presenters.main_presenter.VectorRepository") as MockRepo,
        ):
            mock_settings.pdf_docs_path = Path("C:\\docs")
            repo = MagicMock()
            repo.ingest_file = AsyncMock()
            MockRepo.return_value.__enter__.return_value = repo

            await main_presenter._process_files([fake_file])

        mock_copy.assert_called_once()
        src_arg, dst_arg = mock_copy.call_args[0]
        assert src_arg == "C:\\source\\doc.pdf"
        assert Path(dst_arg).name == "doc.pdf"
        repo.ingest_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_ingest_error(self, main_presenter):
        fake_file = MagicMock()
        fake_file.path = "C:\\source\\doc.pdf"

        with (
            patch("shutil.copy2"),
            patch.object(main_presenter, "refresh_files_list") as mock_refresh,
            patch("app.presenters.main_presenter.settings") as mock_settings,
            patch("app.presenters.main_presenter.VectorRepository") as MockRepo,
        ):
            mock_settings.pdf_docs_path = Path("C:\\docs")
            repo = MagicMock()
            repo.ingest_file = AsyncMock(side_effect=Exception("Index error"))
            MockRepo.return_value.__enter__.return_value = repo

            await main_presenter._process_files([fake_file])

        mock_refresh.assert_called_once()


class TestShowDocsDialog:
    @pytest.mark.asyncio
    async def test_opens_dialog_and_refreshes(self, main_presenter):
        with patch.object(main_presenter, "refresh_files_list") as mock_refresh:
            await main_presenter.show_docs_dialog(MagicMock())

        assert main_presenter.docs_dialog.open is True
        mock_refresh.assert_called_once()
