import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.presenters.chat_presenter import ChatPresenter


@pytest.fixture
def chat_presenter(mock_ui_classes, mock_model_and_chat_manager):
    agent, chat_manager = mock_model_and_chat_manager
    presenter = ChatPresenter(agent, chat_manager)
    return presenter


class TestChatPresenterInit:
    def test_creates_history_view(self, chat_presenter, mock_ui_classes):
        mock_ui_classes["ChatHistoryView"].assert_called_once()
        assert chat_presenter.history_view is mock_ui_classes["history_view"]

    def test_creates_input_view(self, chat_presenter, mock_ui_classes):
        mock_ui_classes["ChatInput"].assert_called_once_with(
            on_send=chat_presenter.handle_send,
            on_cancel=chat_presenter.cancel_generation,
        )
        assert chat_presenter.input_view is mock_ui_classes["input_view"]

    def test_stores_agent_and_chat_manager(self, chat_presenter, mock_model_and_chat_manager):
        agent, chat_manager = mock_model_and_chat_manager
        assert chat_presenter.model is agent
        assert chat_presenter.chat_manager is chat_manager

    def test_current_thread_id_is_none(self, chat_presenter):
        assert chat_presenter.current_thread_id is None

    def test_sources_panel_exists(self, chat_presenter):
        assert chat_presenter.sources_panel is not None


class TestShowSources:
    def test_shows_sources_with_links(self, chat_presenter):
        with (
            patch.object(chat_presenter.sources_list, "controls", []),
            patch.object(chat_presenter.sources_panel, "update"),
        ):
            chat_presenter.show_sources(["link1", "link2"])
            assert chat_presenter.sources_panel.visible is True
            assert chat_presenter.sources_panel.right == 0

    def test_does_not_show_empty_links(self, chat_presenter):
        with (
            patch.object(chat_presenter.sources_panel, "update") as mock_update,
            patch.object(chat_presenter.sources_list, "controls", []),
        ):
            chat_presenter.show_sources([])
            mock_update.assert_not_called()

    def test_does_not_show_none_links(self, chat_presenter):
        with (
            patch.object(chat_presenter.sources_panel, "update") as mock_update,
            patch.object(chat_presenter.sources_list, "controls", []),
        ):
            chat_presenter.show_sources(None)
            mock_update.assert_not_called()

    def test_creates_tiles_for_links(self, chat_presenter):
        with (
            patch.object(chat_presenter.sources_list, "controls", []),
            patch.object(chat_presenter.sources_panel, "update"),
        ):
            chat_presenter.show_sources(["http://example.com"])
            assert len(chat_presenter.sources_list.controls) == 1


class TestHideSources:
    def test_hides_panel(self, chat_presenter):
        chat_presenter.sources_panel.visible = True
        chat_presenter.sources_panel.right = 0
        with patch.object(chat_presenter.sources_panel, "update"):
            chat_presenter.hide_sources()
            assert chat_presenter.sources_panel.visible is False
            assert chat_presenter.sources_panel.right == -350


class TestCancelGeneration:
    def test_cancels_current_task(self, chat_presenter):
        task = MagicMock()
        task.done.return_value = False
        chat_presenter.current_task = task

        chat_presenter.cancel_generation()

        task.cancel.assert_called_once()

    def test_noop_when_no_task(self, chat_presenter):
        chat_presenter.current_task = None
        chat_presenter.cancel_generation()

    def test_noop_when_task_already_done(self, chat_presenter):
        task = MagicMock()
        task.done.return_value = True
        chat_presenter.current_task = task
        chat_presenter.cancel_generation()
        task.cancel.assert_not_called()


class TestHandleSend:
    @pytest.mark.asyncio
    async def test_creates_chat_when_no_thread_id(self, chat_presenter):
        chat_presenter.chat_manager.create_chat = AsyncMock(return_value="new-thread-id")

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task", return_value=asyncio.Future()),
        ):
            mock_add.side_effect = [MagicMock(), MagicMock()]
            fut = asyncio.Future()
            fut.set_result(None)
            asyncio.create_task.return_value = fut

            await chat_presenter.handle_send("user text")

        chat_presenter.chat_manager.create_chat.assert_called_once_with("user text")
        assert chat_presenter.current_thread_id == "new-thread-id"

    @pytest.mark.asyncio
    async def test_reuses_existing_thread_id(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"
        chat_presenter.chat_manager.create_chat = AsyncMock()

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            mock_add.side_effect = [MagicMock(), MagicMock()]
            fut = asyncio.Future()
            fut.set_result(None)
            mock_task.return_value = fut

            await chat_presenter.handle_send("user text")

        chat_presenter.chat_manager.create_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_text(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned") as mock_clean,
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            mock_add.side_effect = [MagicMock(), MagicMock()]
            fut = asyncio.Future()
            fut.set_result(None)
            mock_task.return_value = fut

            await chat_presenter.handle_send("raw text")

        mock_clean.assert_called_once_with(text="raw text")

    @pytest.mark.asyncio
    async def test_adds_user_and_ai_messages(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            mock_add.side_effect = [MagicMock(), MagicMock()]
            fut = asyncio.Future()
            fut.set_result(None)
            mock_task.return_value = fut

            await chat_presenter.handle_send("hello")

            assert mock_add.call_count == 2
            mock_add.assert_any_call("cleaned", is_user=True)
            mock_add.assert_any_call("...", is_user=False, on_show_sources=chat_presenter.show_sources)

    @pytest.mark.asyncio
    async def test_sets_loading_during_generation(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            mock_add.side_effect = [MagicMock(), MagicMock()]
            fut = asyncio.Future()
            fut.set_result(None)
            mock_task.return_value = fut

            await chat_presenter.handle_send("hello")

        chat_presenter.input_view.set_loading.assert_any_call(True)
        chat_presenter.input_view.set_loading.assert_any_call(False)

    @pytest.mark.asyncio
    async def test_handles_cancelled_error(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            ai_msg = MagicMock()
            mock_add.side_effect = [MagicMock(), ai_msg]
            fut = asyncio.Future()
            fut.set_exception(asyncio.CancelledError())
            mock_task.return_value = fut

            await chat_presenter.handle_send("hello")

        ai_msg.update_text.assert_called_once_with("Генерация прервана.")

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, chat_presenter):
        chat_presenter.current_thread_id = "existing-id"

        with (
            patch("app.presenters.chat_presenter.clean_text", return_value="cleaned"),
            patch.object(chat_presenter.history_view, "add_message") as mock_add,
            patch("asyncio.create_task") as mock_task,
        ):
            ai_msg = MagicMock()
            mock_add.side_effect = [MagicMock(), ai_msg]
            fut = asyncio.Future()
            fut.set_exception(Exception("Network error"))
            mock_task.return_value = fut

            await chat_presenter.handle_send("hello")

        ai_msg.update_text.assert_called_once_with("Ошибка: Сервер долго не отвечает. Попробуйте еще раз.")


class TestLoadChat:
    @pytest.mark.asyncio
    async def test_loads_chat_sets_thread_id(self, chat_presenter):
        state = MagicMock()
        state.values = {}
        chat_presenter.model.app.aget_state = AsyncMock(return_value=state)

        await chat_presenter.load_chat("thread-1")

        assert chat_presenter.current_thread_id == "thread-1"
        chat_presenter.history_view.clear_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_displays_human_messages(self, chat_presenter):
        state = MagicMock()
        state.values = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        }
        chat_presenter.model.app.aget_state = AsyncMock(return_value=state)

        await chat_presenter.load_chat("thread-1")

        chat_presenter.history_view.add_message.assert_any_call("Hello", is_user=True)
        chat_presenter.history_view.add_message.assert_any_call(
            "Hi there!", is_user=False, links=[], on_show_sources=chat_presenter.show_sources
        )

    @pytest.mark.asyncio
    async def test_displays_ai_message_with_links(self, chat_presenter):
        msg = AIMessage(content="Response with sources")
        msg.additional_kwargs["citation_links"] = ["src1", "src2"]

        state = MagicMock()
        state.values = {"messages": [HumanMessage(content="Q"), msg]}
        chat_presenter.model.app.aget_state = AsyncMock(return_value=state)

        await chat_presenter.load_chat("thread-1")

        chat_presenter.history_view.add_message.assert_any_call(
            "Response with sources",
            is_user=False,
            links=["src1", "src2"],
            on_show_sources=chat_presenter.show_sources,
        )

    @pytest.mark.asyncio
    async def test_skips_empty_ai_messages(self, chat_presenter):
        state = MagicMock()
        state.values = {
            "messages": [
                HumanMessage(content="Q"),
                AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
                AIMessage(content="Final"),
            ]
        }
        chat_presenter.model.app.aget_state = AsyncMock(return_value=state)

        await chat_presenter.load_chat("thread-1")

        calls = chat_presenter.history_view.add_message.call_args_list
        displayed_contents = [c[1]["text"] if "text" in c[1] else c[0][0] for c in calls]
        assert "Q" in displayed_contents
        assert "Final" in displayed_contents

    @pytest.mark.asyncio
    async def test_empty_state(self, chat_presenter):
        state = MagicMock()
        state.values = None
        chat_presenter.model.app.aget_state = AsyncMock(return_value=state)

        await chat_presenter.load_chat("thread-1")

        assert chat_presenter.current_thread_id == "thread-1"
        chat_presenter.history_view.add_message.assert_not_called()


class TestGenerateResponse:
    @pytest.mark.asyncio
    async def test_streams_values_and_messages(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {"messages": [AIMessage(content="Final answer")], "citation_links": ["cit1"]}
            yield "messages", (AIMessage(content="Final answer"), {})

        chat_presenter.model.app.astream = mock_astream

        with patch("app.presenters.chat_presenter.load_chat_memory", return_value=""):
            await chat_presenter._generate_response("hello", ai_msg)

        ai_msg.set_loading.assert_called()
        ai_msg.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_saves_summary_when_changed(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {
                "messages": [AIMessage(content="Final")],
                "citation_links": [],
                "summary": "new summary",
            }

        chat_presenter.model.app.astream = mock_astream

        with (
            patch("app.presenters.chat_presenter.load_chat_memory", return_value="old summary") as mock_load,
            patch("app.presenters.chat_presenter.save_chat_memory") as mock_save,
        ):
            await chat_presenter._generate_response("hello", ai_msg)

        mock_save.assert_called_once_with("test-thread", "new summary")

    @pytest.mark.asyncio
    async def test_does_not_save_summary_when_unchanged(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {
                "messages": [AIMessage(content="Final")],
                "citation_links": [],
                "summary": "",
            }

        chat_presenter.model.app.astream = mock_astream

        with (
            patch("app.presenters.chat_presenter.load_chat_memory", return_value="") as mock_load,
            patch("app.presenters.chat_presenter.save_chat_memory") as mock_save,
        ):
            await chat_presenter._generate_response("hello", ai_msg)

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_links_on_ai_message(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        last_ai = AIMessage(content="Final")
        last_ai.additional_kwargs = {}

        async def mock_astream(*args, **kwargs):
            yield "values", {
                "messages": [last_ai],
                "citation_links": ["cit1", "cit2"],
            }

        chat_presenter.model.app.astream = mock_astream
        chat_presenter.model.app.aupdate_state = AsyncMock()

        with patch("app.presenters.chat_presenter.load_chat_memory", return_value=""):
            await chat_presenter._generate_response("hello", ai_msg)

        assert last_ai.additional_kwargs["citation_links"] == ["cit1", "cit2"]

    @pytest.mark.asyncio
    async def test_no_links_when_no_citations(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "values", {
                "messages": [AIMessage(content="Final")],
                "citation_links": [],
            }

        chat_presenter.model.app.astream = mock_astream

        with patch("app.presenters.chat_presenter.load_chat_memory", return_value=""):
            await chat_presenter._generate_response("hello", ai_msg)

        ai_msg.update_links.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_stream_error(self, chat_presenter):
        chat_presenter.current_thread_id = "test-thread"
        ai_msg = MagicMock()

        async def mock_astream(*args, **kwargs):
            raise Exception("Stream failed")
            yield

        chat_presenter.model.app.astream = mock_astream

        with patch("app.presenters.chat_presenter.load_chat_memory", return_value=""):
            await chat_presenter._generate_response("hello", ai_msg)

        ai_msg.set_loading.assert_called_with(False)
