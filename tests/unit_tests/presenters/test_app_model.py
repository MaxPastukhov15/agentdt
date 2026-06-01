from unittest.mock import MagicMock

from app.presenters.app_model import AppModel


class TestAppModel:
    def test_init_stores_references(self):
        agent = MagicMock()
        chat_manager = MagicMock()
        app_graph = MagicMock()
        model = AppModel(agent, chat_manager, app_graph)

        assert model.agent is agent
        assert model.chat_manager is chat_manager
        assert model.app is app_graph

    def test_init_accepts_different_types(self):
        model = AppModel("agent_str", 42, None)
        assert model.agent == "agent_str"
        assert model.chat_manager == 42
        assert model.app is None
