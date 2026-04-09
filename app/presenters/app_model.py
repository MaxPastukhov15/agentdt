from core.agent import Agent
from core.chat_manager import ChatManager


class AppModel:
    """
    Data Container (Model).
    Хранит ссылки на ключевые компоненты системы, чтобы не передавать
    их по отдельности в каждый презентер.
    """

    def __init__(self, agent: Agent, chat_manager: ChatManager, app_graph):
        self.agent = agent
        self.chat_manager = chat_manager
        self.app = app_graph
