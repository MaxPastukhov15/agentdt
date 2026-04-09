import flet as ft
from core.agent import Agent
from core.chat_manager import ChatManager
from langchain_core.messages import AIMessage, HumanMessage
from presenters.app_model import AppModel
from ui.chat_display import ChatHistoryView
from ui.input_field import ChatInput


class ChatPresenter:
    def __init__(self, model_core: Agent | AppModel, chat_manager: ChatManager):
        self.model = model_core
        self.chat_manager = chat_manager
        self.current_thread_id = None

        # Создаем View-компоненты чата
        self.history_view = ChatHistoryView()
        self.input_view = ChatInput(on_send=self.handle_send)
        self.view = ft.Stack(
            controls=[
                ft.Container(content=self.history_view, expand=True),
                ft.Container(
                    bgcolor=ft.Colors.SURFACE, height=100, bottom=0, left=0, right=0
                ),
                ft.Container(content=self.input_view, bottom=0, left=0, right=0),
            ],
            expand=True,
        )

    async def handle_send(self, text: str):
        if not self.current_thread_id:
            self.current_thread_id = await self.chat_manager.create_chat(text)

        self.history_view.add_message(text, is_user=True)
        ai_msg = self.history_view.add_message("...", is_user=False)
        self.input_view.set_loading(True)

        try:
            config = {"configurable": {"thread_id": self.current_thread_id}}
            last_links = []

            async for (
                chunk
            ) in self.model.app.astream(  # Используем скомпилированный app
                {"messages": [HumanMessage(content=text)]},
                config=config,
                stream_mode="values",
            ):
                # Обновляем текст ответа
                if chunk.get("messages"):
                    last_msg = chunk["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        ai_msg.update_text(last_msg.content)

                # Сохраняем ссылки из текущего состояния стейта
                if chunk.get("citation_links"):
                    last_links = chunk["citation_links"]

            # После завершения стрима выводим все найденные ссылки
            if last_links:
                ai_msg.update_links(last_links)

        finally:
            self.input_view.set_loading(False)

    async def load_chat(self, thread_id: str):
        self.current_thread_id = thread_id
        self.history_view.clear_history()

        config = {"configurable": {"thread_id": thread_id}}
        state = await self.model.app.aget_state(config)

        if state.values and "messages" in state.values:
            # links = state.values.get("citation_links", [])

            for i, msg in enumerate(state.values["messages"]):
                is_last = i == len(state.values["messages"]) - 1
                if isinstance(msg, HumanMessage):
                    self.history_view.add_message(msg.content, is_user=True)
                elif isinstance(msg, AIMessage) and msg.content:
                    self.history_view.add_message(
                        msg.content,
                        is_user=False,
                    )
