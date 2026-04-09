import flet as ft
from core.agent import Agent
from core.chat_manager import ChatManager
from presenters.app_model import AppModel
from presenters.chat_presenter import ChatPresenter
from ui.sidebar import SidebarView


class MainPresenter:
    def __init__(
        self, page: ft.Page, model_core: Agent | AppModel, chat_manager: ChatManager
    ):
        self.page = page
        self.model_core = model_core
        self.chat_manager = chat_manager

        self.chat_ptr = ChatPresenter(self.model_core, self.chat_manager)

        self.sidebar = SidebarView(
            on_new_chat=self.create_new_chat,
            on_chat_selected=lambda tid: self.page.run_task(
                self.chat_ptr.load_chat, tid
            ),
            on_delete_chat=self.delete_chat,
            on_rename_chat=self.rename_chat,
        )

        self.layout = ft.Row(
            [
                self.sidebar,
                ft.VerticalDivider(width=1),
                ft.Container(content=self.chat_ptr.view, expand=True, padding=10),
            ],
            expand=True,
        )

        self.page.run_task(self.refresh_sidebar)

    async def refresh_sidebar(self):
        chats = await self.chat_manager.list_chats()
        self.sidebar.update_chats(chats)

    async def create_new_chat(self, e):
        new_id = await self.chat_manager.create_chat("Новый чат")
        await self.refresh_sidebar()
        await self.chat_ptr.load_chat(new_id)

    async def delete_chat(self, thread_id: str):
        await self.chat_manager.delete_chat(thread_id)

        if self.chat_ptr.current_thread_id == thread_id:
            self.chat_ptr.current_thread_id = None
            self.chat_ptr.history_view.clear_history()

        await self.refresh_sidebar()

    async def rename_chat(self, thread_id: str, new_title: str):
        await self.chat_manager.rename_chat(thread_id, new_title)
        await self.refresh_sidebar()

    def get_view(self):
        return self.layout
