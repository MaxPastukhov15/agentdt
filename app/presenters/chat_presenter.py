import asyncio

import flet as ft
from core.agent import Agent
from core.chat_manager import ChatManager
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from ui.chat_display import ChatHistoryView
from ui.input_field import ChatInput
from utils.text_cleaner import clean_text

from presenters.app_model import AppModel
from utils.long_term_mem import load_chat_memory, save_chat_memory

class ChatPresenter:
    def __init__(self, model_core: Agent | AppModel, chat_manager: ChatManager):
        self.current_task: asyncio.Task | None = None
        self.model = model_core
        self.chat_manager = chat_manager
        self.current_thread_id = None

        self.history_view = ChatHistoryView()
        self.input_view = ChatInput(on_send=self.handle_send, on_cancel=self.cancel_generation)

        self.sources_list = ft.ListView(expand=True, spacing=10)
        self.sources_panel = ft.Container(
            width=320,
            bgcolor=ft.Colors.RED_100,
            border=ft.border.all(1, ft.Colors.RED_100),
            border_radius=ft.border_radius.only(top_left=20, bottom_left=20),
            padding=10,
            right=-350,
            top=0,
            bottom=100,
            visible=False,
            animate_position=ft.Animation(300, ft.AnimationCurve.DECELERATE),
            content=ft.Column(
                [
                    ft.ListTile(
                        title=ft.Text("Источники", weight=ft.FontWeight.BOLD, size=18),
                        trailing=ft.IconButton(ft.Icons.CLOSE_ROUNDED, on_click=self.hide_sources),
                        content_padding=0,
                    ),
                    ft.Divider(height=1, color=ft.Colors.OUTLINE),
                    self.sources_list,
                ],
                tight=True,
            ),
        )

        self.view = ft.Stack(
            controls=[
                ft.Container(content=self.history_view, expand=True),
                ft.Container(bgcolor=ft.Colors.SURFACE, height=100, bottom=0, left=0, right=0),
                ft.Container(content=self.input_view, bottom=0, left=0, right=0),
                self.sources_panel,
            ],
            expand=True,
        )

    def show_sources(self, links: list):
        if not links:
            return

        self.sources_list.controls.clear()

        for item in links:
            is_url = str(item).startswith(("http://", "https://"))

            icon_name = ft.Icons.LINK if is_url else ft.Icons.ARTICLE
            url_to_open = item

            async def on_click_handler(e, url=url_to_open):
                if is_url:
                    await e.page.launch_url(
                        url,
                    )

            tile = ft.ListTile(
                leading=ft.Icon(icon_name, size=16, color=ft.Colors.BLACK),
                title=ft.Text(item, size=12, selectable=True),
                on_click=on_click_handler if is_url else None,
                visual_density=ft.VisualDensity.COMPACT,
            )
            self.sources_list.controls.append(tile)

        self.sources_panel.visible = True
        self.sources_panel.right = 0
        self.sources_panel.update()

    def hide_sources(self, e=None):
        self.sources_panel.right = -350
        self.sources_panel.visible = False
        self.sources_panel.update()

    async def handle_send(self, text: str):
        if not self.current_thread_id:
            self.current_thread_id = await self.chat_manager.create_chat(text)

        text = clean_text(text=text)

        self.history_view.add_message(text, is_user=True)
        ai_msg = self.history_view.add_message("...", is_user=False, on_show_sources=self.show_sources)
        self.input_view.set_loading(True)

        self.current_task = asyncio.create_task(self._generate_response(text, ai_msg))

        try:
            await self.current_task

        except asyncio.CancelledError:
            ai_msg.update_text("Генерация прервана.")
            print("Задача генерации была принудительно отменена.")

        except Exception as e:
            ai_msg.update_text("Ошибка: Сервер долго не отвечает. Попробуйте еще раз.")
            print(f"Network error: {e}")
        finally:
            self.current_task = None
            self.input_view.set_loading(False)

    async def load_chat(self, thread_id: str):
        self.current_thread_id = thread_id
        self.history_view.clear_history()

        config = {"configurable": {"thread_id": thread_id}}
        state = await self.model.app.aget_state(config)

        if state.values and "messages" in state.values:
            for i, msg in enumerate(state.values["messages"]):
                if isinstance(msg, HumanMessage):
                    self.history_view.add_message(msg.content, is_user=True)
                elif isinstance(msg, AIMessage) and msg.content:
                    saved_links = msg.additional_kwargs.get("citation_links", [])
                    self.history_view.add_message(msg.content, is_user=False, links=saved_links, on_show_sources=self.show_sources)
            self.history_view.update()

    async def _generate_response(self, text: str, ai_msg):
        config = {"configurable": {"thread_id": self.current_thread_id}}
        last_links = []
        last_msg = None
        full_response = ""
        
        saved_summary = load_chat_memory(self.current_thread_id)
        current_summary = saved_summary

        ai_msg.update_status("Думаю...", visible=True)
        ai_msg.set_loading(True)

        try:
            initial_input = {
            "messages": [HumanMessage(content=text)],
            "summary": saved_summary}

            async for mode, payload in self.model.app.astream(
                initial_input,
                config=config,
                stream_mode=["values", "messages"],
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        full_response += content
                        ai_msg.update_text(full_response)

                elif mode == "values":
                    if payload.get("summary"):
                        current_summary = payload["summary"]

                    if payload.get("messages"):
                        last_msg = payload["messages"][-1]

                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                t_name = tool_call["name"]
                                t_args = tool_call["args"]

                                ai_msg.update_status(f"""Запуск инструмента {t_name}\nПараметры: {t_args}""", visible=True)

                        elif isinstance(last_msg, ToolMessage):
                            ai_msg.update_status("Данные получены. Анализирую...", visible=True)

                        elif isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content

                    if payload.get("citation_links"):
                        last_links = payload["citation_links"]

            if current_summary != saved_summary and self.current_thread_id is not None:
                print(f"DEBUG: Обнаружено новое summary, сохраняем для {self.current_thread_id}")
                save_chat_memory(self.current_thread_id, current_summary)

            if last_links and last_msg:
                ai_msg.set_loading(False)
                unique_links = list(dict.fromkeys(last_links))
                print(f"DEBUG: Отправляем ссылки в UI: {unique_links}")

                last_msg.additional_kwargs["citation_links"] = unique_links

                await self.model.app.aupdate_state(config, {"messages": [last_msg]})
                ai_msg.update_links(unique_links)
            ai_msg.update_status("", visible=False)

        except Exception as e:
            ai_msg.set_loading(False)
            print(f"LLM Error: {e}")
            error_display = f"\n\n[Ошибка: {e}]"
            ai_msg.update_text(full_response + error_display)

    def cancel_generation(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
