import flet as ft
import os
import asyncio
import shutil
from db.vectordb import VectorRepository
from core.agent import Agent
from core.chat_manager import ChatManager
from presenters.app_model import AppModel
from presenters.chat_presenter import ChatPresenter
from ui.sidebar import SidebarView
from pathlib import Path
from config.config import settings



class MainPresenter:
    def __init__(self, page: ft.Page, model_core: Agent | AppModel, chat_manager: ChatManager):
        self.page = page
        self.model_core = model_core
        self.chat_manager = chat_manager
        self.file_picker = ft.FilePicker()

        self.file_list_column = ft.Column(spacing=5, scroll=ft.ScrollMode.ALWAYS, height=300)

        self.docs_dialog = ft.AlertDialog(
            title=ft.Text("Управление документами"),
            content=ft.Container(
                width=400,
                content=ft.Column([
                    ft.Text("Загруженные PDF:"),
                    self.file_list_column,
                    ft.ElevatedButton(
                        "Выбрать файлы", 
                        icon=ft.Icons.ADD, 
                        on_click=self.pick_files_clicked
                    )
                ], tight=True)
            )
        )
        self.page.overlay.append(self.docs_dialog)

        self.chat_ptr = ChatPresenter(self.model_core, self.chat_manager)

        self.sidebar = SidebarView(
            on_new_chat=self.create_new_chat,
            on_chat_selected=lambda tid: self.page.run_task(self.chat_ptr.load_chat, tid),
            on_delete_chat=self.delete_chat,
            on_rename_chat=self.rename_chat,
            on_manage_docs=self.show_docs_dialog
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
    
    async def pick_files_clicked(self, e):
        """Прямой вызов без колбэков"""
        e.control.disabled = True
        self.page.update()

        result = await self.file_picker.pick_files(
            allow_multiple=True, 
            allowed_extensions=["pdf"]
        )
        files = result if isinstance(result, list) else []
        
        if files:
            await self._process_files(files)

        e.control.disabled = False
        self.page.update()
    
    async def _process_files(self, files):
        """Логика обработки файлов"""
        for file in files:
            if not file.path: continue
            dst = settings.pdf_docs_path / Path(file.path).name
            
            await asyncio.to_thread(shutil.copy2, file.path, dst)
            try:
                with VectorRepository("chemistry_collection") as repo:
                    await repo.ingest_file(dst)
            except Exception as ex:
                print(f"Ошибка индексации: {ex}")

        await self.refresh_files_list()

    async def refresh_files_list(self):
        """Обновление только содержимого колонки"""
        if not settings.pdf_docs_path.exists():
            settings.pdf_docs_path.mkdir(parents=True)
            
        files = [f for f in os.listdir(settings.pdf_docs_path) if f.endswith('.pdf')]
        
        self.file_list_column.controls = [
            ft.ListTile(
                leading=ft.Icon(ft.Icons.PICTURE_AS_PDF, color=ft.Colors.RED_400),
                title=ft.Text(f, size=12),
                trailing=ft.IconButton(
                    ft.Icons.DELETE_OUTLINE, 
                    on_click=lambda _, fn=f: self.page.run_task(self.delete_file, fn)
                )
            ) for f in files
        ]
        
        if not files:
            self.file_list_column.controls = [ft.Text("Пусто", italic=True)]
            
        self.page.update()

    async def delete_file(self, filename):
        file_path = settings.pdf_docs_path / filename
        if file_path.exists():
            os.remove(file_path)
        await self.refresh_files_list()

    async def show_docs_dialog(self, e):
        """Метод для кнопки в Sidebar"""
        self.docs_dialog.open = True
        await self.refresh_files_list()