import flet as ft


class SidebarView(ft.Container):
    def __init__(self, on_new_chat, on_chat_selected, on_delete_chat, on_rename_chat):
        super().__init__()
        self.on_new_chat = on_new_chat
        self.on_chat_selected = on_chat_selected
        self.on_delete_chat = on_delete_chat
        self.on_rename_chat = on_rename_chat
        self.expanded: bool = True

        self.width = 400
        self.bgcolor = ft.Colors.SURFACE
        self.padding = 10
        self.animate = ft.Animation(300, ft.AnimationCurve.DECELERATE)

        self.chat_list = ft.Column(scroll=ft.ScrollMode.ADAPTIVE)
        self.chats_label = ft.Text("Chats", weight=ft.FontWeight.BOLD)
        self.new_chat_btn = ft.Button(
            on_click=self.on_new_chat,
            content=ft.Icon(ft.Icons.ADD),
            style=ft.ButtonStyle(shape=ft.CircleBorder(), padding=10),
        )

        self.content = ft.Column(
            controls=[
                ft.IconButton(ft.Icons.MENU, on_click=self.toggle_sidebar),
                ft.Divider(),
                self.new_chat_btn,
                self.chats_label,
                ft.Divider(),
                self.chat_list,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

    def toggle_sidebar(self, e):
        self.expanded = not self.expanded

        if self.expanded:
            self.width = 400
            self.chat_list.visible = True
            self.chats_label.visible = True

        else:
            self.width = 60
            self.chat_list.visible = False
            self.chats_label.visible = False

        self.update()

    def update_chats(self, chats: list):
        self.chat_list.controls.clear()
        for chat in chats:
            rename_btn = ft.IconButton(
                icon=ft.Icons.EDIT,
                icon_color=ft.Colors.BLUE_400,
                icon_size=16,
                tooltip="Переименовать чат",
                data={"thread_id": chat["thread_id"], "title": chat["title"]},
                on_click=self._on_rename_chat,
            )

            delete_btn = ft.IconButton(
                icon=ft.Icons.DELETE,
                icon_color=ft.Colors.RED_400,
                icon_size=16,
                tooltip="Удалить чат",
                data=chat["thread_id"],
                on_click=self._on_delete_click,
            )

            controls_row = ft.Row(
                [rename_btn, delete_btn],
                alignment=ft.MainAxisAlignment.END,
                spacing=0,
                tight=True,
            )

            self.chat_list.controls.append(
                ft.ListTile(
                    title=ft.Text(
                        chat["title"],
                        overflow=ft.TextOverflow.ELLIPSIS,
                        size=13,
                        no_wrap=True,
                    ),
                    subtitle=ft.Text(chat["updated_at"], size=9),
                    on_click=lambda e, tid=chat["thread_id"]: self.on_chat_selected(tid),
                    trailing=controls_row,
                    toggle_inputs=True,
                    dense=True,
                )
            )
        self.update()

    async def _on_delete_click(self, e):
        e.control.disabled = True
        self.update()

        thread_id = e.control.data
        if thread_id and self.on_delete_chat:
            await self.on_delete_chat(thread_id)

    async def _on_rename_chat(self, e):
        thread_id = e.control.data["thread_id"]
        current_title = e.control.data["title"]

        name_input = ft.TextField(label="Новое название", value=current_title, autofocus=True)

        async def save_clicked(e_save):
            if name_input.value:
                await self.on_rename_chat(thread_id, name_input.value)
            dialog.open = False
            self.page.update()

        def close_clicked(e_close):
            dialog.open = False
            self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Переименовать чат"),
            content=name_input,
            actions=[
                ft.TextButton("Отмена", on_click=close_clicked),
                ft.TextButton("Сохранить", on_click=save_clicked),
            ],
        )

        if self.page:
            self.page.overlay.append(dialog)
            dialog.open = True
            self.page.update()

    def _close_rename_dialog(self, e):
        if hasattr(self, "page") and self.page:
            self.page.dialog.open = False
            self.page.update()

    async def _save_rename(self, dialog, thread_id: str):
        new_title = dialog.content.value
        if new_title and self.on_rename_chat(thread_id, new_title):
            await self.on_rename_chat(thread_id, new_title)

        if hasattr(self, "page") and self.page:
            self.page.dialog.open = False
            self.page.update()
