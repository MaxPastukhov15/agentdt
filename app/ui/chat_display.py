from typing import Optional

import flet as ft


class ChatMessage(ft.Column):
    def __init__(self, text: str, is_user: bool, links: Optional[list] = None):
        super().__init__()

        self.text_control = ft.Text(text, selectable=True, color=ft.Colors.BLACK)
        self.links_column = ft.Column(spacing=5, visible=False)

        self.controls = [
            ft.Container(
                content=self.text_control,
                bgcolor=ft.Colors.RED_50 if is_user else ft.Colors.SURFACE_BRIGHT,
                border_radius=15,
                padding=10,
                alignment=ft.Alignment.CENTER_RIGHT if is_user else ft.Alignment.CENTER_LEFT,
            )
        ]

        self.horizontal_alignment = ft.CrossAxisAlignment.END if is_user else ft.CrossAxisAlignment.START

        # if links:
        #    self.update_links(links)

    def update_links(self, links: list):
        if not links:
            return

        # Очищаем и наполняем блок ссылок
        self.links_column.controls = [
            ft.Divider(height=10, color=ft.Colors.GREY_700),
            ft.Text("Источники:", size=12, weight="bold", color=ft.Colors.GREY_400),
        ]

        for link in links:
            self.links_column.controls.append(ft.Text(f"🔗 {link}", size=11, color=ft.Colors.BLUE_300, italic=True))
        self.links_column.visible = True
        self.update()

    def update_text(self, text: str):
        self.text_control.value = text
        self.text_control.update()


class ChatHistoryView(ft.ListView):
    def __init__(self) -> None:
        super().__init__(
            expand=True,
            spacing=10,
            auto_scroll=True,
            padding=ft.padding.only(top=20, left=20, right=20, bottom=120),
        )

    def add_message(self, text: str, is_user: bool) -> ChatMessage:
        new_msg = ChatMessage(text, is_user)
        self.controls.append(new_msg)
        self.update()
        return new_msg

    def clear_history(self) -> None:
        self.controls.clear()
        self.update()
