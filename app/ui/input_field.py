import asyncio

import flet as ft


class ChatInput(ft.Container):
    def __init__(self, on_send):
        super().__init__()
        self.on_send = on_send
        self.padding = ft.padding.symmetric(horizontal=20, vertical=5)
        self.alignment = ft.Alignment.BOTTOM_RIGHT
        self.shadow = ft.BoxShadow(
            blur_radius=50, color=ft.Colors.SURFACE, offset=ft.Offset(0, -5)
        )

        self.text_field = ft.TextField(
            bgcolor=ft.Colors.WHITE,
            hint_text="Type...",
            expand=True,
            border_radius=20,
            content_padding=15,
            multiline=True,
            border_color=ft.Colors.RED_300,
            suffix=ft.IconButton(
                icon=ft.Icons.SEND_ROUNDED,
                icon_color=ft.Colors.BLACK,
                bgcolor=ft.Colors.RED_50,
                on_click=self._on_submit,
            ),
        )

        self.content = ft.Row([self.text_field])

    async def _on_submit(self, e):
        text = self.text_field.value
        if text and self.on_send:
            if asyncio.iscoroutinefunction(self.on_send):
                await self.on_send(text)
            else:
                self.on_send(text)
        self.text_field.value = ""
        self.update()

    def set_loading(self, is_loading: bool):
        self.text_field.disabled = is_loading
        self.text_field.update()
