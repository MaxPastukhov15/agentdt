import asyncio

import flet as ft


class ChatInput(ft.Container):
    def __init__(self, on_send, on_cancel):
        super().__init__(padding=ft.padding.symmetric(horizontal=20, vertical=5), alignment = ft.Alignment.BOTTOM_RIGHT,
            shadow = ft.BoxShadow(blur_radius=50, color=ft.Colors.SURFACE, offset=ft.Offset(0, -5)))

        self.on_send = on_send
        self.on_cancel = on_cancel

        self.send_button = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED,
            icon_color=ft.Colors.BLACK,
            bgcolor=ft.Colors.RED_50,
            on_click=self._on_submit,
        )

        self.text_field = ft.TextField(
            bgcolor=ft.Colors.WHITE,
            hint_text="Type...",
            expand=True,
            border_radius=20,
            content_padding=15,
            multiline=True,
            border_color=ft.Colors.RED_300,
            suffix=self.send_button,
        )

        self.is_active_loading = False
        self.content = ft.Row([self.text_field])

    async def _handle_click(self, e):
        if self.is_active_loading:
            if self.on_cancel:
                self.on_cancel()

        else:
            text = self.text_field.value.strip()
            if text and self.on_send:
                await self.on_send(text)

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
        self.is_active_loading = is_loading

        if is_loading:
            self.send_button.icon = ft.Icons.STOP_CIRCLE_SHARP
            self.send_button.icon_color = ft.Colors.RED_700
            self.text_field.read_only = True

        else:
            self.text_field.read_only = False
            self.send_button.icon = ft.Icons.SEND_ROUNDED
            self.send_button.icon_color = ft.Colors.BLACK
            self.text_field.disabled = False
            self.text_field.value = ""

        self.update()
