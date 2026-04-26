from typing import Optional

import flet as ft


class ChatMessage(ft.Column):
    def __init__(self, text: str, is_user: bool, links: Optional[list] = None, on_show_sources=None):
        super().__init__()

        self.saved_links = links or []
        self.on_show_sources = on_show_sources

        self.text_control = ft.Markdown(
            value=text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme=ft.MarkdownCodeTheme.GITHUB,
        )

        self.sources_btn = ft.IconButton(
            icon=ft.Icons.LIBRARY_BOOKS_OUTLINED, tooltip="Показать источники", icon_size=14, visible=len(self.saved_links) > 0, on_click=self._handle_show_sources
        )

        self.switcher = ft.AnimatedSwitcher(
            content=self.text_control,
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=400,
            reverse_duration=200,
            switch_in_curve=ft.AnimationCurve.EASE_OUT,
        )

        self.controls = [
            ft.Container(
                content=ft.Column([self.switcher, ft.Row([self.sources_btn], alignment=ft.MainAxisAlignment.END) if not is_user else ft.Container()], spacing=0),
                bgcolor=ft.Colors.RED_50 if is_user else ft.Colors.SURFACE_BRIGHT,
                border_radius=15,
                padding=10,
                alignment=(ft.Alignment.CENTER_RIGHT if is_user else ft.Alignment.CENTER_LEFT),
            ),
        ]

        self.horizontal_alignment = ft.CrossAxisAlignment.END if is_user else ft.CrossAxisAlignment.START

    def _handle_show_sources(self, e):
        if self.on_show_sources and self.saved_links:
            self.on_show_sources(self.saved_links)

    def update_links(self, links: list):
        try:
            self.saved_links = links
            if self.saved_links:
                self.sources_btn.visible = True
                if self.page:
                    self.sources_btn.update()
                    self.update()

        except Exception as e:
            print(f"Error updating links: {e}")

    def update_text(self, text: str):
        try:
            self.text_control.value = text
            if self.page:
                self.switcher.content = ft.Markdown(
                    value=text,
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    code_theme=ft.MarkdownCodeTheme.GITHUB,
                )
                self.switcher.update()
        except Exception as e:
            print(f"Error updating text: {e}")


class ChatHistoryView(ft.ListView):
    def __init__(self) -> None:
        super().__init__(
            expand=True,
            spacing=10,
            auto_scroll=True,
            padding=ft.padding.only(top=20, left=20, right=20, bottom=120),
        )

    def add_message(self, text: str, is_user: bool, links: Optional[list] = None, on_show_sources=None) -> ChatMessage:
        new_msg = ChatMessage(text, is_user, links=links, on_show_sources=on_show_sources)
        self.controls.append(new_msg)
        self.update()
        return new_msg

    def remove_message(self, text: ChatMessage):
        if text in self.controls:
            self.controls.remove(text)
            self.update()

    def clear_history(self) -> None:
        self.controls.clear()
        self.update()
