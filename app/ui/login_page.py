import os

import flet as ft
from config.config import settings
from dotenv import set_key
from pydantic import SecretStr


class LoginView(ft.Container):
    def __init__(self, on_login_success):
        super().__init__()
        self.on_login_success = on_login_success
        self.expand = True
        self.alignment = ft.alignment.Alignment.CENTER
        self.bgcolor = ft.Colors.WHITE_10

        self.key_input = ft.TextField(
            label="OpenRouter API Key", password=True, can_reveal_password=True, width=400, tooltip="Ключ будет сохранен в файле .env", on_submit=self.handle_login
        )

        self.button = ft.ElevatedButton("Сохранить и войти", on_click=self.handle_login, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)))

        self.content = ft.Card(
            content=ft.Container(
                padding=40,
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.LOCK_OUTLINED, size=50, color=ft.Colors.RED),
                        ft.Text("Требуется авторизация", size=24, weight=ft.FontWeight.BOLD),
                        ft.Text("Введите ваш API ключ OpenRouter для начала работы(https://openrouter.ai/)", text_align=ft.TextAlign.CENTER),
                        self.key_input,
                        self.button,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                    tight=True,
                ),
            )
        )

    async def handle_login(self, e):
        self.button.disabled = True
        api_key = self.key_input.value.strip()
        if not api_key or api_key == "sk-or-v1-...":
            self.key_input.error = "Введите корректный ключ"
            self.update()
            return

        env_path = settings.env_path

        try:
            set_key(env_path, "OPENROUTER_API_KEY", api_key)

            set_key(env_path, "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
            set_key(env_path, "USER_AGENT", "DesktopAgentCH")

            set_key(env_path, "MAIN_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
            set_key(env_path, "SUMMARIZATION_MODEL", "baidu/qianfan-ocr-fast:free")
            set_key(env_path, "EMBEDDING_MODEL", "BAAI/bge-m3")

            set_key(env_path, "HF_HUB_OFFLINE", "1")

            set_key(env_path, "TRANSFORMERS_OFFLINE", "1")

            set_key(env_path, "MAX_STEPS", "7")

            settings.openrouter_api_key = SecretStr(api_key)
            os.environ["OPENROUTER_API_KEY"] = api_key

            await self.on_login_success()

        except Exception as ex:
            self.key_input.error = f"Ошибка сохранения: {ex}"
            self.update()

        finally:
            self.button.disabled = False
