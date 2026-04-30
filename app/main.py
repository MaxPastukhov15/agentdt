import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Agent/1.0"

import config.logger_config

import multiprocessing

import flet as ft
import psutil
from config.config import settings
from core.agent import Agent
from core.chat_manager import ChatManager
from db.vectordb import VectorRepository
from presenters.main_presenter import AppModel, MainPresenter
from ui.login_page import LoginView

# import phoenix as px
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from phoenix.otel import register


async def main(page: ft.Page):
    page.window.prevent_close = True
    chat_manager = None

    async def handle_close(e):
        if e.type == ft.WindowEventType.CLOSE:
            print("Сигнал закрытия получен. Мгновенное завершение...")
            if chat_manager is not None:
                await chat_manager.close()

            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            print("Приложение закрыто")

    page.window.on_event = handle_close

    # --- 1. Настройки окна ---
    icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "app_icon.png")

    page.title = "AI Desktop Agent"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.icon = icon_path

    # --- 2. Инициализация Core-слоя ---
    async def start_main_app():
        nonlocal chat_manager
        page.controls.clear()
        # Менеджер чатов создает БД и инициализирует SqliteSaver (checkpointer)
        chat_manager = ChatManager()
        await chat_manager.initialize()

        # Агент создается с использованием чекпоинтера из менеджера
        agent_instance = Agent(checkpointer=chat_manager.checkpointer)
        # Компилируем граф один раз при старте
        app_graph = agent_instance.make()

        # Собираем модель для MVP
        model_core = AppModel(agent_instance, chat_manager, app_graph)

        # --- 3. Запуск UI через Презентер ---
        # MainPresenter возьмет на себя создание всех вьюх и дочерних презентеров
        presenter = MainPresenter(page, model_core, chat_manager)

        page.add(presenter.get_view())
        page.update()

    current_key = settings.openrouter_api_key.get_secret_value()

    if not current_key or current_key == "sk-or-v1-...":
        # Если ключа нет, рисуем экран авторизации
        login_view = LoginView(on_login_success=start_main_app)
        page.add(login_view)
        page.update()
    else:
        # Если ключ есть, сразу грузим чат
        await start_main_app()


if __name__ == "__main__":
    # Запуск приложения
    try:
        # tracer_provider = register(endpoint=settings.phoenix_collector_endpoint)
        # LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        # session = px.launch_app()

        with VectorRepository("chemistry_collection") as repo:
            repo.startup_db()
        multiprocessing.freeze_support()
        ft.run(main=main)

    except KeyboardInterrupt:
        os._exit(0)
    # finally:
    #    if session:
    #        session.end()
