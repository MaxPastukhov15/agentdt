import os

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Agent/1.0"


import logging
import multiprocessing

import flet as ft
import psutil
from config.config import settings
from core.agent import Agent
from core.chat_manager import ChatManager
from db.vectordb import VectorRepository
from presenters.main_presenter import AppModel, MainPresenter
from ui.login_page import LoginView
from utils.icon_path import get_resource

# import phoenix as px
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from phoenix.otel import register

main_logger = logging.getLogger()


async def main(page: ft.Page):
    page.window.prevent_close = True
    chat_manager = None

    async def handle_close(e):
        if e.type == ft.WindowEventType.CLOSE:
            main_logger.debug("Closing the app")
            if chat_manager is not None:
                await chat_manager.close()

            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

    page.window.on_event = handle_close
    icon_path = get_resource("assets/app_icon.ico")

    page.title = "Chembot"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.icon = icon_path

    async def start_main_app():
        nonlocal chat_manager
        page.controls.clear()

        chat_manager = ChatManager()
        await chat_manager.initialize()

        agent_instance = Agent(checkpointer=chat_manager.checkpointer)
        app_graph = agent_instance.make()

        model_core = AppModel(agent_instance, chat_manager, app_graph)
        presenter = MainPresenter(page, model_core, chat_manager)

        page.add(presenter.get_view())
        page.update()

    current_key = settings.openrouter_api_key.get_secret_value()

    if not current_key or current_key == "sk-or-v1-...":
        login_view = LoginView(on_login_success=start_main_app)
        page.add(login_view)
        page.update()
    else:
        await start_main_app()


if __name__ == "__main__":
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
