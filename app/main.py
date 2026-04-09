"""from core.agent import Agent
from langchain_core.messages import HumanMessage
from asyncio import run
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
import asyncio
from config.config import settings
async def main():
    session = px.launch_app()

    try:
        tracer_provider = register(endpoint=settings.phoenix_collector_endpoint)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        graph = Agent().make()

        while True:
            query = await asyncio.to_thread(input, "Вопрос: ")

            if not query.strip():
                break

            answer = await graph.ainvoke({"messages" : [HumanMessage(query)], "citation_links" : None},
                          config={"configurable": {"thread_id": "haha12020"}})
    finally:
        session.end()
"""

import os

import flet as ft
from core.agent import Agent
from core.chat_manager import ChatManager
from presenters.main_presenter import AppModel, MainPresenter


async def main(page: ft.Page):
    # --- 1. Настройки окна ---
    icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "app_icon.png")

    page.title = "AI Desktop Agent"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.icon = icon_path
    # --- 2. Инициализация Core-слоя ---
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

    # Добавляем главный лейаут на страницу
    page.add(presenter.get_view())
    page.update()


if __name__ == "__main__":
    # Запуск приложения
    try:
        ft.run(main=main)
    except KeyboardInterrupt:
        pass
