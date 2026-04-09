import os

import pandas as pd
import phoenix as px
import pytest
from langchain_core.messages import HumanMessage
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.evals import OpenAIModel
from phoenix.otel import register

from app.config.config import settings
from app.core.agent import Agent

# ==========================================
# Глобальные фикстуры
# ==========================================


@pytest.fixture(scope="session")
def phoenix_app():
    """Запускает Phoenix один раз на всю сессию тестов"""
    print("\n🚀 Запуск Phoenix для тестов...")
    app = px.launch_app()
    yield app
    print("\n⏹️ Остановка Phoenix...")
    app.end()


@pytest.fixture(scope="session")
def tracer_provider(phoenix_app):
    """Настраивает OpenTelemetry для трассировки"""
    endpoint = settings.phoenix_collector_endpoint
    provider = register(endpoint=endpoint)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


@pytest.fixture(scope="function")
def agent_graph(tracer_provider):
    """Создает экземпляр агента для каждого теста"""
    return Agent().make()


@pytest.fixture(scope="function")
def test_session_id(request):
    """Генерирует уникальный thread_id для каждого теста"""
    return f"test_{request.node.name}_{os.getpid()}"


# ==========================================
# Фикстуры для оценки (Evals)
# ==========================================


@pytest.fixture(scope="session")
def judge_model():
    """Модель-судья для автоматической оценки"""
    return OpenAIModel(
        model="qwen/qwen3-4b:free",  # Используйте ID модели из OpenRouter
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key.get_secret_value(),
    )


# ==========================================
# Утилиты для тестов
# ==========================================


@pytest.fixture
def invoke_agent(agent_graph, test_session_id):
    """Хелпер для инвока агента с правильной конфигурацией"""

    async def _invoke(query: str, **kwargs):
        response = await agent_graph.ainvoke(
            {"messages": [HumanMessage(query)], "citation_links": None},
            config={
                "configurable": {"thread_id": test_session_id},
                "metadata": {"test_mode": True, **kwargs},
            },
        )

        if isinstance(response, dict) and "messages" in response:
            return response["messages"][-1].content
        return str(response)

    return _invoke


@pytest.fixture(scope="session")
def chemistry_golden_dataset():
    """Загружает химический датасет"""
    dataset_path = os.path.join(
        os.path.dirname(__file__), "datasets", "chemistry_questions.csv"
    )
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return pd.DataFrame()


@pytest.fixture
def chemistry_evaluator_config():
    """Конфигурация для оценки химических ответов"""
    return {
        "relevance_threshold": 0.75,
        "correctness_threshold": 0.7,
        "safety_refusal_required": True,
        "expected_terminology": ["моль", "реакция", "соединение", "ион", "электрон"],
    }
