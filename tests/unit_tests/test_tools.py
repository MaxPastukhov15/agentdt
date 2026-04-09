import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.tools import (
    create_summary,
    search_arxiv,
    search_chemistry_collection,
    search_duckduckgo,
)


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Автоматически мокать переменные окружения для всех тестов"""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_chain_ainvoke():
    """Фикстура для мока вызова LLM-цепочки"""
    with patch("core.tools.chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value="Mocked summary text")
        yield mock_chain


@pytest.fixture
def mock_vector_repo():
    """Фикстура для мока VectorRepository"""
    with patch("core.tools.VectorRepository") as MockRepo:
        mock_instance = MagicMock()
        mock_retriever = AsyncMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.get_retriever.return_value = mock_retriever
        MockRepo.return_value = mock_instance
        yield mock_instance, mock_retriever


@pytest.mark.asyncio
async def test_search_chemistry_collection_success(
    mock_vector_repo, mock_chain_ainvoke
):
    """Успешный поиск: проверяем форматирование контекста и цитат"""
    mock_instance, mock_retriever = mock_vector_repo

    # Подготовка тестовых данных
    test_docs = [
        Document(
            page_content="Water is H2O",
            metadata={"source": "chemistry_basics.pdf", "page": 42},
        )
    ]
    mock_retriever.ainvoke.return_value = test_docs

    result = await search_chemistry_collection.ainvoke("что такое вода")

    # Проверка структуры ответа
    assert "context" in result
    assert "citations" in result
    assert "(chemistry_basics.pdf, 42): Water is H2O" in result["context"]
    assert "chemistry_basics.pdf, 42" in result["citations"]


@pytest.mark.asyncio
async def test_search_chemistry_collection_empty_results(mock_vector_repo):
    """Поиск без результатов — корректная обработка"""
    mock_instance, mock_retriever = mock_vector_repo
    mock_retriever.ainvoke.return_value = []

    result = await search_chemistry_collection.ainvoke("несуществующий термин")

    assert result["context"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_search_chemistry_collection_exception(mock_vector_repo):
    """Обработка исключения при работе с репозиторием"""
    mock_instance, mock_retriever = mock_vector_repo
    mock_retriever.ainvoke.side_effect = ConnectionError("DB unavailable")

    with pytest.raises(ConnectionError):
        await search_chemistry_collection.ainvoke("тест")


@pytest.mark.asyncio
async def test_create_summary_metadata_combinations(mock_chain_ainvoke):
    """Тестируем логику форматирования вывода при разных комбинациях метаданных"""

    test_cases = [
        # (metadata, expected_title_pattern, expected_in_detail)
        ({"title": "Test Paper"}, "Test Paper", "Title: Test Paper"),
        ({"title": "Paper", "authors": "Smith"}, "Paper, Smith", "Author: Smith"),
        (
            {"title": "Paper", "link": "http://example.com"},
            "Paper, http://example.com",
            "Title: Paper",
        ),
        ({}, "N/A", "Title: N/A"),
    ]

    for metadata, expected_title_part, expected_detail_part in test_cases:
        doc = Document(page_content="Test content", metadata=metadata)
        title, detail = await create_summary("query", doc)

        assert expected_title_part in title
        assert expected_detail_part in detail
        assert "Summary:" in detail


@pytest.mark.asyncio
async def test_search_arxiv_success(mock_chain_ainvoke):
    """Успешный поиск в arXiv"""

    with patch("core.tools.ArxivRetriever") as MockRetriever:
        mock_retriever_instance = MagicMock()
        MockRetriever.return_value = mock_retriever_instance

        # Mock для asyncio.to_thread
        test_docs = [
            Document(
                page_content="DFT study",
                metadata={"title": "DFT Paper", "authors": "Lee", "link": "arxiv:1234"},
            )
        ]
        mock_retriever_instance.invoke.return_value = test_docs

        result = await search_arxiv.ainvoke("density functional theory")

        assert "context" in result
        assert "citations" in result
        assert "DFT Paper, Lee" in str(result["citations"])


@pytest.mark.asyncio
async def test_search_arxiv_no_results(mock_chain_ainvoke):
    """Пустой ответ от arXiv"""

    with patch("core.tools.ArxivRetriever") as MockRetriever:
        mock_retriever_instance = MagicMock()
        MockRetriever.return_value = mock_retriever_instance
        mock_retriever_instance.invoke.return_value = []

        result = await search_arxiv.ainvoke("nonexistent topic")

        assert "No relevant Arxiv Research Papers were found" in result["context"]


@pytest.mark.asyncio
async def test_search_duckduckgo_success(mock_chain_ainvoke):
    """Успешный веб-поиск"""

    mock_results = [
        {
            "title": "Chemistry Guide",
            "snippet": "Chemistry is the study of matter...",
            "link": "https://example.com/guide",
        }
    ]

    with patch("core.tools.DuckDuckGoSearchAPIWrapper") as MockWrapper:
        mock_wrapper_instance = MagicMock()
        MockWrapper.return_value = mock_wrapper_instance
        mock_wrapper_instance.results.return_value = mock_results

        result = await search_duckduckgo.ainvoke("что такое химия")

        assert "context" in result
        assert "citations" in result
        assert "Chemistry Guide" in str(result["citations"])
