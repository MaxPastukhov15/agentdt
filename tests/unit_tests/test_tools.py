from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.core.tools import search_arxiv, search_chemistry_collection, search_duckduckgo


@pytest.fixture(autouse=True)
def clear_repo_cache():
    from app.core.tools import _REPOS
    _REPOS.clear()


@pytest.fixture
def mock_vector_repo():
    with (
        patch("app.core.tools.VectorRepository") as MockRepo,
        patch("app.core.tools.get_repo") as mock_get_repo,
    ):
        mock_instance = MagicMock()
        mock_retriever = AsyncMock()
        mock_instance.get_retriever.return_value = mock_retriever
        MockRepo.return_value = mock_instance
        mock_get_repo.return_value = mock_instance
        yield mock_instance, mock_retriever


@pytest.mark.asyncio
async def test_search_chemistry_collection_success(mock_vector_repo):
    mock_instance, mock_retriever = mock_vector_repo

    test_docs = [
        Document(
            page_content="Water is H2O",
            metadata={"source": "chemistry_basics.pdf", "page": 42},
        )
    ]
    mock_retriever.ainvoke.return_value = test_docs

    result = await search_chemistry_collection.ainvoke("что такое вода")

    assert "context" in result
    assert "citations" in result
    assert "Water is H2O" in result["context"]
    assert "chemistry_basics.pdf, 42" in result["citations"]


@pytest.mark.asyncio
async def test_search_chemistry_collection_empty_results(mock_vector_repo):
    mock_instance, mock_retriever = mock_vector_repo
    mock_retriever.ainvoke.return_value = []

    result = await search_chemistry_collection.ainvoke("несуществующий термин")

    assert result["context"] == ""
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_search_chemistry_collection_exception(mock_vector_repo):
    mock_instance, mock_retriever = mock_vector_repo
    mock_retriever.ainvoke.side_effect = ConnectionError("DB unavailable")

    with pytest.raises(ConnectionError):
        await search_chemistry_collection.ainvoke("тест")


@pytest.mark.asyncio
async def test_search_arxiv_success():
    with patch("app.core.tools.ArxivRetriever") as MockRetriever:
        mock_retriever_instance = MagicMock()
        MockRetriever.return_value = mock_retriever_instance

        test_docs = [
            Document(
                page_content="DFT study of catalysts",
                metadata={
                    "Title": "DFT Paper",
                    "Authors": "Lee et al.",
                    "Entry ID": "http://arxiv.org/abs/1234.5678",
                },
            )
        ]
        mock_retriever_instance.invoke.return_value = test_docs

        result = await search_arxiv.ainvoke("density functional theory")

        assert "context" in result
        assert "citations" in result
        assert "DFT Paper" in str(result["context"])
        assert "Lee et al." in str(result["context"])


@pytest.mark.asyncio
async def test_search_arxiv_no_results():
    with patch("app.core.tools.ArxivRetriever") as MockRetriever:
        mock_retriever_instance = MagicMock()
        MockRetriever.return_value = mock_retriever_instance
        mock_retriever_instance.invoke.return_value = []

        result = await search_arxiv.ainvoke("nonexistent topic")

        assert "Статьи не найдены" in result["context"]


@pytest.mark.asyncio
async def test_search_arxiv_exception():
    with patch("app.core.tools.ArxivRetriever") as MockRetriever:
        mock_retriever_instance = MagicMock()
        MockRetriever.return_value = mock_retriever_instance
        mock_retriever_instance.invoke.side_effect = Exception("Network error")

        result = await search_arxiv.ainvoke("broken request")

        assert "Error during Arxiv search" in result["context"]
        assert result["citations"] == []


@pytest.mark.asyncio
async def test_search_duckduckgo_success():
    mock_results = [
        {
            "title": "Chemistry Guide",
            "snippet": "Chemistry is the study of matter...",
            "link": "https://example.com/guide",
        }
    ]

    with patch("app.core.tools.DuckDuckGoSearchAPIWrapper") as MockWrapper:
        mock_wrapper_instance = MagicMock()
        MockWrapper.return_value = mock_wrapper_instance
        mock_wrapper_instance.results.return_value = mock_results

        result = await search_duckduckgo.ainvoke("что такое химия")

        assert "context" in result
        assert "citations" in result
        assert "https://example.com/guide" in str(result["citations"])


@pytest.mark.asyncio
async def test_search_duckduckgo_no_results():
    with patch("app.core.tools.DuckDuckGoSearchAPIWrapper") as MockWrapper:
        mock_wrapper_instance = MagicMock()
        MockWrapper.return_value = mock_wrapper_instance
        mock_wrapper_instance.results.return_value = []

        result = await search_duckduckgo.ainvoke("asdfghjkl")

        assert "Ничего не найдено" in result["context"]
        assert result["citations"] == []
