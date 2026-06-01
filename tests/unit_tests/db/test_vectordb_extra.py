import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.db.vectordb import VectorRepository


class TestVectorRepositoryAdditional:
    """Additional tests beyond the existing test_vectordb.py"""

    @pytest.fixture(autouse=True)
    def isolate_settings_paths(self, tmp_path):
        test_dir = tmp_path / "agentdt_test"
        mock_settings = MagicMock()
        mock_settings.db_path = test_dir / "db"
        mock_settings.models_path = test_dir / "models"
        mock_settings.pdf_docs_path = test_dir / "pdf_docs"
        mock_settings.chat_history_path = test_dir / "history" / "test.db"
        mock_settings.long_term_memory = test_dir / "memory"
        with patch("app.db.vectordb.settings", mock_settings):
            yield

    @pytest.fixture
    def mock_qdrant_client(self):
        with patch("app.db.vectordb.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.collection_exists.return_value = True
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_embeddings(self):
        with patch("app.db.vectordb.EmbeddingModel.get_model") as mock_get_model:
            mock_instance = Mock()
            mock_get_model.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_vector_store(self):
        with patch("app.db.vectordb.QdrantVectorStore") as mock_store:
            mock_instance = Mock()
            mock_store.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def repository(self, mock_qdrant_client, mock_embeddings, mock_vector_store):
        repo = VectorRepository(collection_name="test_collection")
        repo.client = mock_qdrant_client
        repo.vector_store = mock_vector_store
        return repo

    def test_constructor_sets_location(self, mock_qdrant_client, mock_embeddings, mock_vector_store):
        repo = VectorRepository(collection_name="custom_coll")
        assert repo.collection_name == "custom_coll"

    def test_constructor_creates_client(self, mock_qdrant_client, mock_embeddings, mock_vector_store):
        repo = VectorRepository(collection_name="coll")
        assert repo.client is not None

    def test_constructor_creates_collection_if_not_exists(self, mock_embeddings, mock_vector_store):
        with patch("app.db.vectordb.QdrantClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.collection_exists.return_value = False
            MockClient.return_value = mock_instance

            VectorRepository(collection_name="new_coll")

            mock_instance.create_collection.assert_called_once()

    def test_enter_returns_self(self, repository):
        with repository as repo:
            assert repo is repository

    def test_exit_closes_client(self, repository, mock_qdrant_client):
        with repository:
            pass
        mock_qdrant_client.close.assert_called_once()

    def test_exit_with_error_still_closes(self, repository, mock_qdrant_client):
        with pytest.raises(RuntimeError):
            with repository:
                raise RuntimeError("test error")
        mock_qdrant_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_new_document(self, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        with patch("app.db.vectordb.PyMuPDF4LLMLoader") as MockLoader:
            mock_loader = MagicMock()
            mock_doc = MagicMock()
            mock_doc.page_content = "# Header\nContent"
            mock_doc.metadata = {"source": str(test_file), "page": 1, "total_pages": 1}
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [mock_doc]
            MockLoader.return_value = mock_loader_instance

            repository._is_document_ingested = Mock(return_value=False)

            result = await repository.ingest_file(test_file)

            assert result > 0
            repository.vector_store.add_documents.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_file_already_ingested(self, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        with patch("app.db.vectordb.PyMuPDF4LLMLoader") as MockLoader:
            repository._is_document_ingested = Mock(return_value=True)

            result = await repository.ingest_file(test_file)

            assert result == 0
            MockLoader.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_file_empty_chunks(self, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        with patch("app.db.vectordb.PyMuPDF4LLMLoader") as MockLoader:
            mock_doc = MagicMock()
            mock_doc.page_content = ""
            mock_doc.metadata = {"source": str(test_file), "page": 1}
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [mock_doc]
            MockLoader.return_value = mock_loader_instance

            repository._is_document_ingested = Mock(return_value=False)

            result = await repository.ingest_file(test_file)

            assert result == 0
