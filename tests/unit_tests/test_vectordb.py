import hashlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from app.db.vectordb import VectorRepository, _normalize_path
from langchain_core.documents import Document


class TestVectorRepository:
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

    def test_context_manager(self, repository, mock_qdrant_client):
        with repository as repo:
            assert repo == repository

        mock_qdrant_client.close.assert_called_once()

    def test_context_manager_with_error(self, repository, mock_qdrant_client):
        with pytest.raises(ValueError):
            with repository:
                raise ValueError("Test error")

        mock_qdrant_client.close.assert_called_once()

    def test_normalize_path(self):
        test_path = Path("/test/path/to/file.pdf")
        normalized = _normalize_path(test_path)

        assert normalized == str(test_path.absolute()).replace("\\", "/")

    def test_doc_id_generation(self, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        stats = test_file.stat()
        normalized_path = _normalize_path(test_file)

        expected_hash_str = f"{normalized_path}_{stats.st_size}_{stats.st_mtime}"
        expected_doc_id = hashlib.md5(expected_hash_str.encode()).hexdigest()

        doc_id = repository._doc_id(test_file)

        assert doc_id == expected_doc_id

    def test_is_document_ingested_true(self, repository, mock_qdrant_client):
        mock_qdrant_client.count.return_value.count = 5

        result = repository._is_document_ingested("test_doc_id")

        assert result is True
        mock_qdrant_client.count.assert_called_once()

    def test_is_document_ingested_false(self, repository, mock_qdrant_client):
        mock_qdrant_client.count.return_value.count = 0

        result = repository._is_document_ingested("test_doc_id")

        assert result is False

    @patch("app.db.vectordb.PyMuPDF4LLMLoader")
    def test_load_pdf_new_document(self, mock_loader, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance

        mock_doc = Document(
            page_content="# Header 1\nContent\n## Header 2\nMore content",
            metadata={
                "source": str(test_file),
                "page": 1,
                "total_pages": 10,
                "creationdate": "2024-01-01",
                "format": "PDF",
            },
        )
        mock_loader_instance.load.return_value = [mock_doc]

        repository._is_document_ingested = Mock(return_value=False)

        result = repository._load_pdf(test_file)

        assert len(result) > 0
        for doc in result:
            assert "doc_id" in doc.metadata
            assert "source" in doc.metadata
            assert "ingestion_timestamp" in doc.metadata
            assert doc.metadata["source"] == _normalize_path(test_file)

    @patch("app.db.vectordb.PyMuPDF4LLMLoader")
    def test_load_pdf_already_ingested(self, mock_loader, repository, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        repository._is_document_ingested = Mock(return_value=True)

        result = repository._load_pdf(test_file)

        assert result == []
        mock_loader.assert_not_called()

    def test_startup_db_no_pdfs(self, repository, monkeypatch):
        mock_rglob = Mock(return_value=[])

        with patch.object(Path, "rglob", mock_rglob):
            result = repository.startup_db()

        assert result == repository

    def test_startup_db_with_pdfs(self, repository, tmp_path):
        pdf_dir = tmp_path / "pdf_docs"
        pdf_dir.mkdir()
        test_pdf = pdf_dir / "test.pdf"
        test_pdf.write_text("dummy content")

        mock_chunks = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2"),
        ]

        with (
            patch.object(Path, "rglob") as mock_rglob,
            patch("app.db.vectordb.ProcessPoolExecutor") as MockExecutor,
        ):
            mock_rglob.return_value = [test_pdf]
            mock_executor_instance = MagicMock()
            mock_executor_instance.__enter__.return_value = mock_executor_instance
            mock_executor_instance.map.return_value = [mock_chunks]
            MockExecutor.return_value = mock_executor_instance

            repository._is_document_ingested = Mock(return_value=False)

            result = repository.startup_db()

            assert result == repository
            repository.vector_store.add_documents.assert_called()

    def test_get_retriever(self, repository, mock_vector_store):
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        result = repository.get_retriever()

        assert result == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def test_close_method(self, repository, mock_qdrant_client):
        repository._VectorRepository__close()
        mock_qdrant_client.close.assert_called_once()

    def test_close_without_client(self, repository):
        repository.client = None
        repository._VectorRepository__close()
