import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.vectordb import VectorRepository


class TestVectorRepository:
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the database"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock QdrantClient"""
        with patch("db.vectordb.QdrantClient") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_embeddings(self):
        """Mock HuggingFaceEmbeddings"""
        with patch("db.vectordb.HuggingFaceEmbeddings") as mock_emb:
            mock_instance = Mock()
            mock_emb.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_vector_store(self):
        """Mock QdrantVectorStore"""
        with patch("db.vectordb.QdrantVectorStore") as mock_store:
            mock_instance = Mock()
            mock_store.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def repository(self, temp_db_path, mock_qdrant_client, mock_embeddings, mock_vector_store):
        """Create a VectorRepository instance with mocks"""
        repo = VectorRepository(location_to_save=temp_db_path, collection_name="test_collection")
        # Replace the client with our mock
        repo.client = mock_qdrant_client
        repo.vector_store = mock_vector_store
        return repo

    def test_context_manager(self, repository, mock_qdrant_client):
        """Test context manager functionality"""
        with repository as repo:
            assert repo == repository

        mock_qdrant_client.close.assert_called_once()

    def test_context_manager_with_error(self, repository, mock_qdrant_client):
        """Test context manager when an error occurs"""
        with pytest.raises(ValueError):
            with repository:
                raise ValueError("Test error")

        mock_qdrant_client.close.assert_called_once()

    def test_normalize_path(self, repository):
        """Test path normalization"""
        test_path = Path("/test/path/to/file.pdf")
        normalized = repository._normalize_path(test_path)

        assert normalized == str(test_path.absolute()).replace("\\", "/")

    def test_doc_id_generation(self, repository, tmp_path):
        """Test document ID generation"""
        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        # Get file stats
        stats = test_file.stat()
        normalized_path = repository._normalize_path(test_file)

        # Generate expected hash
        expected_hash_str = f"{normalized_path}_{stats.st_size}_{stats.st_mtime}"
        expected_doc_id = hashlib.md5(expected_hash_str.encode()).hexdigest()

        # Execute
        doc_id = repository._doc_id(test_file)

        # Assert
        assert doc_id == expected_doc_id

    def test_is_document_ingested_true(self, repository, mock_qdrant_client):
        """Test checking if document is already ingested (returns True)"""
        # Setup
        mock_qdrant_client.count.return_value.count = 5

        # Execute
        result = repository._is_document_ingested("test_doc_id")

        # Assert
        assert result is True
        mock_qdrant_client.count.assert_called_once()

    def test_is_document_ingested_false(self, repository, mock_qdrant_client):
        """Test checking if document is not ingested (returns False)"""
        # Setup
        mock_qdrant_client.count.return_value.count = 0

        # Execute
        result = repository._is_document_ingested("test_doc_id")

        # Assert
        assert result is False

    @patch("db.vectordb.PyMuPDF4LLMLoader")
    def test_load_pdf_new_document(self, mock_loader, repository, tmp_path):
        """Test loading a PDF that hasn't been ingested"""
        # Setup
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        # Mock the loader
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance

        # Create mock documents
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

        # Mock the document check to return False (not ingested)
        repository._is_document_ingested = Mock(return_value=False)

        # Execute
        result = repository._load_pdf(test_file)

        # Assert
        assert len(result) > 0
        for doc in result:
            assert "doc_id" in doc.metadata
            assert "source" in doc.metadata
            assert "ingestion_timestamp" in doc.metadata
            assert doc.metadata["source"] == repository._normalize_path(test_file)

    @patch("db.vectordb.PyMuPDF4LLMLoader")
    def test_load_pdf_already_ingested(self, mock_loader, repository, tmp_path):
        """Test loading a PDF that has already been ingested"""
        # Setup
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        # Mock the document check to return True (already ingested)
        repository._is_document_ingested = Mock(return_value=True)

        # Execute
        result = repository._load_pdf(test_file)

        # Assert
        assert result == []
        mock_loader.assert_not_called()

    def test_startup_db_no_pdfs(self, repository, monkeypatch):
        """Test startup when no PDF files exist"""
        # Mock the directory search to return empty list
        mock_rglob = Mock(return_value=[])

        with patch.object(Path, "rglob", mock_rglob):
            result = repository.startup_db()

        assert result == repository
        mock_rglob.assert_called_once_with("*.pdf")

    def test_startup_db_with_pdfs(self, repository, tmp_path):
        """Test startup with PDF files"""
        # Create mock PDF files
        pdf_dir = tmp_path / "pdf_docs"
        pdf_dir.mkdir()
        test_pdf = pdf_dir / "test.pdf"
        test_pdf.write_text("dummy content")

        # Mock Path resolution
        with patch.object(Path, "resolve") as mock_resolve:
            mock_resolve.return_value.parent = tmp_path

            # Mock _load_pdf to return some chunks
            mock_chunks = [
                Document(page_content="chunk1"),
                Document(page_content="chunk2"),
            ]
            repository._load_pdf = Mock(return_value=mock_chunks)

            # Execute
            with patch.dict(os.environ, {"BATCH_SIZE": "1"}):
                result = repository.startup_db()

            # Assert
            assert result == repository
            repository._load_pdf.assert_called_once()
            repository.vector_store.add_documents.assert_called()

    def test_get_retriever(self, repository, mock_vector_store):
        """Test getting retriever"""
        # Setup mock retriever
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        # Execute
        result = repository.get_retriever()

        # Assert
        assert result == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 5})

    def test_close_method(self, repository, mock_qdrant_client):
        """Test close method"""
        repository._VectorRepository__close()
        mock_qdrant_client.close.assert_called_once()

    def test_close_without_client(self, repository):
        """Test close when client doesn't exist"""
        repository.client = None
        # This should not raise an exception
        repository._VectorRepository__close()


class TestVectorRepositoryIntegration:
    """Integration tests with actual dependencies"""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the database"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_workflow(self, temp_db_path, tmp_path):
        """Test the full workflow with actual components"""
        # Create a test PDF directory
        pdf_dir = tmp_path / "pdf_docs"
        pdf_dir.mkdir()

        # Create a dummy PDF file (you might want to create a real PDF for testing)
        test_pdf = pdf_dir / "test.pdf"
        test_pdf.write_text("%PDF-1.4\n%PDF content would go here")

        # Test startup
        with VectorRepository(location_to_save=temp_db_path, collection_name="test_collection") as repo:
            result = repo.startup_db()

            # Get retriever
            retriever = result.get_retriever()

            assert retriever is not None
