import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.db.embedding_factory import EmbeddingModel


class TestEmbeddingModel:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        EmbeddingModel._instance = None
        yield

    @pytest.fixture
    def mock_settings(self, tmp_path):
        with patch("app.db.embedding_factory.settings") as mock:
            mock.embedding_model = "BAAI/bge-m3"
            mock.models_path = tmp_path / "models"
            yield mock

    def test_singleton_returns_same_instance(self, mock_settings):
        with patch("app.db.embedding_factory.HuggingFaceEmbeddings"):
            first = EmbeddingModel.get_model()
            second = EmbeddingModel.get_model()
            assert first is second

    def test_get_model_calls_init_once(self, mock_settings):
        with (
            patch("app.db.embedding_factory.HuggingFaceEmbeddings"),
            patch.object(EmbeddingModel, "_init_model") as mock_init,
        ):
            mock_init.return_value = "fake_model"
            EmbeddingModel.get_model()
            EmbeddingModel.get_model()
            mock_init.assert_called_once()

    def test_reset_reinitializes(self, mock_settings):
        first_fake = MagicMock()
        second_fake = MagicMock()
        with patch("app.db.embedding_factory.HuggingFaceEmbeddings", side_effect=[first_fake, second_fake]):
            first = EmbeddingModel.get_model()
            EmbeddingModel._instance = None
            second = EmbeddingModel.get_model()
            assert first is not second

    def test_get_model_returns_valid_instance(self, mock_settings):
        fake_emb = MagicMock()
        with patch("app.db.embedding_factory.HuggingFaceEmbeddings", return_value=fake_emb):
            result = EmbeddingModel.get_model()
            assert result is fake_emb

    @patch("app.db.embedding_factory.HuggingFaceEmbeddings")
    def test_init_model_sets_env_vars(self, MockHF, mock_settings):
        EmbeddingModel._init_model()

        cache_str = str(mock_settings.models_path.absolute())
        assert os.environ["HF_HOME"] == cache_str
        assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == cache_str

    @patch("app.db.embedding_factory.HuggingFaceEmbeddings")
    def test_init_model_with_local_cache(self, MockHF, mock_settings):
        models_path = mock_settings.models_path
        models_path.mkdir(parents=True)
        (models_path / "model_file.bin").write_text("fake")

        EmbeddingModel._init_model()

        assert os.environ["HF_HUB_OFFLINE"] == "1"
        MockHF.assert_called_once_with(
            model_name=mock_settings.embedding_model,
            cache_folder=str(models_path.absolute()),
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"local_files_only": True, "backend": "onnx"},
        )

    @patch("app.db.embedding_factory.HuggingFaceEmbeddings")
    def test_init_model_without_local_cache(self, MockHF, mock_settings):
        models_path = mock_settings.models_path
        models_path.mkdir(parents=True)

        EmbeddingModel._init_model()

        assert os.environ["HF_HUB_OFFLINE"] == "0"
        MockHF.assert_called_once_with(
            model_name=mock_settings.embedding_model,
            cache_folder=str(models_path.absolute()),
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"local_files_only": False, "backend": "onnx"},
        )

    @patch("app.db.embedding_factory.HuggingFaceEmbeddings")
    def test_init_model_cache_dir_not_exists(self, MockHF, mock_settings):
        EmbeddingModel._init_model()

        assert os.environ["HF_HUB_OFFLINE"] == "0"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
        MockHF.assert_called_once_with(
            model_name=mock_settings.embedding_model,
            cache_folder=str(mock_settings.models_path.absolute()),
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"local_files_only": False, "backend": "onnx"},
        )

    @patch("app.db.embedding_factory.HuggingFaceEmbeddings")
    def test_init_model_raises_on_hf_error(self, MockHF, mock_settings):
        MockHF.side_effect = ValueError("Model load failed")

        with pytest.raises(ValueError, match="Model load failed"):
            EmbeddingModel._init_model()

    def test_clear_instance(self, mock_settings):
        with patch("app.db.embedding_factory.HuggingFaceEmbeddings"):
            EmbeddingModel.get_model()
            assert EmbeddingModel._instance is not None
            EmbeddingModel._instance = None
            assert EmbeddingModel._instance is None
