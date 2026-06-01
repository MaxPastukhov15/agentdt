import logging
import os

from app.config.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("db")


class EmbeddingModel:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = cls._init_model()
        return cls._instance

    @classmethod
    def _init_model(cls):
        model_name = settings.embedding_model
        cache_path = settings.models_path.absolute()

        model_exists = cache_path.exists() and any(cache_path.iterdir())

        os.environ["HF_HOME"] = str(cache_path)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_path)

        if model_exists:
            logger.info(f"Local model is found in: {cache_path}. Mode OFFLINE.")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            local_files_only = True
        else:
            logger.warning(f"Model in {cache_path} isn't found. Trying to load from Internet.")
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            local_files_only = False

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=str(cache_path),
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"local_files_only": local_files_only, 
                            "backend" : "onnx"},
            )
            return embeddings
        except Exception as e:
            logger.error(f"Critical error in initializing EmbeddingModel: {e}")
            raise e
