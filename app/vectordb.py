import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Self

from config.config import settings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger("db")

load_dotenv()


class VectorRepository:
    def __init__(self, location_to_save: str, collection_name: str) -> None:
        self.client = QdrantClient(path=location_to_save)

        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.collection_name = collection_name

        if not self.client.collection_exists(self.collection_name):
            logger.debug(
                f"creating new collection {self.collection_name} in direcotry {location_to_save}"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        self.vector_store: QdrantVectorStore = QdrantVectorStore(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self.client,
        )

        self.text_splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=204,
                length_function=len,
                is_separator_regex=False,
            )
        )

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )

    def __enter__(self) -> Self:
        logger.info(f"Connection with the {self.client} has been established")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type:
            logger.error(f"Client {self.client} is closed incorrectly")
        self.__close()

    def _normalize_path(self, file_path: Path) -> str:
        abs_path = file_path.absolute()

        return str(abs_path).replace("\\", "/")

    def _doc_id(self, file_name: Path) -> str:
        stats = file_name.stat()
        normalized_path = self._normalize_path(file_name)

        hash_str = f"{normalized_path}_{stats.st_size}_{stats.st_mtime}"

        doc_id = hashlib.md5(hash_str.encode()).hexdigest()

        logger.debug(f"Generated doc_id {doc_id} for {normalized_path}")

        return doc_id

    def _is_document_ingested(self, doc_id: str) -> bool:
        results = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id", match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
        )
        return results.count > 0

    def _load_pdf(self, file_name: Path) -> List[Document]:
        doc_id = self._doc_id(file_name)
        count = 0

        if self._is_document_ingested(doc_id):
            logger.debug(f"the {file_name} has already been ingested")
            return []

        loader = PyMuPDF4LLMLoader(
            file_path=file_name,
            table_strategy=None,
            ignore_graphics=True,
            ignore_images=True,
            mode="page",
        )

        docs = loader.load()

        md_header_splits: List[Document] = []

        for doc in docs:
            logger.debug(
                f"Document processing begins {doc.metadata.get('source')}, page {doc.metadata.get('page')}\n"
            )

            doc.metadata = {
                "doc_id": doc_id,
                "source": self._normalize_path(file_name),
                "ingestion_timestamp": datetime.now().isoformat(),
                "creationdate": doc.metadata.get("creationdate"),
                "total_pages": doc.metadata.get("total_pages"),
                "format": doc.metadata.get("format"),
                "page": doc.metadata.get("page"),
            }

            header_split = self.markdown_splitter.split_text(doc.page_content)

            for split in header_split:
                split.metadata.update(doc.metadata)
                count += 1

                if count % 10 == 0:
                    logger.debug(f"\n{split.metadata}\n")

                md_header_splits.append(split)

        return self.text_splitter.split_documents(md_header_splits)

    def startup_db(self) -> Self:
        BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))

        directory: Path = Path(__file__).resolve().parent.parent / "pdf_docs"

        files_dirs: List = list(directory.rglob("*.pdf"))

        logger.debug(f"Found {len(files_dirs)} PDF files in {directory}")

        if len(files_dirs) == 0:
            logger.warning(f"{directory} doesn't contain any pdf files")
            return self

        for file_name in files_dirs:
            all_chunks = self._load_pdf(file_name=file_name)

            if all_chunks:
                for i in range(0, len(all_chunks), BATCH_SIZE):
                    self.vector_store.add_documents(all_chunks[i : i + BATCH_SIZE])

            logger.debug(f"Document {file_name} was successfully loaded!")

        logger.info(f"Files {files_dirs} was successfully loaded")
        return self

    def get_retriever(self) -> VectorStoreRetriever:
        logger.info(f"Retriever for {self.collection_name} has been created")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    def __close(self):
        if hasattr(self, "client") and self.client:
            logger.info(f"Connection with {self.client} is closed")
            self.client.close()

        else:
            logger.error("Client doesn't exists")
