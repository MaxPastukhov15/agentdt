import asyncio
import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Self

from app.config.config import settings
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
from tqdm import tqdm
from app.utils.text_cleaner import clean_text

from app.db.embedding_factory import EmbeddingModel

db_logger = logging.getLogger("db")


def _normalize_path(file_path: Path) -> str:
    return str(file_path.absolute()).replace("\\", "/")


class VectorRepository:
    def __init__(self, collection_name: str) -> None:
        self.location = str(settings.db_path)
        self.client = QdrantClient(path=self.location)
        self.collection_name = collection_name

        self.embeddings: HuggingFaceEmbeddings = EmbeddingModel.get_model()

        if not self.client.collection_exists(self.collection_name):
            db_logger.debug(f"""creating new collection 
                {self.collection_name} in direcotry {self.location}""")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        self.vector_store: QdrantVectorStore = QdrantVectorStore(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self.client,
        )

        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=204,
            length_function=len,
            is_separator_regex=False,
        )

        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])

    @staticmethod
    def _process_file_worker(task_data):
        file_path, doc_id, _ = task_data

        loader = PyMuPDF4LLMLoader(file_path=file_path, mode="page", ignore_graphics=True, ignore_images=True, table_strategy=None)
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=204,
            length_function=len,
            is_separator_regex=False,
        )

        docs = loader.load()
        md_header_splits: List[Document] = []

        for doc in tqdm(docs):
            db_logger.debug(f"""Document processing begins 
                    {doc.metadata.get("source")}, 
                    page {doc.metadata.get("page")}\n""")

            doc.metadata = {
                "doc_id": doc_id,
                "source": _normalize_path(file_path),
                "ingestion_timestamp": datetime.now().isoformat(),
                "creationdate": doc.metadata.get("creationdate"),
                "total_pages": doc.metadata.get("total_pages"),
                "format": doc.metadata.get("format"),
                "page": doc.metadata.get("page"),
            }

            cleaned = clean_text(doc.page_content, mode="data")

            splits = md_splitter.split_text(cleaned)
            for s in splits:
                s.metadata.update({"doc_id": doc_id, "source": str(file_path).replace("\\", "/"), "page": doc.metadata.get("page")})
                md_header_splits.append(s)

        return text_splitter.split_documents(md_header_splits)

    def __enter__(self) -> Self:
        db_logger.info(f"Connection with the {self.client} has been established")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type:
            db_logger.error(f"Client {self.client} is closed incorrectly")
        self.__close()

    def _doc_id(self, file_name: Path) -> str:
        stats = file_name.stat()
        normalized_path = _normalize_path(file_name)

        hash_str = f"{normalized_path}_{stats.st_size}_{stats.st_mtime}"

        doc_id = hashlib.md5(hash_str.encode()).hexdigest()

        db_logger.debug(f"Generated doc_id {doc_id} for {normalized_path}")

        return doc_id

    def _is_document_ingested(self, doc_id: str) -> bool:
        results = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id))]),
        )
        return results.count > 0

    def _load_pdf(self, file_name: Path) -> List[Document]:
        doc_id = self._doc_id(file_name)
        count = 0

        if self._is_document_ingested(doc_id):
            db_logger.debug(f"the {file_name} has already been ingested")
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

        for doc in tqdm(docs):
            db_logger.debug(f"""Document processing begins 
                    {doc.metadata.get("source")}, 
                    page {doc.metadata.get("page")}\n""")

            doc.metadata = {
                "doc_id": doc_id,
                "source": _normalize_path(file_name),
                "ingestion_timestamp": datetime.now().isoformat(),
                "creationdate": doc.metadata.get("creationdate"),
                "total_pages": doc.metadata.get("total_pages"),
                "format": doc.metadata.get("format"),
                "page": doc.metadata.get("page"),
            }
            cleaned_text = clean_text(doc.page_content, mode="data")

            header_split = self.markdown_splitter.split_text(cleaned_text)

            for split in header_split:
                split.metadata.update(doc.metadata)
                count += 1

                if count % 10 == 0:
                    db_logger.debug(f"\n{split.metadata}\n")

                md_header_splits.append(split)

        return self.text_splitter.split_documents(md_header_splits)

    def startup_db(self) -> Self:
        BATCH_SIZE = 100

        directory: Path = settings.pdf_docs_path

        files_dirs: List = list(directory.rglob("*.pdf"))

        db_logger.debug(f"Found {len(files_dirs)} PDF files in {directory}")

        tasks = []

        for file_path in files_dirs:
            doc_id = self._doc_id(file_path)
            if not self._is_document_ingested(doc_id):
                tasks.append((file_path, doc_id, {}))
            else:
                db_logger.debug(f"the {file_path} has already been ingested")

        if not tasks:
            db_logger.warning("All files have already been indexed.")
            return self

        num_files = len(tasks)

        if num_files == 1:
            max_workers = 1
        elif num_files == 2:
            max_workers = 2
        else:
            max_workers = min(os.cpu_count() or 1, 3)

        db_logger.info(f"Start of indexing {num_files} files on {max_workers} kernels of cpu")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(VectorRepository._process_file_worker, tasks))

        for all_chunks in results:
            if all_chunks:
                for i in range(0, len(all_chunks), BATCH_SIZE):
                    batch = all_chunks[i : i + BATCH_SIZE]
                    self.vector_store.add_documents(batch)

        db_logger.info(f"Indexing is ended of {num_files} files")

        return self

    async def ingest_file(self, file_path: Path):
        chunks = self._load_pdf(file_path)
        if chunks:
            await asyncio.to_thread(self.vector_store.add_documents, chunks)
        return len(chunks)

    def get_retriever(self) -> VectorStoreRetriever:
        db_logger.info(f"Retriever for {self.collection_name} has been created")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    def __close(self):
        if hasattr(self, "client") and self.client:
            db_logger.info(f"Connection with {self.client} is closed")
            self.client.close()

        else:
            db_logger.error("Client doesn't exists")
