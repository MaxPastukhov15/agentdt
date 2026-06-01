import asyncio
import logging
from typing import Any

from app.db.vectordb import VectorRepository
from langchain.tools import tool
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from app.utils.summarization_chain import create_summarizer
from app.utils.text_cleaner import clean_text

tools_logger = logging.getLogger("tools")

LIMIT = asyncio.Semaphore(2)
_REPOS = {}


def get_repo(collection_name: str):
    if collection_name not in _REPOS:
        _REPOS[collection_name] = VectorRepository(collection_name)
    return _REPOS[collection_name]



async def process_arxiv_doc(doc: Document) -> str:
    """Очистка без использования сторонней LLM"""
    content = doc.page_content

    content = clean_text(content, mode="content")
    
    metadata = doc.metadata
    return f"Title: {metadata.get('Title')}\nAuthors: {metadata.get('Authors')}\nContent: {content}\nLink: {metadata.get('Entry ID')}\n"


@tool
async def search_chemistry_collection(query: str) -> dict[str, Any]:
    """
    Осуществляет семантический поиск по внутренней базе знаний в области химии.

    WHEN TO USE:
    Используйте этот инструмент, когда пользователь задает вопрос о химии или связанных с ней областях,
    и ответа нет в общем контексте диалога.

    ARGS:
    - query: Строка поиска. Сформулируй её как ключевые слова, а не полный вопрос.
             Например, вместо "Что такое атом?" используйте "определение атома".

    RETURNS:
    Список словарей (str). Строка состоит из нескольких фактов, где каждый факт имеет вид :
      "(источник, страница): информация из книги"

    EXAMPLES:
    search_chemistry_collection(query="определение атома")

    WARNINGS:
    - Если результатов 0, сообщите пользователю, что информация не найдена.
    """
    repo = get_repo("chemistry_collection")
    results = await repo.get_retriever().ainvoke(query)

    tools_logger.info(f"Search in collections: found {len(results)} documents for queries '{query}'")

    formatted_results = [
        (
            f"{doc.metadata['source']}, {doc.metadata['page']}",
            doc.page_content,
        )
        for doc in results
    ]

    context = "\n\n".join([res[1] for res in formatted_results])
    citations = [res[0] for res in formatted_results]

    tools_logger.info(f"Used for context: {context[:50]}\n\n {citations}" + "...")
    return {"context": context, "citations": citations}


@tool
async def search_arxiv(query: str) -> dict[str, Any]:
    """
    Поиск научных статей на arXiv по химии и смежным областям.

    WHEN TO USE:
    - Когда нужны актуальные исследования, препринты, ссылки на работы
    - Когда вопрос требует научной новизны или цитирования источников
    - НЕ используйте для базовых определений (для этого есть search_chemistry_collection)

    ARGS:
    - query: Поисковый запрос на английском (arXiv индексирует en).
             Пример: "density functional theory catalysis"

    RETURNS:
    [1] Title: str
        Authors: str
        Summary: <100-200 words>

    WARNINGS:
    - Если статей не найдено, верните "No relevant papers found"
    - Не цитируйте статью, не проверив её релевантность
    """

    retriever = ArxivRetriever(load_max_docs=5, get_full_documents=False)

    try:
        docs = await asyncio.to_thread(retriever.invoke, query)
    
        if not docs:
            return {"context": "Статьи не найдены.", "citations": []}

        processed_contents = [await process_arxiv_doc(d) for d in docs]
    
    
        full_context = "\n---\n".join(processed_contents)
        citations = [f"{d.metadata.get('Authors', 'N/A')}; {d.metadata.get('Title', 'N/A')}; {d.metadata.get('Entry ID', 'N/A')}" 
                     for d in docs]
    
        return {"context": full_context, "citations": citations}

    except Exception as e:
        return {"context": f"Error during Arxiv search: {str(e)}", "citations": []}


@tool
async def search_duckduckgo(query: str, deep_read: bool = False) -> dict[str, Any]:
    '''"""
    Выполняет веб-поиск с помощью DuckDuckGo для поиска общей информации и ресурсов.

    WHEN TO USE:
    Используйте этот инструмент, когда пользователь задает вопросы, которые требуют
    общих знаний, текущих событий или ресурсов, недоступных во внутренней базе знаний по химии.

    ARGS:
    - query: Поисковый запрос. Формулируйте его как ключевые слова, а не полный вопрос.
             Например, вместо "Что такое машинное обучение?" используйте "определение машинного обучения".
    - deep_read: Позволяет либо воспользоваться сниппетами для простых вопросов, либо полностью глубоко изучить
    источник

    RETURNS:
    Словарь, содержащий:
    - context: Детальная информация из результатов поиска
    - citations: Список названий источников и ссылок

    EXAMPLES:
    search_duckduckgo(query="производственный процесс поливинилхлорида")

    WARNINGS:
    - Если результаты не найдены, сообщите пользователю, что соответствующая информация не найдена.
    - Учитывайте, что веб-источники могут различаться по надежности по сравнению с академическими источниками.
    """'''
    wrapper = DuckDuckGoSearchAPIWrapper()
    results = await asyncio.to_thread(wrapper.results, query, max_results=4)

    if not results:
        return {"context": "Ничего не найдено.", "citations": []}

    if deep_read:
        bs_transformer = BeautifulSoupTransformer()

        url = results[1]["link"]
        loader = AsyncHtmlLoader([url])

        docs = await asyncio.to_thread(loader.load)

        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["p", "li", "article", "h1", "h2"])

        full_text = docs_transformed[0].page_content if docs_transformed else "Не удалось извлечь текст."

        full_text = clean_text(full_text, mode="content")

        tools_logger.info(f"Used for context: {full_text[:50]}\n\n {url}" + "...")

        return {"context": f"Глубокий анализ источника:\n{full_text[:10000]}", "citations": [url]}

    else:
        documents = [
            Document(
                page_content=doc["snippet"],
                metadata={"title": doc["title"], "source": doc["link"]},
            )
            for doc in results
        ]

        context = "\n\n".join([doc.page_content for doc in documents])
        citations = [doc.metadata.get("source", "N/A") for doc in documents]

        tools_logger.info(f"Used for context: {context[:50]}\n\n {citations}" + "...")
        return {"context": context, "citations": citations}


TOOLS = [search_chemistry_collection, search_arxiv, search_duckduckgo]

def cleanup_repos():
    for repo in _REPOS.values():
        try:
            repo._VectorRepository__close()
        except Exception:
            pass
    _REPOS.clear()