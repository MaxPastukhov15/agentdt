from langchain.tools import tool
from langchain_community.retrievers import ArxivRetriever
from db.vectordb import VectorRepository
import logger_config
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import asyncio
from typing import Any

load_dotenv()

logger = logging.getLogger("tools")

LIMIT = asyncio.Semaphore(2)

llm_summarizer = ChatOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", 
                 model="arcee-ai/trinity-large-preview:free", temperature=0.0, max_completion_tokens=2000)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Действуй как аналитик. Подготовь сжатый отчет (summary) на основе предоставленного контекста.
    Этот отчет будет использован другой нейросетью, поэтому пиши максимально информативно, без вводных фраз. 
    Особо концентрируй внимание на выводах в статье, если это информация из интернета со ссылкой просто отыщи ответ на вопрос"""),
    ("human", "Контекст: {context}\n\nЗадача: {query}\n\nФормат: тезис, bullet points, вывод (100-200 слов).")
])

chain = prompt | llm_summarizer.with_retry(wait_exponential_jitter=True, stop_after_attempt=3) | StrOutputParser()


@tool
async def search_chemistry_collection(query: str) -> dict[str, Any]:
    '''
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
    '''
    with VectorRepository('./db/collections/chemistry_collection', 
                         collection_name="chemistry_collection") as repo:

        results = await repo.get_retriever().ainvoke(query)
        
        formatted_results = [
            (f"{doc.metadata['source']}, {doc.metadata['page']}",
             f"({doc.metadata['source']}, {doc.metadata['page']}): {doc.page_content}") 
             for doc in results]
        
        context = "\n\n".join([res[1] for res in formatted_results])
        citations = [res[0] for res in formatted_results] 

    print(f"Used for context: {context}\n\n {'\n'.join(citations)}")
    return {'context' : context, 'citations' : citations}


async def create_summary(query: str, content: Document)-> tuple[str, str] | str:
    """Вспомогательная функция для обработки одного документа"""

    try:
        async with LIMIT:
            summary = await chain.ainvoke({"context": content.page_content[:300000], "query": query})
        
        title = content.metadata.get('title', 'N/A')
        author = content.metadata.get('authors', 'N/A')
        link = content.metadata.get('link', 'N/A')
        
        if author != 'N/A' and link != 'N/A':
            return (f"{title}, {author}, {link}", f"Title: {title}\nAuthor: {author}\nLink: {link}\nSummary:\n{summary}")
        elif author != 'N/A':
            return (f"{title}, {author}", f"Title: {title}\nAuthor: {author}\nSummary:\n{summary}")
        elif link != 'N/A':
            return (f"{title}, {link}", f"Title: {title}\nLink: {link}\nSummary:\n{summary}")
        else:
            return (f"{title}", f"Title: {title}\nSummary:\n{summary}")
    
    except Exception as e:
        return f"Error processing document: {str(e)}"

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


    retriever = ArxivRetriever(load_max_docs=3, get_full_documents=True)

    try:

        docs = await asyncio.to_thread(retriever.invoke, query)

        if not docs:
            return {'context' : "No relevant Arxiv Research Papers were found", 'citations' : []}

        tasks = [create_summary(query, doc) for doc in docs]

        results = await asyncio.gather(*tasks)         

        context = "\n\n".join([res[1] for res in results])
        citations = [res[0] for res in results] 

        print(f"Used for context: {context}\n\n {'\n'.join(citations)}")
        return {'context' : context, 'citations' : citations}

    except Exception as e:
        return {'context' : f"Error during Arxiv search: {str(e)}", 'citations' : []}

@tool
async def search_duckduckgo(query: str)->dict[str, Any]:
    '''"""
    Выполняет веб-поиск с помощью DuckDuckGo для поиска общей информации и ресурсов.
    
    WHEN TO USE:
    Используйте этот инструмент, когда пользователь задает вопросы, которые требуют 
    общих знаний, текущих событий или ресурсов, недоступных во внутренней базе знаний по химии.
    
    ARGS:
    - query: Поисковый запрос. Формулируйте его как ключевые слова, а не полный вопрос.
             Например, вместо "Что такое машинное обучение?" используйте "определение машинного обучения".
    
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
    results = wrapper.results(query, max_results=4)

    documents = [Document(page_content=doc['snippet'],
                          metadata={"title": doc["title"], "source": doc["link"]}) for doc in results]
    
    tasks = [create_summary(query, doc) for doc in documents]

    results = await asyncio.gather(*tasks)

    context = "\n\n".join([res[1] for res in results])
    citations = [res[0] for res in results] 

    print(f"Used for context: {context}\n\n {'\n'.join(citations)}")
    return {'context' : context, 'citations' : citations}


TOOLS = [search_chemistry_collection, search_arxiv, search_duckduckgo]
tools_by_name = {tool.name: tool for tool in TOOLS}