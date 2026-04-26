from config.config import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI


def create_summarizer() -> RunnableSerializable:
    llm_summarizer = ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=settings.summarization_model,
        temperature=0.0,
        max_completion_tokens=2000,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Действуй как аналитик. Подготовь сжатый отчет (summary) на основе предоставленного контекста.
        Этот отчет будет использован другой нейросетью, поэтому пиши максимально информативно, без вводных фраз. 
        Особо концентрируй внимание на выводах в статье, если это информация из интернета со ссылкой просто отыщи ответ на вопрос""",
            ),
            (
                "human",
                "Контекст: {context}\n\nЗадача: {query}\n\nФормат: тезис, bullet points, вывод (100-200 слов).",
            ),
        ]
    )

    return prompt | llm_summarizer.with_retry(wait_exponential_jitter=True, stop_after_attempt=3) | StrOutputParser()
