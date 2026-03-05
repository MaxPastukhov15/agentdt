from core.agent import Agent
from langchain_core.messages import HumanMessage
from asyncio import run
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
async def main():
    session = px.launch_app()

    try:
        tracer_provider = register(endpoint=os.getenv('PHOENIX_COLLECTOR_ENDPOINT'))
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        graph = Agent().make()

        while True:
            query = await asyncio.to_thread(input, "Вопрос: ")

            if not query.strip():
                break

            answer = await graph.ainvoke({"messages" : [HumanMessage(query)], "citation_links" : None}, 
                          config={"configurable": {"thread_id": "haha12020"}})
    finally:
        session.end()

if __name__ == "__main__":
    run(main())