import logging

from config.config import settings
from core.prompts import SYSTEM_PROMPT
from core.state import AgentState
from core.tools import TOOLS, tools_by_name
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

graph_logger = logging.getLogger("graph")


class Agent:
    def __init__(self, checkpointer) -> None:
        self.llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=settings.main_model,
            temperature=0.1,
            max_retries=3,
            timeout=60,
        ).bind_tools(TOOLS)
        self.checkpointer = checkpointer
        self.app = self.make()

    def __entry_node(self, state: AgentState) -> dict:
        messages = state.messages
        last_msg: BaseMessage = messages[-1]

        graph_logger.info(f"Вы: {last_msg}\n")

        if last_msg.content.strip("~@)(><,&'*/!.\\|$;:-_^%#№ ") == "":
            graph_logger.info("No input provided")
            return {"messages": [AIMessage(content="Извините, я не получил текст вашего запроса.")], "step_count": 0}

        return {"messages": [], "step_count": 0}

    async def __llm_node(self, state: AgentState) -> dict:
        messages = list(state.messages)
        update = {"messages": [], "step_count": state.step_count}

        if state.step_count >= settings.max_steps:
            messages.append(
                SystemMessage(
                    content="""ВНИМАНИЕ: Лимит поиска исчерпан. Сформулируй финальный ответ на
                основе имеющейся информации, не вызывай новые инструменты."""
                )
            )

        if isinstance(messages[-1], HumanMessage):
            update["citation_links"] = []
        else:
            update["citation_links"] = state.citation_links

        response = await self.llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)

        graph_logger.info(f"-----\n{[SystemMessage(content=SYSTEM_PROMPT)] + messages}\n----")

        update["messages"] = [response]

        if response.tool_calls:
            graph_logger.info(f"🔧 Tool calls: {response.tool_calls}\n")

        else:
            graph_logger.info(f"🤖 ИИ: {response.content}\n")

        update["step_count"] += 1

        return update

    def __should_continue(self, state: AgentState) -> str:
        last_msg = state.messages[-1]

        if state.step_count >= settings.max_steps:
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                return "end"

            graph_logger.warning(f"""Лимит шагов ({state.step_count}) 
                                 достигнут. Финализируем ответ...""")
            return "final_answer"

        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "continue"
        else:
            return "end"

    async def __tool_node(self, state: AgentState) -> dict:
        result, citations = [], []
        for tool_call in state.messages[-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]

            try:
                observation = await tool.ainvoke(tool_call["args"])

                if isinstance(observation, dict):
                    context = observation.get("context", str(observation))
                    cits = observation.get("citations", [])
                else:
                    context = str(observation)
                    cits = []

                result.append(ToolMessage(content=context, tool_call_id=tool_call["id"]))

                if isinstance(observation.get("citations"), list):
                    citations.extend(cits)
                else:
                    citations.append(str(cits))

            except Exception as e:
                error_msg = f"Ошибка при вызове инструмента: {str(e)}"
                result.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))

        return {"messages": result, "citation_links": state.citation_links + citations, "step_count": state.step_count + 1}

    def make(self):
        graph = StateGraph(AgentState)

        graph.add_node("entry", self.__entry_node)
        graph.add_node("llm_node", self.__llm_node)
        graph.add_node("tool_node", self.__tool_node)

        graph.set_entry_point("entry")
        graph.add_edge("entry", "llm_node")

        graph.add_conditional_edges(
            "llm_node",
            self.__should_continue,
            {
                "end": END,
                "final_answer": "llm_node",
                "continue": "tool_node",
            },
        )

        graph.add_edge("tool_node", "llm_node")

        self.app = graph.compile(checkpointer=self.checkpointer)
        return self.app
