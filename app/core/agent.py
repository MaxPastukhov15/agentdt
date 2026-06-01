import logging
from typing import Optional

from app.config.config import settings
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages
)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph


from app.core.prompts import SYSTEM_PROMPT
from app.core.state import AgentState
from app.core.tools import TOOLS
from app.utils.model_context_window import get_context_limit
graph_logger = logging.getLogger("graph")


class Agent:
    def __init__(self, checkpointer, tools: list = TOOLS, model_name: str = settings.main_model, 
                 tunnel_url: Optional[str] = None, 
                 system_prompt: str = SYSTEM_PROMPT) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

        self.model_name = model_name

        if tunnel_url:
            self.base_llm = ChatOllama(
                base_url=tunnel_url.rstrip("/"),
                model=self.model_name,
                temperature=0.1)
        else:
            self.base_llm = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=self.model_name,
                temperature=0.1,
                extra_body={"sort": "provider", "route": "fallback"},
            )

        self.llm = self.base_llm.bind_tools(tools)
        self.checkpointer = checkpointer
        self.system_prompt = system_prompt

        try:
            limit = get_context_limit(self.model_name)
        except Exception:
            limit = 64000

        self.trimmer = trim_messages(
            max_tokens=int(limit * 0.65),
            strategy="last",
            token_counter=self.base_llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        self.app = self.make()
    
    async def __summarize_history(self, state: AgentState) -> str:
        """Вспомогательный метод для генерации суммаризации"""
        messages = list(state.messages)
        existing_summary = state.summary
        
        summary_prompt = (
            f"Текущее краткое содержание: {existing_summary}\n\n"
            f"Обнови краткое содержание беседы, включив новые важные детали из сообщений выше. "
            f"Пиши максимально сжато, сохраняя только факты и суть запросов пользователя."
        )
        
        response = await self.llm.ainvoke(messages + [SystemMessage(content=summary_prompt)])
        return str(response.content)

    def __entry_node(self, state: AgentState) -> dict:
        messages = state.messages
        last_msg: BaseMessage = messages[-1]

        graph_logger.info(f"Вы: {last_msg}\n")

        if last_msg.content.strip("~@)(><,&'*/!.\\|$;:-_^%#№ ") == "":
            graph_logger.info("No input provided")
            return {"messages": [AIMessage(content="Извините, я не получил текст вашего запроса.")], "step_count": 0}

        return {"messages": [], "step_count": 0, "citation_links": []}

    async def __llm_node(self, state: AgentState) -> dict:
        messages = list(state.messages)
        limit = get_context_limit(self.model_name)
        current_tokens = self.base_llm.get_num_tokens(str(messages)) 
        curr_summary = state.summary

        update = {"messages": [], "step_count": state.step_count, "summary": curr_summary}

        if current_tokens > (limit * 0.65):
            graph_logger.info("Context is filled on 70%. Strarting summarization...")
            curr_summary = await self.__summarize_history(state)
            update["summary"] = curr_summary

        enriched_system_prompt = self.system_prompt
        if curr_summary:
            enriched_system_prompt += f"\n\nКраткий контекст предыдущих бесед: {curr_summary}"

        trimmed_messages = self.trimmer.invoke(messages)


        if state.step_count >= settings.max_steps:
            messages.append(
                SystemMessage(
                    content="""ВНИМАНИЕ: Лимит поиска исчерпан. Сформулируй финальный ответ на
                основе имеющейся информации, не вызывай новые инструменты."""
                )
            )

        update["citation_links"] = state.citation_links

        full_input = [SystemMessage(content=enriched_system_prompt)] + trimmed_messages
        response = await self.llm.ainvoke(full_input)

        update["messages"] = [response]

        if response.tool_calls:
            graph_logger.info(f"🔧 Tool calls: {response.tool_calls}\n")

        else:
            graph_logger.info(f"🤖 AI: {response.content}\n")

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
            tool = self.tools_by_name[tool_call["name"]]

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
