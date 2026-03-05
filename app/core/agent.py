from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage, BaseMessage
from core.state import AgentState
from langchain_openai import ChatOpenAI
from core.prompts import SYSTEM_PROMPT
from langgraph.checkpoint.memory import InMemorySaver
from core.tools import TOOLS, tools_by_name
import os
import re

load_dotenv()

class Agent:

    def __init__(self) -> None:
        self.llm = ChatOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", 
                 model="stepfun/step-3.5-flash:free", temperature=0.1, ).bind_tools(TOOLS)
        self.checkpointer = InMemorySaver()

    def __user_input_node(self, state : AgentState)-> dict:

        messages = state.messages
        last_msg: BaseMessage = messages[-1]
    
        if isinstance(last_msg, HumanMessage) and last_msg.content.strip("~@)(><,&'*/!.\\|$;:-_^%#№ ") != "":
            user_input = re.sub(pattern=r"[^А-Яа-яA-Za-z0-9!?.,;:()\ '-_]", repl='', string=last_msg.content).strip()

            print(f"-----\n{messages}\n----")

            print(f"Вы: {user_input}\n")

            if len(messages) == 1:
                id_to_remove = messages[0].id
                message = [RemoveMessage(id=id_to_remove), SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
                return {"messages" : message}
                
            else:
                id_to_remove = messages[-1].id
                message = [RemoveMessage(id=id_to_remove), HumanMessage(content=user_input)]
                return {"messages" : message}
        else:
            print("No input provided")
            return {"messages" : [SystemMessage("No input")]}

    async def __llm_node(self, state : AgentState) -> dict:
        messages = list(state.messages)  

        response = await self.llm.ainvoke(messages)
        if response.tool_calls:
            print(f"🔧 Tool calls: {response.tool_calls}\n")
        else:
            print(f"🤖 ИИ: {response.content}\n")

        return {"messages" : [response]}

    def __should_continue(self, state : AgentState) -> str:

        last_msg = state.messages[-1]

        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "continue"
        else:
            return "end"
    
    async def __tool_node(self, state: AgentState) -> dict:
        result, citations = [], []
        for tool_call in state.messages[-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            result.append(ToolMessage(content=observation['context'], 
                                      tool_call_id=tool_call['id']))
            citations.append(observation['citations'])
        return {"messages" : result, "citations" : citations}

    def make(self):
        graph  = StateGraph(AgentState)

        graph.add_node("user_input_node", self.__user_input_node)
        graph.add_node("llm_node", self.__llm_node)
        graph.add_node("tool_node", self.__tool_node)

        graph.add_edge(START, "user_input_node")
        graph.add_edge("user_input_node", "llm_node")
        
        graph.add_conditional_edges("llm_node", self.__should_continue, {
            "end" : END,
            "continue" : "tool_node",
            
        })

        graph.add_edge("tool_node", "llm_node")


        app = graph.compile(checkpointer=self.checkpointer)
        return app
    
