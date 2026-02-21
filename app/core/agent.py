from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from core.state import AgentState
from langchain_openai import ChatOpenAI
from core.prompts import SYSTEM_PROMPT
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from core.tools import TOOLS
import os
import re

load_dotenv()

class Agent:

    def __init__(self) -> None:
        self.llm = ChatOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", 
                 model="arcee-ai/trinity-large-preview:free", temperature=0.1, )
        self.__tool_node = ToolNode(tools=TOOLS)
        self.checkpointer = InMemorySaver()

    def __user_input_node(self, state : AgentState)-> dict:

        messages = state.messages
        last_msg = messages[-1]
    
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



    def __llm_node(self, state : AgentState) -> dict:
        messages = list(state.messages)  

        response = self.llm.invoke(messages)

        print(f"ИИ: {response.content}\n")

        return {"messages" : [AIMessage(response.content)]}

    def __should_continue(self, state : AgentState) -> str:

        last_msg = state.messages[-1]

        if isinstance(last_msg, AIMessage) or isinstance(last_msg, ToolMessage) or last_msg.content == "No input":
            return "end"
    
        else:
            return "continue" 
    
    def make(self):
        graph  = StateGraph(AgentState)

        graph.add_node("user_input_node", self.__user_input_node)
        graph.add_node("llm_node", self.__llm_node)
        graph.add_node("tool_node", self.__tool_node)
        
        graph.add_conditional_edges("tool_node", self.__should_continue, {
            "end" : END,
            "continue" : "llm_node",
            
        })

        graph.add_edge(START, "user_input_node")
        graph.add_edge("user_input_node", "llm_node")
        graph.add_edge("llm_node", "tool_node")

        app = graph.compile(checkpointer=self.checkpointer)
        return app
    
