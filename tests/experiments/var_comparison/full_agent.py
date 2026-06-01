import asyncio
import time
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from app.core.agent import Agent
from app.core.tools import TOOLS
from logger import logger

async def run_stage_agent(questions, tunnel_url: str):
    """Этап 2: Агент с RAG + Tools"""
    checkpointer = MemorySaver()

    agent_instance = Agent(checkpointer=checkpointer, tools=TOOLS, tunnel_url=tunnel_url, model_name="llama3.1-64k")
    results = []

    for item in questions:
        q_id = item["id"]
        text = item["question"]
        logger.info(f"[Agent {q_id}/25] {item['category']}")

        initial_state = {
            "messages": [HumanMessage(content=text)],
            "citation_links": [],
            "step_count": 0
        }
        config = {"configurable": {"thread_id": f"q_{q_id}"}}
        
        start = time.time()
        try:
            final_state = await agent_instance.app.ainvoke(initial_state, config=config)
            ans = final_state["messages"][-1].content
            lat = time.time() - start
            steps = final_state.get("step_count", 0)
            cits = final_state.get("citation_links", [])
        except Exception as e:
            logger.error(f"Ошибка в agent {q_id}: {e}")
            ans, lat, steps, cits = f"Error: {e}", 0, 0, []

        results.append({
            "id": q_id,
            "agent_latency": round(lat, 2),
            "agent_steps": steps,
            "citations": len(cits),
            "agent_answer": ans
        })
        
        await asyncio.sleep(2)  # Rate limiting
    
    return results