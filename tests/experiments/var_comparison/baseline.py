# baseline.py
import asyncio
import time
import polars as pl
from collections import deque
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from logger import logger
from rate_limiter import RateLimitedQueue


async def run_stage_baseline(questions, tunnel_url: str):
    """
    tunnel_url: например "http://localhost:11434"
    """
    queue = RateLimitedQueue(requests_per_minute=1000)

    llm = ChatOllama(base_url=tunnel_url.rstrip("/"),
        model="llama3.1-64k", temperature=0.1)
    
    results = []
    for i, item in enumerate(questions):
        q_id = item["id"]
        text = item["question"]
        logger.info(f"[Baseline {q_id}/25] {item['category']}")

        start = time.time()
        try:
            res = await queue.execute(
                llm.ainvoke([HumanMessage(content=text)])
            )
            ans = res.content
            lat = time.time() - start
            logger.info(f"✅ {lat:.2f}s")
        except Exception as e:
            logger.error(f"❌ Ошибка {q_id}: {e}")
            ans, lat = f"Error: {e}", 0

        results.append({
            "id": q_id,
            "category": item["category"],
            "base_latency": round(lat, 2),
            "base_answer": ans
        })

        if (i + 1) % 5 == 0:
            pl.DataFrame(results).write_parquet("baseline_results_temp.parquet")
            logger.info(f"💾 Сохранено {i+1}/25")

    return results