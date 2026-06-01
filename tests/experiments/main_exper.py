# main_exper.py
import argparse
import asyncio
import json
import os
import polars as pl
from agentdt.tests.experiments.var_comparison.baseline import run_stage_baseline
from agentdt.tests.experiments.var_comparison.rag import run_stage_rag_only
from agentdt.tests.experiments.var_comparison.search import run_stage_search_only
from agentdt.tests.experiments.var_comparison.full_agent import run_stage_agent
from logger import logger
from app.core.tools import cleanup_repos
async def main():
    parser = argparse.ArgumentParser(description="Run experiment benchmark")
    parser.add_argument("--tunnel-url", default="http://localhost:11434",
                        help="Tunnel URL pointing to Ollama server")
    args = parser.parse_args()
    TUNNEL_URL = args.tunnel_url
    
    dataset_path = "agentdt/tests/experiments/questions_exper.json"
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем датасет
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f).get("data", [])

    logger.info(f"📚 Загружено {len(questions)} вопросов")
    logger.info(f"🌐 Tunnel URL: {TUNNEL_URL}\n")

    # --- STAGE 1: Baseline (LLM only) ---
    baseline_path = f"{output_dir}/baseline_results.jsonl"
    if not os.path.exists(baseline_path):
        logger.info("\n=== STAGE 1: BASELINE (LLM) ===")
        base_results = await run_stage_baseline(questions, TUNNEL_URL)
        pl.DataFrame(base_results).write_ndjson(baseline_path)
        logger.info(f"✅ Baseline завершён: {baseline_path}")
        await asyncio.sleep(10)
    else:
        logger.info("✓ Baseline уже выполнен, пропускаем...")

    # --- STAGE 2: RAG Only ---
    rag_path = f"{output_dir}/rag_only_results.jsonl"
    if not os.path.exists(rag_path):
        logger.info("\n=== STAGE 2: RAG ONLY ===")
        rag_results = await run_stage_rag_only(questions, TUNNEL_URL)
        pl.DataFrame(rag_results).write_ndjson(rag_path)
        logger.info(f"✅ RAG завершён: {rag_path}")
        await asyncio.sleep(10)
    else:
        logger.info("✓ RAG-only уже выполнен, пропускаем...")

    # --- STAGE 3: Search Only ---
    search_path = f"{output_dir}/search_only_results.jsonl"
    if not os.path.exists(search_path):
        logger.info("\n=== STAGE 3: SEARCH ONLY ===")
        search_results = await run_stage_search_only(questions, TUNNEL_URL)
        pl.DataFrame(search_results).write_ndjson(search_path)
        logger.info(f"✅ Search завершён: {search_path}")
        await asyncio.sleep(10)
    else:
        logger.info("✓ Search-only уже выполнен, пропускаем...")

    # --- STAGE 4: Full Agent ---
    agent_path = f"{output_dir}/agent_full_results.jsonl"
    if not os.path.exists(agent_path):
        logger.info("\n=== STAGE 4: FULL AGENT (RAG + SEARCH) ===")
        agent_results = await run_stage_agent(questions, TUNNEL_URL)
        pl.DataFrame(agent_results).write_ndjson(agent_path)
        logger.info(f"✅ Full agent завершён: {agent_path}")
    else:
        logger.info("✓ Full agent уже выполнен, пропускаем...")

    # --- ФИНАЛЬНАЯ СТАТИСТИКА ---
    logger.info("\n=== ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ ===")
    df_base = pl.read_ndjson(baseline_path)
    df_rag = pl.read_ndjson(rag_path)
    df_search = pl.read_ndjson(search_path)
    df_agent = pl.read_ndjson(agent_path)

    final_df = (df_base
                .join(df_rag, on="id", how="inner")
                .join(df_search, on="id", how="inner")
                .join(df_agent, on="id", how="inner"))

    final_path = f"{output_dir}/benchmark_comparison.jsonl"
    final_df.write_ndjson(final_path)
    
    summary = final_df.select([
        pl.col("base_latency").mean().alias("avg_baseline_sec"),
        pl.col("rag_latency").mean().alias("avg_rag_sec"),
        pl.col("search_latency").mean().alias("avg_search_sec"),
        pl.col("agent_latency").mean().alias("avg_full_agent_sec"),
        pl.col("rag_steps").mean().alias("avg_rag_steps"),
        pl.col("search_steps").mean().alias("avg_search_steps"),
        pl.col("agent_steps").mean().alias("avg_agent_steps"),
    ])
    
    print("\n" + "="*60)
    print("ФИНАЛЬНАЯ СТАТИСТИКА")
    print("="*60)
    print(summary)
    print(f"\nПолные результаты: {final_path}")
    logger.info("\n✅ Эксперимент завершён!")
    cleanup_repos()


asyncio.run(main())
