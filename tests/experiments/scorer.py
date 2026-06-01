import argparse
import json
import asyncio
import polars as pl
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from logger import logger

class Scores(BaseModel):
    baseline: int = Field(description="Score for baseline answer (1-5)")
    rag: int = Field(description="Score for RAG answer (1-5)")
    search: int = Field(description="Score for Search answer (1-5)")
    agent: int = Field(description="Score for Agent answer (1-5)")
    reasoning: str = Field(description="Brief explanation of score differences")

SYSTEM_PROMPT = """You are an expert chemistry professor evaluating answers to chemistry questions.
Score each answer from 1 to 5 based on the following detailed criteria:

SCORING RUBRIC:
- 5 (Excellent): Fully correct, complete, and thorough. Covers all key concepts with accurate details. Well-structured and precise. No factual errors.
- 4 (Good): Mostly correct with minor gaps. May miss some nuance or detail but core answer is accurate and complete.
- 3 (Fair): Partially correct. Contains accurate information but has notable gaps, imprecise language, or missing key elements.
- 2 (Poor): Significant errors or largely incomplete. Core concepts may be misunderstood or major elements missing.
- 1 (Very Poor): Incorrect, irrelevant, or fundamentally misunderstands the question.

EVALUATION DIMENSIONS (consider all):
1. FACTUAL ACCURACY: Are all stated facts chemically correct?
2. COMPLETENESS: Does the answer address all parts of the question thoroughly?
3. DEPTH OF EXPLANATION: Does it explain the "why" and "how", not just the "what"?
4. STRUCTURE & CLARITY: Is the answer well-organized and easy to follow?
5. USE OF TOOLS/SOURCES: For RAG/Agent answers, does it effectively use retrieved context or tools?

IMPORTANT RULES:
- Evaluate each answer INDEPENDENTLY on its merits, not relative to others.
- Do NOT penalize for verbosity. Detailed answers are encouraged.
- Do NOT penalize for including tool output, citations, or reasoning steps.
- For Agent answers: Credit them for using tools effectively to find accurate information.
- For RAG answers: Credit them for incorporating retrieved context appropriately.
- Focus on: Is this answer correct and complete? Would it fully answer the student's question?

RESPOND WITH VALID JSON: {"baseline": int, "rag": int, "search": int, "agent": int, "reasoning": "brief explanation of key differences"}
"""

OUTPUT_DIR = "benchmark_results"

async def run_scorer(tunnel_url: str, model: str = "gemma2:27b", temperature: float = 0.1):
    comparison_path = f"{OUTPUT_DIR}/benchmark_comparison.jsonl"
    questions_path = "agentdt/tests/experiments/questions_exper.json"
    output_path = f"{OUTPUT_DIR}/benchmark_scored.jsonl"

    df = pl.read_ndjson(comparison_path)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f).get("data", [])}

    logger.info(f"Using judge model: {model} (temperature={temperature})")

    llm = ChatOllama(
        base_url=tunnel_url.rstrip("/"),
        model=model,
        temperature=temperature,
    )
    parser = JsonOutputParser(pydantic_object=Scores)
    chain = llm | parser

    scored = []
    for row in df.to_dicts():
        q_id = row["id"]
        q = questions.get(q_id, {})
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        difficulty = q.get("difficulty", "unknown")

        user_prompt = (
            f"Question: {question_text}\n"
            f"Category: {category}, Difficulty: {difficulty}\n\n"
            f"--- Baseline answer ---\n{row.get('base_answer', '')}\n\n"
            f"--- RAG answer ---\n{row.get('rag_answer', '')}\n\n"
            f"--- Search answer ---\n{row.get('search_answer', '')}\n\n"
            f"--- Agent answer ---\n{row.get('agent_answer', '')}\n\n"
            f"Score each 1-5."
        )

        logger.info(f"Scoring question {q_id} ({category})...")
        scores = None
        for attempt in range(3):
            try:
                res = await chain.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ])
                scores = Scores(**res)
                logger.info(f"  -> baseline={scores.baseline}, rag={scores.rag}, search={scores.search}, agent={scores.agent}")
                break
            except Exception as e:
                logger.warning(f"  Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2)

        if scores is None:
            logger.error(f"Failed to score {q_id} after 3 attempts, skipping.")
            continue

        scored.append({
            "id": q_id,
            "category": category,
            "difficulty": difficulty,
            "score_baseline": scores.baseline,
            "score_rag": scores.rag,
            "score_search": scores.search,
            "score_agent": scores.agent,
            "reasoning": scores.reasoning,
            "base_latency": row["base_latency"],
            "rag_latency": row["rag_latency"],
            "search_latency": row["search_latency"],
            "agent_latency": row["agent_latency"],
            "rag_steps": row.get("rag_steps", 0),
            "search_steps": row.get("search_steps", 0),
            "agent_steps": row.get("agent_steps", 0),
        })

    if scored:
        scored_df = pl.DataFrame(scored)
        scored_df.write_ndjson(output_path)
        logger.info(f"Scored results saved → {output_path}")

        summary = scored_df.group_by(["category"]).agg([
            pl.col("score_baseline").mean().alias("avg_baseline"),
            pl.col("score_rag").mean().alias("avg_rag"),
            pl.col("score_search").mean().alias("avg_search"),
            pl.col("score_agent").mean().alias("avg_agent"),
        ])
        print("\n" + "="*60)
        print("SCORE SUMMARY BY CATEGORY")
        print("="*60)
        print(summary)

        overall = scored_df.select([
            pl.col("score_baseline").mean().alias("overall_baseline"),
            pl.col("score_rag").mean().alias("overall_rag"),
            pl.col("score_search").mean().alias("overall_search"),
            pl.col("score_agent").mean().alias("overall_agent"),
        ])
        print("\nOVERALL SCORES:")
        print(overall)
    else:
        logger.warning("No scored results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score benchmark answers with LLM")
    parser.add_argument("--tunnel-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma2:27b",
                        help="Judge model to use (default: gemma2:27b)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for judge LLM (default: 0.1)")
    args = parser.parse_args()
    asyncio.run(run_scorer(args.tunnel_url, args.model, args.temperature))
