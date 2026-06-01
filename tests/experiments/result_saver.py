import os
import polars as pl
from logger import logger

OUTPUT_DIR = "benchmark_results"

def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _path(name: str) -> str:
    return f"{OUTPUT_DIR}/{name}.jsonl"

def save_stage(name: str, results: list[dict]):
    """Save a stage's results to JSONL."""
    _ensure_dir()
    path = _path(name)
    pl.DataFrame(results).write_ndjson(path)
    logger.info(f"  Сохранено {len(results)} записей → {path}")

def load_stage(name: str) -> pl.DataFrame | None:
    """Load a stage's results; returns None if file missing."""
    path = _path(name)
    if not os.path.exists(path):
        return None
    return pl.read_ndjson(path)

def stage_file_exists(name: str) -> bool:
    return os.path.exists(_path(name))

def save_comparison(final_df: pl.DataFrame):
    """Save the merged comparison result."""
    _ensure_dir()
    path = _path("benchmark_comparison")
    final_df.write_ndjson(path)
    logger.info(f"  Сравнение сохранено → {path}")
