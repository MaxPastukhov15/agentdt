
MODEL_CONTEXT_LIMITS = {
    # Gemma серии
    "google/gemma-4-26b-a4b-it:free": 262100,
    "google/gemma-4-31b-it:free": 262100,
    
    # NVIDIA (Nemotron)
    "nvidia/nemotron-3-super-120b-a12b:free": 262100,
    "nvidia/llama-3.1-nemotron-70b-instruct": 131072,
    
    # Другие популярные на OpenRouter
    "minimax/minimax-m2.5:free": 196600,
    "z-ai/glm-4.5-air:free": 131072,
    "baidu/cobuddy:free": 131100,
    "llama3.1-64k": 64000
}

def get_context_limit(model_name: str) -> int:
    # Возвращаем лимит из словаря или 4096 как безопасный минимум
    return MODEL_CONTEXT_LIMITS.get(model_name, 4096)
