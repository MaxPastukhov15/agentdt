from app.utils.model_context_window import MODEL_CONTEXT_LIMITS, get_context_limit


def test_get_context_limit_known_model():
    limit = get_context_limit("google/gemma-4-26b-a4b-it:free")
    assert limit == 262100


def test_get_context_limit_unknown_model_returns_4096():
    limit = get_context_limit("unknown-model-12345")
    assert limit == 4096


def test_get_context_limit_llama_64k():
    limit = get_context_limit("llama3.1-64k")
    assert limit == 64000


def test_get_context_limit_nemotron():
    limit = get_context_limit("nvidia/llama-3.1-nemotron-70b-instruct")
    assert limit == 131072


def test_model_context_limits_dict_is_not_empty():
    assert len(MODEL_CONTEXT_LIMITS) > 0
