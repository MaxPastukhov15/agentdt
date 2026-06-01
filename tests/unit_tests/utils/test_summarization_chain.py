from unittest.mock import patch

from app.utils.summarization_chain import create_summarizer


def test_create_summarizer_returns_runnable():
    with patch("app.utils.summarization_chain.ChatOpenAI") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.with_retry.return_value = mock_instance
        pipe = create_summarizer()
        assert pipe is not None


def test_create_summarizer_uses_correct_model():
    with patch("app.utils.summarization_chain.ChatOpenAI") as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.with_retry.return_value = mock_instance
        create_summarizer()
        call_kwargs = MockLLM.call_args.kwargs
        assert "model" in call_kwargs
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.0
