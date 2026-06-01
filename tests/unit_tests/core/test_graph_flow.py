from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("app.core.agent.settings") as mock:
        mock.max_steps = 4
        mock.main_model = "test-model"
        mock.openrouter_api_key = "sk-test-key"
        yield mock


@pytest.fixture
def mock_llm():
    with (
        patch("app.core.agent.ChatOpenAI") as MockChatOpenAI,
        patch("app.core.agent.ChatOllama") as MockChatOllama,
        patch("app.core.agent.get_context_limit") as mock_get_limit,
        patch("app.core.agent.trim_messages") as mock_trimmer,
    ):
        mock_get_limit.return_value = 64000
        instance = MagicMock()
        instance.get_num_tokens.return_value = 1000
        instance.bind_tools.return_value = instance
        instance.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        MockChatOpenAI.return_value = instance
        mock_trimmer.return_value.invoke.side_effect = lambda msgs: msgs
        yield instance


def create_agent(**kwargs):
    from app.core.agent import Agent

    checkpointer = MemorySaver()
    defaults = {"checkpointer": checkpointer, "tools": [], "model_name": "test-model"}
    defaults.update(kwargs)
    return Agent(**defaults)


class TestGraphFlow:
    @pytest.mark.asyncio
    async def test_direct_answer_ends(self, mock_llm):
        mock_llm.ainvoke.return_value = AIMessage(content="Direct answer.")
        agent = create_agent()

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Hello")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        assert isinstance(result, dict)
        assert result["messages"][-1].content == "Direct answer."
        assert result["step_count"] == 1
        assert result["citation_links"] == []

    @pytest.mark.asyncio
    async def test_tool_call_then_answer_flow(self, mock_llm):
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.ainvoke = AsyncMock(
            return_value={"context": "tool data", "citations": ["src1"]}
        )

        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "test_tool", "args": {}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage(content="Final answer based on tool data."),
        ]

        agent = create_agent()
        agent.tools_by_name = {"test_tool": mock_tool}

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Search")]},
            config={"configurable": {"thread_id": "t2"}},
        )

        assert result["messages"][-1].content == "Final answer based on tool data."
        assert result["step_count"] >= 2
        assert "src1" in result["citation_links"]

    @pytest.mark.asyncio
    async def test_empty_input_entry_node_message_preserved(self, mock_llm):
        agent = create_agent()

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="")]},
            config={"configurable": {"thread_id": "t3"}},
        )

        messages = result["messages"]
        assert len(messages) >= 2
        assert any("не получил" in m.content.lower() for m in messages)
        assert result["step_count"] == 1

    @pytest.mark.asyncio
    async def test_max_steps_reached_forces_final_answer(self, mock_llm):
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.ainvoke = AsyncMock(
            return_value={"context": "data", "citations": []}
        )

        mock_llm.ainvoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="Final after limit."),
        ]

        agent = create_agent()
        agent.tools_by_name = {"test_tool": mock_tool}

        with patch("app.core.agent.settings.max_steps", 2):
            result = await agent.app.ainvoke(
                {"messages": [HumanMessage(content="Do research")]},
                config={"configurable": {"thread_id": "t4"}},
            )

        assert "Final" in result["messages"][-1].content
        assert result["step_count"] >= 2

    @pytest.mark.asyncio
    async def test_tool_error_propagates(self, mock_llm):
        mock_tool = MagicMock()
        mock_tool.name = "bad_tool"
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("Tool crashed"))

        mock_llm.ainvoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "bad_tool", "args": {}, "id": "c2", "type": "tool_call"}]),
            AIMessage(content="I handled the error."),
        ]

        agent = create_agent()
        agent.tools_by_name = {"bad_tool": mock_tool}

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Run tool")]},
            config={"configurable": {"thread_id": "t5"}},
        )

        assert "handled" in result["messages"][-1].content.lower()

    @pytest.mark.asyncio
    async def test_citation_links_preserved_across_llm_calls(self, mock_llm):
        mock_tool = MagicMock()
        mock_tool.name = "cite_tool"
        mock_tool.ainvoke = AsyncMock(
            return_value={"context": "data", "citations": ["paper1", "paper2"]}
        )

        mock_llm.ainvoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "cite_tool", "args": {}, "id": "c3", "type": "tool_call"}]),
            AIMessage(content="Answer with citations."),
        ]

        agent = create_agent()
        agent.tools_by_name = {"cite_tool": mock_tool}

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Find papers")]},
            config={"configurable": {"thread_id": "t6"}},
        )

        assert "paper1" in result["citation_links"]
        assert "paper2" in result["citation_links"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(self, mock_llm):
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool_a"
        mock_tool1.ainvoke = AsyncMock(return_value={"context": "A data", "citations": []})
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool_b"
        mock_tool2.ainvoke = AsyncMock(return_value={"context": "B data", "citations": []})

        mock_llm.ainvoke.side_effect = [
            AIMessage(content="", tool_calls=[{"name": "tool_a", "args": {}, "id": "ca", "type": "tool_call"}]),
            AIMessage(content="", tool_calls=[{"name": "tool_b", "args": {}, "id": "cb", "type": "tool_call"}]),
            AIMessage(content="Combined answer."),
        ]

        agent = create_agent()
        agent.tools_by_name = {"tool_a": mock_tool1, "tool_b": mock_tool2}

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Use tools")]},
            config={"configurable": {"thread_id": "t7"}},
        )

        assert result["messages"][-1].content == "Combined answer."
        assert mock_tool1.ainvoke.called
        assert mock_tool2.ainvoke.called

    @pytest.mark.asyncio
    async def test_graph_is_compiled_state_graph(self, mock_llm):
        agent = create_agent()
        assert isinstance(agent.app, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_thread_isolation(self, mock_llm):
        mock_llm.ainvoke.return_value = AIMessage(content="Answer.")

        agent = create_agent()

        r1 = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="First")]},
            config={"configurable": {"thread_id": "ta"}},
        )
        r2 = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="Second")]},
            config={"configurable": {"thread_id": "tb"}},
        )

        assert r1["messages"][-1].content == "Answer."
        assert r2["messages"][-1].content == "Answer."

    @pytest.mark.asyncio
    async def test_special_chars_entry_node_message_preserved(self, mock_llm):
        agent = create_agent()

        result = await agent.app.ainvoke(
            {"messages": [HumanMessage(content="~@#$%^&")]},
            config={"configurable": {"thread_id": "t8"}},
        )

        messages = result["messages"]
        assert len(messages) >= 2
        assert any("не получил" in m.content.lower() for m in messages)
        assert result["step_count"] == 1
