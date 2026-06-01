from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.state import AgentState


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
        mock_llm_instance = AsyncMock()
        mock_llm_instance.get_num_tokens.return_value = 1000
        mock_llm_instance.ainvoke.return_value = AIMessage(content="Test response")
        MockChatOpenAI.return_value = mock_llm_instance

        mock_trimmer.return_value.invoke.side_effect = lambda msgs: msgs

        yield mock_llm_instance


def create_agent(**kwargs):
    from app.core.agent import Agent

    checkpointer = MemorySaver()
    defaults = {"checkpointer": checkpointer, "tools": [], "model_name": "test-model"}
    defaults.update(kwargs)
    agent = Agent(**defaults)
    return agent


class TestEntryNode:
    def test_entry_node_valid_input(self):
        agent = create_agent()
        state = AgentState(messages=[HumanMessage(content="Привет")])
        result = agent._Agent__entry_node(state)
        assert result["step_count"] == 0
        assert result["citation_links"] == []
        assert result["messages"] == []

    def test_entry_node_empty_input(self):
        agent = create_agent()
        state = AgentState(messages=[HumanMessage(content="")])
        result = agent._Agent__entry_node(state)
        assert len(result["messages"]) == 1
        assert "не получил" in result["messages"][0].content.lower()

    def test_entry_node_only_special_chars(self):
        agent = create_agent()
        state = AgentState(messages=[HumanMessage(content="@#$%^&")])
        result = agent._Agent__entry_node(state)
        assert len(result["messages"]) == 1
        assert "не получил" in result["messages"][0].content.lower()


class TestShouldContinue:
    def test_should_continue_tool_call(self):
        agent = create_agent()
        state = AgentState(
            messages=[AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])],
            step_count=1,
        )
        result = agent._Agent__should_continue(state)
        assert result == "continue"

    def test_should_continue_end(self):
        agent = create_agent()
        state = AgentState(
            messages=[AIMessage(content="Final answer")],
            step_count=1,
        )
        result = agent._Agent__should_continue(state)
        assert result == "end"

    def test_should_continue_max_steps_reached_with_tool_call(self):
        agent = create_agent()
        with patch("app.core.agent.settings.max_steps", 2):
            state = AgentState(
                messages=[AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])],
                step_count=2,
            )
            result = agent._Agent__should_continue(state)
            assert result == "final_answer"

    def test_should_continue_max_steps_reached_no_tool_call(self):
        agent = create_agent()
        with patch("app.core.agent.settings.max_steps", 2):
            state = AgentState(
                messages=[AIMessage(content="Done")],
                step_count=2,
            )
            result = agent._Agent__should_continue(state)
            assert result == "end"

    def test_should_continue_human_message(self):
        agent = create_agent()
        state = AgentState(
            messages=[HumanMessage(content="Question")],
            step_count=1,
        )
        result = agent._Agent__should_continue(state)
        assert result == "end"


class TestMakeGraph:
    def test_make_creates_graph_with_nodes(self):
        agent = create_agent()
        from langgraph.graph.state import CompiledStateGraph

        graph = agent.make()
        assert isinstance(graph, CompiledStateGraph)

    def test_make_graph_has_correct_nodes(self):
        agent = create_agent()
        graph = agent.make()
        node_names = set(graph.nodes.keys())
        assert "entry" in node_names
        assert "llm_node" in node_names
        assert "tool_node" in node_names

    def test_tool_node_success(self):
        agent = create_agent(tools=[])
        agent.tools_by_name = {
            "test_tool": MagicMock(ainvoke=AsyncMock(return_value={"context": "tool result", "citations": ["cit1"]}))
        }
        state = AgentState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "test_tool", "args": {"query": "test"}, "id": "call_1"}],
                )
            ],
            step_count=0,
            citation_links=[],
        )
        result = asyncio_run(agent._Agent__tool_node(state))
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert "tool result" in result["messages"][0].content
        assert "cit1" in result["citation_links"]

    def test_tool_node_error(self):
        agent = create_agent(tools=[])
        agent.tools_by_name = {
            "test_tool": MagicMock(ainvoke=AsyncMock(side_effect=Exception("Tool failed")))
        }
        state = AgentState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}],
                )
            ],
            step_count=0,
            citation_links=[],
        )
        result = asyncio_run(agent._Agent__tool_node(state))
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert "Ошибка" in result["messages"][0].content


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
