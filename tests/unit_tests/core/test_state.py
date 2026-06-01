from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from app.core.state import AgentState


def test_agent_state_defaults():
    state = AgentState(messages=[])
    assert state.messages == []
    assert state.citation_links == []
    assert state.step_count == 0
    assert state.summary == ""


def test_agent_state_with_messages():
    msg = HumanMessage(content="Hello")
    state = AgentState(messages=[msg])
    assert len(state.messages) == 1
    assert state.messages[0].content == "Hello"


def test_agent_state_citation_links():
    state = AgentState(messages=[], citation_links=["src1", "src2"])
    assert state.citation_links == ["src1", "src2"]


def test_agent_state_step_count():
    state = AgentState(messages=[], step_count=3)
    assert state.step_count == 3


def test_agent_state_summary():
    state = AgentState(messages=[], summary="Brief summary")
    assert state.summary == "Brief summary"


def test_agent_state_messages_use_add_messages_annotation():
    assert hasattr(AgentState.model_fields["messages"], "annotation")
