from typing import Annotated, List, Sequence

from langgraph.graph.message import BaseMessage, add_messages
from pydantic import BaseModel


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    citation_links: List[str] = []
    step_count: int = 0
    summary: str = ""
