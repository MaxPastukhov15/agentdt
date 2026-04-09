from typing import Annotated, Optional, Sequence

from langgraph.graph.message import BaseMessage, add_messages
from pydantic import BaseModel, Field


def add_links(left: Sequence[str], right: Sequence[str]) -> Sequence[str]:
    if not left:
        left = []
    if not right:
        right = []
    return list(left) + list(right)


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    citation_links: Optional[Annotated[Sequence[str], add_links]] = Field(
        default_factory=list
    )
