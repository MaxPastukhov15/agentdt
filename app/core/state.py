from pydantic import BaseModel
from langgraph.graph.message import BaseMessage
from typing import Annotated, Sequence, Optional
from langgraph.graph.message import add_messages

def add_links(old_list: list, new_list: list) -> list:
    return list(set(old_list + new_list))

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    citation_links: Optional[Annotated[Sequence[str], add_links]]

