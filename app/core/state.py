from pydantic import BaseModel
from langgraph.graph.message import BaseMessage
from typing import Annotated, Sequence, Optional
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    citation_links: Optional[Sequence[str]]

