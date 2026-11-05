from typing import Any, List, Optional, TypedDict, NotRequired, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from src.application.models.query_agent_models import QueryAgentOutput
from src.application.models.moderation_agent_models import ModerationAgentOutput
from langgraph.graph.message import add_messages
from src.application.models.parser_agent_models import ParserAgentOutput
from src.application.models.sql_agent_models import SQLAgentOutput


class AgentState(TypedDict):
    """Состояние графа, разделяемое между всеми агентами"""
    messages: NotRequired[Annotated[List[BaseMessage], add_messages]]
    moderation_output: NotRequired[ModerationAgentOutput]
    query_agent_output: NotRequired[QueryAgentOutput]
    parser_output: NotRequired[ParserAgentOutput]
    sql_agent_output: NotRequired[SQLAgentOutput]