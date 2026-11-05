from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Annotated[..., add_messages] говорит LangGraph, что новые сообщения 
    # нужно не перезаписывать, а добавлять в конец списка (историю чата)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Сюда RAG-агент или ToolNode запишут текст найденных мануалов
    relevant_docs: list[str]
    
    # Флаг от агента модерации: безопасен ли и релевантен ли запрос пользователя
    is_safe: bool