import logging
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from langchain_core.messages import ToolMessage
from src.application.workflows.state import AgentState
from src.application.agents.moderation_agent import moderation_node
from src.application.agents.rag_agent import rag_router_node
from src.application.agents.writer import writer_node
from src.application.tools.vector_search import search_smart_home_docs

logger = logging.getLogger(__name__)

#Создаем стандартный ToolNode, содержащий инструмент векторного поиска Qdrant
tools_node = ToolNode([search_smart_home_docs])

def custom_tools_wrapper(state: AgentState) -> dict:
    """
    Узел явно вызывает инструмент векторного поиска,
    забирает текстовое содержимое документов и гарантированно сохраняет 
    его в relevant_docs, чтобы FastAPI мог отдать его клиенту.
    """
    logger.info("[Workflow] Вызов векторного поиска в Qdrant напрямую...")
    
    # Извлекаем последнее сообщение от RAG-агента, содержащее tool_calls
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    
    # Извлекаем поисковый запрос, который сформулировал RAG-агент
    query_args = tool_call.get("args", {})
    search_query = query_args.get("query", "")
    
    # Вызываем наш инструмент поиска напрямую как обычную функцию
    # search_smart_home_docs.invoke выполнит поиск в Qdrant по всем TXT и PDF
    search_result_text = search_smart_home_docs.invoke({"query": search_query})
    
    logger.info(f"[Workflow] Поиск завершен. Найдено символов контекста: {len(search_result_text)}")
    
    # Формируем обязательное для LangGraph служебное сообщение ToolMessage
    tool_message = ToolMessage(
        content=search_result_text,
        name=tool_call["name"],
        tool_call_id=tool_call["id"]
    )
    
    # Возвращаем обновление состояния: добавляем сообщение в историю 
    return {
        "messages": [tool_message],
        "relevant_docs": [search_result_text]
    }
# Функции для условных переходов
def route_after_moderation(state: AgentState):
    """Решает, куда идти после модератора"""
    is_safe = state.get("is_safe", True)
    logger.info(f"[Routing] Проверка после модерации.is_safe = {is_safe}")
    if not is_safe:
        logger.warning("[Routing] Запрос признан нерелевантным! Направление к Агенту-писателю за отказом.")
        return "writer"
    logger.info("[Routing] Запрос безопасен. Направление к RAG-роутеру.")
    return "rag_router"

def route_after_rag(state: AgentState):
    """Решает, нужно ли вызывать векторный поиск Qdrant"""
    messages = state["messages"]
    last_message = messages[-1]

    # Если последняя подсказка от LLM содержит требования вызвать инструмент
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"[Routing] RAG-агент затребовал вызов инструментов: {last_message.tool_calls}")
        return "vector_search_tools"
    # Если инструментов вызывать не нужно — идем собирать ответ
    logger.info("[Routing] RAG-агенту не нужны инструменты. Направление к Агенту-писателю.")
    return "writer"
# Инициализируем
workflow = StateGraph(AgentState)

# Добавляем все наши узлы в граф
workflow.add_node("moderation", moderation_node)
workflow.add_node("rag_router", rag_router_node)
workflow.add_node("vector_search_tools", custom_tools_wrapper)
workflow.add_node("writer", writer_node)

# Настраиваем связи между узлами.
# Всегда старт с модерации
workflow.add_edge(START, "moderation")

# После модерации проверяем флаг безопасности через условное ребро
workflow.add_conditional_edges(
    "moderation",
    route_after_moderation,
    {
        "writer": "writer",
        "rag_router": "rag_router"
    }
)
# RAG-агент решает: идти в базу Qdrant или сразу к писателю
workflow.add_conditional_edges(
    "rag_router",
    route_after_rag,
    {
        "vector_search_tools": "vector_search_tools",
        "writer": "writer"
    }
)

# После того как поиск в базе завершен, возвращаем управление RAG-агенту
workflow.add_edge("vector_search_tools", "rag_router")

# После Агента-писателя граф завершает свою работу
workflow.add_edge("writer", END)

app_graph = workflow.compile()