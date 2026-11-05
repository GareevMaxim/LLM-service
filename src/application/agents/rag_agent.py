import os
import logging
from src.infrastructure.llm_factory import get_llm
from src.application.tools.vector_search import search_smart_home_docs
from src.application.workflows.state import AgentState
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)

def rag_router_node(state: AgentState) -> dict:
    """
    RAG-агент (Маршрутизатор).
    
    """
    # Загружаем текст промпта из txt-файла
    prompt_path = os.path.join(settings.PROMPTS_PATH, "rag_agent_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm = get_llm()

    # Привязываем инструмент векторного поиска к нашей модели
    
    llm_with_tools = llm.bind_tools([search_smart_home_docs])

    # Формируем историю сообщений, добавляя системный промпт в начало
    messages = [{"role": "system", "content": system_prompt}] + list(state["messages"])

    # Вызываем модель
    response = llm_with_tools.invoke(messages)

    logger.info(f"[RAG Agent] Модель приняла решение. Вызовы инструментов: {getattr(response, 'tool_calls', [])}")

    # Возвращаем новое сообщение в историю сообщений графа
    return {
        "messages": [response]
    }