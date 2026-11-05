import os
import logging
from langchain_core.messages import AIMessage
from src.infrastructure.llm_factory import get_llm
from src.application.workflows.state import AgentState
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)

def writer_node(state: AgentState) -> dict:
    """
    Writer Agent. Очищает служебные сообщения и генерирует чистый ответ.
    """
    # 1. Загружаем текст промпта из txt-файла
    prompt_path = os.path.join(settings.PROMPTS_PATH, "writer_agent_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    # 2. Достаем документы
    docs = state.get("relevant_docs", [])
    context = "\n\n".join(docs) if docs else "Документы не найдены."
    system_prompt = system_prompt_template.format(context=context)

    # 3. Находим САМЫЙ ПЕРВЫЙ вопрос пользователя (обычно это первое сообщение в истории)
    # Это отсечет ToolMessage и лишние AIMessage с вызовами инструментов
    user_query = ""
    for msg in state["messages"]:
        if msg.type == "human":  # Или isinstance(msg, HumanMessage)
            user_query = msg.content
            break
            
    if not user_query:
        user_query = "Пожалуйста, ответь на основе контекста."

    # Инициализируем LLM
    llm = get_llm()

    # 4. Формируем чистый контекст для модели: Промпт с инструкцией + чистый вопрос
    clean_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Вызываем модель
    response = llm.invoke(clean_messages)

    logger.info(f"[Writer Agent] Сформирован финальный ответ.")

    return {
        "messages": [response]
    }
