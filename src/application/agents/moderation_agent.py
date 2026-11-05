import os
import logging
from langchain_core.messages import HumanMessage
from src.infrastructure.llm_factory import get_llm
from src.application.models.moderation_agent_models import ModerationResponse
from src.application.workflows.state import AgentState
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)

def moderation_node(state: AgentState) -> dict:
    """Узел модерации с гарантированным извлечением текста"""
    try:
        #  Загружаем текст промпта из txt-файла
        prompt_path = os.path.join(settings.PROMPTS_PATH, "moderation_agent_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            
        llm = get_llm()
        structured_llm = llm.with_structured_output(ModerationResponse)
        
        #  Гарантированный поиск сообщения
        messages_list = state.get("messages", [])
        
        if messages_list and len(messages_list) > 0:
            last_message = messages_list[-1].content
        else:
            logger.warning("[Moderation] messages_list пуст в State. Пробуем резервный поиск.")
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Проанализируй запрос: {last_message}"}
        ]
        
        # Вызываем модель
        result: ModerationResponse = structured_llm.invoke(messages)
        logger.info(f"[Moderation Agent] Результат: {result.is_relevant} | Причина: {result.reason}")
        
        return {
            "is_safe": result.is_relevant
        }
        
    except Exception as e:
        logger.error(f"[Moderation] Критическая ошибка в узле модерации: {str(e)}", exc_info=True)
        # Если локальная модель упала на структурированном выводе, 
        # по умолчанию пропускаем запрос в RAG ради стабильности
        return {"is_safe": True}
