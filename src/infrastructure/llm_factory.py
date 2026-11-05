from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src.settings.app_settings import settings

def get_llm() -> BaseChatModel:
    """
    Фабрика для создания объекта LLM на основе конфигурации из .env.

    """
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("Выбран провайдер 'openai', но OPENAI_API_KEY не задан в .env")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0  # Ставим 0 для жесткого следования промптам и структуре
        )
        
    elif settings.LLM_PROVIDER == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL_NAME,
            temperature=0.0,
            num_ctx=4096,      # Контекстное окно (чтобы влезали все документы)
            num_predict=256,   # Максимальная длина ответа (256 токенов хватит с головой)
            top_k=10,          # Ограничиваем выбор слов для ускорения генерации на GPU
            top_p=0.9
        )
        
    raise ValueError(f"Неподдерживаемый провайдер LLM_PROVIDER: {settings.LLM_PROVIDER}")
