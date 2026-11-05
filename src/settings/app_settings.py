"""Application settings with .env file support."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    # Какую модель использовать: 'openai' или 'ollama'
    LLM_PROVIDER: Literal["openai", "ollama"] = "ollama"
    
    # Настройки для OpenAI (опционально)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    
    # Настройки для локальной Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3"
    
    # Настройки Векторной БД (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "smart_home_docs"
    
    # Путь к промптам (оставляем для совместимости со структурой шаблона)
    PROMPTS_PATH: str = "src/application/prompts"

    # extra="ignore" разрешает иметь другие переменные в .env без падения в ошибку
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

def get_settings() -> AppSettings:
    return AppSettings()

# Создаем объект настроек
settings = get_settings()
