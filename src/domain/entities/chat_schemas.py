from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Схема входящего запроса от пользователя"""
    message: str = Field(..., description="Текстовый запрос пользователя к чат-боту")

class ChatResponse(BaseModel):
    """Схема исходящего ответа от сервиса"""
    answer: str = Field(..., description="Финальный текстовый ответ от мультиагентной системы")
    relevant_docs: list[str] = Field(default=[], description="Список документов, которые использовались при поиске")