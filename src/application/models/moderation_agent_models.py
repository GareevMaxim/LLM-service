from pydantic import BaseModel, Field

class ModerationResponse(BaseModel):
    """Схема структурированного ответа от агента модерации"""
    is_relevant: bool = Field(
        description="True, если запрос относится к технике и умному дому. False в противном случае."
    )
    reason: str = Field(
        description="Краткое объяснение на русском языке, почему было принято такое решение."
    )

