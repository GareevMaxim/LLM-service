"""Repository for ChatMessage entity."""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from src.infrastructure.db.repositories.base_repo import BaseRepository
from src.domain.entities.db.chat_message import ChatMessage


class ChatMessageRepository(BaseRepository[ChatMessage]):
    """Repository for managing ChatMessage entities."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ChatMessage)
        self._unique_fields = ['chat_id', 'created_at', 'role']
    
    async def get_messages_by_chat_id(self, chat_id: int) -> List[ChatMessage]:
        """Get all messages for a specific chat ordered by creation time."""
        query = (
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
        )
        result = await self.session.execute(query)
        return result.scalars().all()

