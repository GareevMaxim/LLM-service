"""Repository for Chat entity."""

from sqlalchemy.ext.asyncio import AsyncSession
from src.infrastructure.db.repositories.base_repo import BaseRepository
from src.domain.entities.db.chat import Chat


class ChatRepository(BaseRepository[Chat]):
    """Repository for managing Chat entities."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Chat)
        self._unique_fields = ['created_at']

