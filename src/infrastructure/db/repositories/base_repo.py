"""Base repository with common database operations."""

from typing import List, Optional, TypeVar, Generic, Type, Union, Set, Dict, Tuple, Any
from sqlmodel import SQLModel, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, tuple_, text
from sqlalchemy.exc import IntegrityError
from src.infrastructure.db.duplicate_strategies import DuplicateStrategy
import numpy as np


T = TypeVar('T', bound=SQLModel)


from src.infrastructure.db.postgres_utils import UnitOfWork


class BaseRepository(Generic[T]):
    """Base repository with common database operations."""
    
    def __init__(
        self, 
        session: AsyncSession, 
        model_class: Type[T]
    ):
        """
        Initialize BaseRepository.
        
        Args:
            session: SQLModel session
            model_class: SQLModel class
            unique_fields: List of unique field names (optional, for bulk operations)
            duplicate_strategy: Strategy for handling duplicates (optional, for bulk operations)
        """
        self.session = session
        self.model_class = model_class


    # Use UnitOfWork as the transactional context


    async def get_all(self, query=None) -> List[T]:
        """
        Retrieves all records from the database.
        
        Args:
            query: A SQLModel query object that can be further modified or executed.
            
        Returns:
            All records from the database.
        """
        if query is None:
            query = select(self.model_class)
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def filter_by(self, filters: List) -> List[T]:
        """
        Filter records by multiple conditions.
        
        Args:
            filters: List of filter conditions. Can be:
                - Direct SQLModel conditions (e.g., Model.field == value)
                - Tuples of (field, value) for automatic None handling
                - Mixed list of both types
                
        Returns:
            Filtered records.
        """
        # If no filters provided, return all records for safety
        if not filters:
            return await self.get_all()

        processed_filters = []
        
        for filter_condition in filters:
            if isinstance(filter_condition, tuple) and len(filter_condition) == 2:
                # Handle (field, value) tuples with automatic None handling
                field, value = filter_condition
                processed_filters.append(self._create_field_filter(field, value))
            else:
                # Handle direct SQLModel conditions
                processed_filters.append(filter_condition)
        
        # If nothing to filter after processing, return all
        if not processed_filters:
            return self.get_all()

        query = select(self.model_class).where(and_(*processed_filters))
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def filter_in(self, field, values: List) -> List[T]:
        """
        Filter records where a field is in a list of values.
        
        Args:
            field: The field to filter on.
            values: List of values to match against.
            
        Returns:
            Filtered records.
        """
        query = select(self.model_class).where(field.in_(values))
        result = await self.session.execute(query)
        return result.scalars().all()
    
    def _create_field_filter(self, field, value) -> Any:
        """
        Create a filter condition that properly handles None values.
        
        Args:
            field: The SQLModel field to filter on.
            value: The value to compare against (can be None).
            
        Returns:
            A SQLModel filter condition.
        """
        if value is None:
            return field.is_(None)
        else:
            return field == value

    
    async def get_by_id(self, record_id: Union[int, str]) -> Optional[T]:
        """Get a single record by its ID."""
        return await self.session.get(self.model_class, record_id)
    
    async def get_by_ids(self, record_ids: Union[List[int], List[str], Set[int], Set[str]]) -> List[T]:
        """Get multiple records by their IDs."""
        if not record_ids:
            return []
        query = select(self.model_class).where(self.model_class.id.in_(record_ids))
        result = await self.session.execute(query)
        return result.scalars().all()
    

    async def count_all(self) -> int:
        query = select(func.count(self.model_class.id))
        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_where_not_null(self, field_name: str, limit: Optional[int] = None) -> List[T]:
        """
        Return records where a given field is not NULL.
        
        Args:
            field_name: Name of the field to check for non-null values
            limit: Optional maximum number of rows to return
        """
        field = getattr(self.model_class, field_name)
        query = select(self.model_class).where(field.is_not(None))
        if limit is not None:
            query = query.limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_where_not_null(self, field_name: str) -> int:
        """
        Count records where a given field is not NULL.
        
        Args:
            field_name: Name of the field to check for non-null values
        """
        field = getattr(self.model_class, field_name)
        query = select(func.count(self.model_class.id)).where(field.is_not(None))
        result = await self.session.execute(query)
        return result.scalar() or 0

    def _resolve_unique_fields_and_strategy(
        self,
        unique_fields: Optional[List[str]],
        duplicate_strategy: Optional[DuplicateStrategy]
    ) -> Tuple[List[str], DuplicateStrategy]:
        key_fields: List[str] = unique_fields or (getattr(self, '_unique_fields', None) or [])
        if not key_fields:
            raise ValueError("unique_fields must be provided")
        strategy: DuplicateStrategy = duplicate_strategy or getattr(self, '_duplicate_strategy', None) or DuplicateStrategy.UPDATE
        return key_fields, strategy

    def _build_key_from_obj(self, obj: T, key_fields: List[str]) -> Tuple[Any, ...]:
        return tuple(getattr(obj, field_name) for field_name in key_fields)

    async def _fetch_existing_by_keys(self, key_fields: List[str], keys: List[Tuple[Any, ...]]) -> Dict[Tuple[Any, ...], T]:
        if not keys:
            return {}
        model_key_fields = [getattr(self.model_class, name) for name in key_fields]
        non_null_keys: List[Tuple[Any, ...]] = []
        null_keys_filters = []
        for key_tuple in keys:
            if any(value is None for value in key_tuple):
                and_parts = []
                for field, value in zip(model_key_fields, key_tuple):
                    and_parts.append(self._create_field_filter(field, value))
                null_keys_filters.append(and_(*and_parts))
            else:
                non_null_keys.append(key_tuple)
        conditions = []
        if non_null_keys:
            conditions.append(tuple_(*model_key_fields).in_(non_null_keys))
        if null_keys_filters:
            conditions.append(or_(*null_keys_filters))
        existing_by_key: Dict[Tuple[Any, ...], T] = {}
        if conditions:
            where_clause = conditions[0] if len(conditions) == 1 else or_(*conditions)
            query = select(self.model_class).where(where_clause)
            result = await self.session.execute(query)
            for existing in result.scalars().all():
                existing_by_key[self._build_key_from_obj(existing, key_fields)] = existing
        return existing_by_key

    async def delete(
        self,
        records_or_ids: Union[T, List[T], int, List[int], Set[int], str, List[str]]
    ) -> int:
        """
        Delete records by IDs or record instances.
        
        Args:
            records_or_ids: Can be:
                - Single record instance
                - List of record instances
                - Single ID (int or str)
                - List of IDs (int or str)
                - Set of IDs (int or str)
                
        Returns:
            Number of records deleted.
        """
        if records_or_ids is None:
            return 0
            
        # Normalize input to list of IDs
        ids_to_delete: List[int] = []
        
        if isinstance(records_or_ids, (int, str)):
            # Single ID
            ids_to_delete = [int(records_or_ids)]
        elif isinstance(records_or_ids, (list, set)):
            # List or set of IDs or records
            for item in records_or_ids:
                if isinstance(item, (int, str)):
                    # It's an ID
                    ids_to_delete.append(int(item))
                elif hasattr(item, 'id'):
                    # It's a record instance
                    ids_to_delete.append(int(item.id))
                else:
                    raise ValueError(f"Invalid item type for deletion: {type(item)}")
        elif hasattr(records_or_ids, 'id'):
            # Single record instance
            ids_to_delete = [int(records_or_ids.id)]
        else:
            raise ValueError(f"Invalid type for deletion: {type(records_or_ids)}")
        
        if not ids_to_delete:
            return 0
            
        # Remove duplicates while preserving order
        unique_ids = list(dict.fromkeys(ids_to_delete))
        
        # Use bulk delete for efficiency. Transaction boundaries are managed by the caller/UoW.
        delete_query = select(self.model_class).where(self.model_class.id.in_(unique_ids))
        result = await self.session.execute(delete_query)
        records_to_delete = result.scalars().all()
        
        for record in records_to_delete:
            await self.session.delete(record)
        # Ensure changes are visible within the current transaction/session
        await self.session.flush()
        
        return len(records_to_delete)

    async def upsert(
        self,
        records: Union[T, List[T]],
        unique_fields: Optional[List[str]] = None,
        duplicate_strategy: Optional[DuplicateStrategy] = None
    ) -> Union[T, List[T]]:
        """Upsert one or many records and return persisted instance(s)."""
        if records is None:
            raise ValueError("records must not be None")
        is_single = not isinstance(records, list)
        input_list: List[T] = records if isinstance(records, list) else [records]
        if not input_list:
            return records  # empty list or single None already handled

        key_fields, strategy = self._resolve_unique_fields_and_strategy(unique_fields, duplicate_strategy)

        # Perform bulkified write in-place
        model_columns: List[str] = [c.name for c in self.model_class.__table__.columns]
        updatable_columns: List[str] = [c for c in model_columns if c != 'id' and c not in key_fields]

        incoming_by_key: Dict[Tuple, T] = {}
        for obj in input_list:
            key = self._build_key_from_obj(obj, key_fields)
            incoming_by_key[key] = obj
        existing_by_key: Dict[Tuple, T] = await self._fetch_existing_by_keys(key_fields, list(incoming_by_key.keys()))

        to_insert: List[T] = []
        # Perform changes within the caller-managed transaction (UnitOfWork outside)
        for key, incoming in incoming_by_key.items():
            if key in existing_by_key:
                if getattr(strategy, "name", str(strategy)) == "ERROR":
                    raise IntegrityError("Duplicate detected during upsert", params=str(key), orig=None)
                if getattr(strategy, "name", str(strategy)) == "SKIP":
                    continue
                existing = existing_by_key[key]
                for col in updatable_columns:
                    setattr(existing, col, getattr(incoming, col))
            else:
                # Build payload for insert, excluding id from updates but INCLUDING it for inserts when provided
                payload = {col: getattr(incoming, col) for col in model_columns if col != 'id'}
                if 'id' in model_columns:
                    incoming_id = getattr(incoming, 'id', None)
                    if incoming_id is not None:
                        payload['id'] = incoming_id
                to_insert.append(self.model_class(**payload))
        if to_insert:
            self.session.add_all(to_insert)
            await self.session.flush()

        # Fetch persisted instances in bulk by keys and map back to input order
        input_keys = [self._build_key_from_obj(obj, key_fields) for obj in input_list]
        existing_map = await self._fetch_existing_by_keys(key_fields, input_keys)
        persisted_list: List[T] = [existing_map.get(k) for k in input_keys]

        if is_single:
            return persisted_list[0]
        return persisted_list

    