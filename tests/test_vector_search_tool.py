import pytest
from unittest.mock import patch, MagicMock
from qdrant_client.models import ScoredPoint
from src.application.tools.vector_search import search_smart_home_docs


@pytest.fixture
def mock_e5_qdrant_response():
    """Создает мок-объект ответа Qdrant для модели E5."""
    fake_point = ScoredPoint(
        id=1,
        version=1,
        score=0.88,
        payload={
            "text": "Техническая документация: Пылесос промышленный Bosch C12-441Д...",
            "filename": "device_specs_errors.pdf (Стр. 1)"
        },
        vector=None
    )
    
    response = MagicMock()
    response.points = [fake_point]
    return response


@patch("src.application.tools.vector_search.mock_embedding_function")
@patch("src.application.tools.vector_search.QdrantClient")
def test_search_smart_home_docs_e5_success(mock_qdrant_class, mock_embedding_fn, mock_e5_qdrant_response):
    """Проверяет успешный семантический поиск документов через модель E5."""
    
    # Имитирует возврат плоского вектора размерностью 1024
    mock_embedding_fn.return_value = [0.1] * 1024
    
    # Настраивает заглушку для QdrantClient
    mock_client_instance = MagicMock()
    mock_client_instance.query_points.return_value = mock_e5_qdrant_response
    mock_qdrant_class.return_value = mock_client_instance

    # Вызываем функцию через .func, чтобы обойти декоратор LangChain @tool
    result = search_smart_home_docs.func(query="Сломался промышленный пылесос")

    # Проверяем результат
    assert "device_specs_errors.pdf" in result
    assert "Bosch" in result
    
    # Проверяем вызовы
    mock_embedding_fn.assert_called_once_with("Сломался промышленный пылесос")
    mock_client_instance.query_points.assert_called_once()


@patch("src.application.tools.vector_search.mock_embedding_function")
@patch("src.application.tools.vector_search.QdrantClient")
def test_search_smart_home_docs_e5_not_found(mock_qdrant_class, mock_embedding_fn):
    """Проверяет поведение поиска, если в базе ничего не нашлось."""
    
    mock_embedding_fn.return_value = [0.1] * 1024
    mock_client_instance = MagicMock()
    
    empty_response = MagicMock()
    empty_response.points = []
    mock_client_instance.query_points.return_value = empty_response
    mock_qdrant_class.return_value = mock_client_instance

    result = search_smart_home_docs.func(query="несуществующий девайс")

    assert result == "В базе данных не найдено подходящих инструкций."


@patch("src.application.tools.vector_search.QdrantClient")
def test_search_smart_home_docs_e5_exception(mock_qdrant_class):
    """Проверяет перехват ошибок блоком except, если Qdrant недоступен."""
    
    mock_qdrant_class.side_effect = Exception("Connection refused")

    result = search_smart_home_docs.func(query="ошибка сети")

    assert "Ошибка векторного поиска" in result
