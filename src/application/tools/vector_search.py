import logging
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from src.settings.app_settings import settings

logger = logging.getLogger(__name__)

# Инициализируем мультиязычную модель, обученную в том числе на русском языке.
embedding_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")



def mock_embedding_function(text: str) -> list[float]:
    """
    Локально генерирует плоский вектор текста с поддержкой русского языка.
    Для моделей семейства E5 перед текстом документов принято добавлять 'passage: ',
    а перед вопросом — 'query: ', это критически важно для точности.
    """
    # Если это поисковый запрос (короткий), маркируем как query
    prefix = "query: " if len(text) < 150 else "passage: "
    
    embeddings_generator = embedding_model.passage_embed([f"{prefix}{text}"])
    vector = list(next(embeddings_generator))
    return vector


@tool
def search_smart_home_docs(query: str) -> str:
    """
    Поиск инструкций и руководств пользователя по технике умного дома.
    Вызывай этот инструмент, когда пользователь спрашивает про ошибки, 
    настройку, сброс или характеристики конкретных гаджетов (пылесос, лампа, датчик).
    """
    try:
        logger.info(f"[Qdrant Tool] Запуск поиска по русскоязычной модели E5: '{query}'")
        client = QdrantClient(url=settings.QDRANT_URL)
        
        # Получаем вектор с префиксом 'query: '
        query_vector = mock_embedding_function(query)
        
        # Обычный чистый семантический поиск по одной коллекции
        search_results = client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            limit=1
        )
        
        found_texts = []
        for point in search_results.points:
            payload = point.payload
            text = payload.get("text", "")
            filename = payload.get("filename", "")
            logger.info(f"[Qdrant Match] Найдено: {filename} | Score: {point.score}")
            found_texts.append(f"--- Документ: {filename} ---\n{text}")
            
        if not found_texts:
            logger.warning("[Qdrant Tool] Поиск не дал результатов.")
            return "В базе данных не найдено подходящих инструкций."
            
        total_context = "\n\n".join(found_texts)
        return total_context
        
    except Exception as e:
        logger.error(f"[Qdrant Tool] Ошибка поиска: {str(e)}", exc_info=True)
        return f"Ошибка векторного поиска: {str(e)}"

