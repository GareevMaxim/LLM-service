import os
import sys
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from src.application.tools.vector_search import mock_embedding_function
from src.settings.app_settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def init_db():
    url = settings.QDRANT_URL
    if "qdrant" in url and not os.path.exists("/.dockerenv"):
        url = "http://localhost:6333"

    logger.info(f"Подключение к Qdrant по адресу: {url}...")
    client = QdrantClient(url=url)
    
    if client.collection_exists(settings.COLLECTION_NAME):
        logger.warning(f"Удаляем старую коллекцию {settings.COLLECTION_NAME}...")
        client.delete_collection(collection_name=settings.COLLECTION_NAME)
        
        logger.info(f"Создаем чистуюколлекцию {settings.COLLECTION_NAME} (E5-large, 1024, COSINE)...")
    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    
    data_dir = "data_source"
    if not os.path.exists(data_dir):
        logger.error(f"Ошибка: Папка '{data_dir}' не найдена!")
        return

    points = []
    point_idx = 0
    logger.info("Начало векторизации файлов через модель intfloat/multilingual-e5-small...")
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            vector = mock_embedding_function(content)
            points.append(
                PointStruct(id=point_idx, vector=vector, payload={"filename": filename, "text": content})
            )
            logger.info(f"Успешно векторизован TXT файл: {filename}")
            point_idx += 1
            
        elif filename.endswith(".pdf"):
            logger.info(f"Обнаружен PDF документ: {filename}. Начинаем парсинг...")
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                for page in pages:
                    page_text = page.page_content
                    if not page_text.strip():
                        continue
                        
                    vector = mock_embedding_function(page_text)
                    page_num = page.metadata.get("page", 0) + 1
                    
                    points.append(
                        PointStruct(
                            id=point_idx,
                            vector=vector,
                            payload={
                                "filename": f"{filename} (Стр. {page_num})",
                                "text": page_text
                            }
                        )
                    )
                    logger.info(f"Успешно векторизована страница {page_num} из PDF: {filename}")
                    point_idx += 1
            except Exception as e:
                logger.error(f"Ошибка при разборе PDF {filename}: {str(e)}")
            
    if points:
        logger.info(f"Загрузка {len(points)} точек в Qdrant...")
        client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
        logger.info("БАЗА ДАННЫХ УСПЕШНО ОБНОВЛЕНА")
    else:
        logger.warning("Файлы для загрузки не найдены.")

if __name__ == "__main__":
    init_db()
