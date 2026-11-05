import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.domain.entities.chat_schemas import ChatRequest, ChatResponse
from src.application.workflows.workflow import app_graph

# Настраиваем конфигурацию логирования для всего приложения
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Home Multi-Agent RAG Service",
    description="Микросервис техподдержки умного дома на базе FastAPI, LangGraph и Qdrant",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Простая проверка работоспособности сервиса"""
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Эндпоинт для общения с мультиагентной RAG-системой (Синхронный).
    Принимает вопрос, прогоняет через LangGraph и вытаскивает финальный ответ Райтера.
    """
    logger.info(f"[API] Получен новый запрос от пользователя: '{request.message}'")
    
    try:
        # Инициализируем стартовое состояние графа
        initial_state = {
            "messages": [{"role": "user", "content": request.message}],
            "relevant_docs": [],
            "is_safe": True
        }

        # Запускаем выполнение графа
        final_state = app_graph.invoke(initial_state)

        final_message = ""
        if "messages" in final_state:
            for msg in reversed(final_state["messages"]):
                if msg.content and isinstance(msg.content, str) and msg.content.strip():
                    # Проверяем, что это не технический AIMessage, содержащий только вызов инструментов
                    if not getattr(msg, "tool_calls", None):
                        final_message = msg.content.strip()
                        break

        # На случай, если цикл не смог отфильтровать историю
        if not final_message and "messages" in final_state and final_state["messages"]:
            final_message = final_state["messages"][-1].content

        # Извлекаем сохраненные документы
        docs = final_state.get("relevant_docs", [])
        
        logger.info(f"[API] Обработка графа успешно завершена. Длина извлеченного ответа: {len(final_message)} симв.")

        return ChatResponse(
            answer=final_message,
            relevant_docs=docs
        )
    except Exception as e:
        logger.error(f"[API] Ошибка при обработке запроса графом: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {str(e)}")


@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Эндпоинт для потокового общения с RAG-системой (Стриминг).
    Постепенно отдает токены финального ответа Райтера прямо на лету(в сваггере неработает :D)
    """
    logger.info(f"[API Стриминг] Получен новый запрос от пользователя: '{request.message}'")
    
    async def event_generator():
        initial_state = {
            "messages": [{"role": "user", "content": request.message}],
            "relevant_docs": [],
            "is_safe": True
        }
        
        # Переменная для сохранения найденных документов из графа
        relevant_docs = []

        try:
            # Асинхронно слушаем события графа LangGraph
            async for event in app_graph.astream_events(initial_state, version="v2"):
                kind = event.get("event")
                
                # Ловим момент завершения работы узла поиска, чтобы забрать документы
                if kind == "on_chain_end" and event.get("name") == "custom_tools_wrapper":
                    output = event.get("data", {}).get("output", {})
                    if "relevant_docs" in output:
                        relevant_docs = output["relevant_docs"]

                # Ловим потоковые текстовые токены от финальной LLM
                elif kind == "on_chat_model_stream":
                    metadata = event.get("metadata", {})
                    # Фильтруем токены, оставляя только те, что сгенерированы узлом writer
                    if metadata.get("langgraph_node") == "writer":
                        chunk = event["data"].get("chunk")
                        if chunk and chunk.content:
                            yield chunk.content

            # В самом конце потока отдаем структуру с документами relevant_docs
            if relevant_docs:
                meta_payload = {
                    "relevant_docs": relevant_docs
                }
                yield f"\n\n---METADATA---\n{json.dumps(meta_payload, ensure_ascii=False)}"

        except Exception as e:
            logger.error(f"[API Стриминг] Критическая ошибка в потоке: {str(e)}", exc_info=True)
            yield f"\n[Ошибка потока сервера: {str(e)}]"

    # Возвращаем стрим с заголовком text/event-stream, который Swagger умеет читать «на лету»
    return StreamingResponse(event_generator(), media_type="text/event-stream")
