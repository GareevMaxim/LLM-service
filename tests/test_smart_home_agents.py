import pytest

from unittest.mock import patch, MagicMock, mock_open
from langchain_core.messages import HumanMessage, AIMessage
from src.application.agents.moderation_agent import moderation_node
from src.application.agents.rag_agent import rag_router_node
from src.application.agents.writer import writer_node
from src.application.models.moderation_agent_models import ModerationResponse

from src.application.workflows.workflow import (
    route_after_moderation,
    route_after_rag,
    custom_tools_wrapper
)


def test_route_after_moderation_unsafe():
    """Если модератор пометил запрос как небезопасный, воркфлоу должен идти к писателю."""
    # Создаем фейковый state, где запрос заблокирован модератором
    state = {"is_safe": False, "messages": []}
    
    next_node = route_after_moderation(state)
    
    assert next_node == "writer"


def test_route_after_moderation_safe():
    """Если запрос безопасен, воркфлоу должен идти к rag_router."""
    state = {"is_safe": True, "messages": []}
    
    next_node = route_after_moderation(state)
    
    assert next_node == "rag_router"


def test_route_after_rag_needs_tools():
    """Если RAG-агент вернул AIMessage с вызовом инструмента, идем в векторный поиск."""
    # Создаем сообщение ИИ, которое имитирует запрос на вызов тулзы
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "search_smart_home_docs", "args": {"query": "лампочка"}, "id": "call_1"}]
    )
    state = {"messages": [ai_message]}
    
    next_node = route_after_rag(state)
    
    assert next_node == "vector_search_tools"


def test_route_after_rag_no_tools():
    """Если RAG-агент вернул обычный текст без вызова инструментов, идем к писателю."""
    ai_message = AIMessage(content="Я могу сам ответить на этот вопрос.")
    state = {"messages": [ai_message]}
    
    next_node = route_after_rag(state)
    
    assert next_node == "writer"


@patch("src.application.workflows.workflow.search_smart_home_docs")
def test_custom_tools_wrapper(mock_search_tool):
    """Проверяет, что узел правильно достает аргументы поиска и сохраняет текст из Qdrant в state."""
    
    # Настраиваем заглушку для инструмента поиска (чтобы он вернул фейковый текст инструкции)
    mock_search_tool.invoke.return_value = "Инструкция: нажмите кнопку сброса."

    # Формируем состояние графа, будто RAG-агент только что запросил вызов инструмента
    last_ai_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_smart_home_docs", 
            "args": {"query": "как сбросить девайс"}, 
            "id": "call_xyz"
        }]
    )
    state = {
        "messages": [HumanMessage(content="Привет"), last_ai_message],
        "relevant_docs": []
    }

    # Вызываем наш узел-обертку
    output_update = custom_tools_wrapper(state)

    # Проверяем, что узел вернул правильные обновления для AgentState
    assert "messages" in output_update
    assert "relevant_docs" in output_update
    
    # Проверяем, что текст из Qdrant попал в relevant_docs
    assert output_update["relevant_docs"] == ["Инструкция: нажмите кнопку сброса."]
    
    # Проверяем, что сформировалось валидное служебное ToolMessage сообщение
    tool_msg = output_update["messages"][0]
    assert tool_msg.content == "Инструкция: нажмите кнопку сброса."
    assert tool_msg.tool_call_id == "call_xyz"
    
    # Убеждаемся, что инструмент действительно вызывался с нужным поисковым запросом
    mock_search_tool.invoke.assert_called_once_with({"query": "как сбросить девайс"})

# Изолируем чтение промптов, чтобы тесты не зависели от физических .txt файлов
FAKE_PROMPT_CONTENT = "Фейковый промпт для теста"


@patch("builtins.open", new_callable=mock_open, read_data=FAKE_PROMPT_CONTENT)
@patch("src.application.agents.moderation_agent.get_llm")
def test_moderation_node_relevant(mock_get_llm, mock_file):
    """Проверяет, что узел модерации пропускает релевантный запрос."""
    
    # Настраиваем фейковый структурированный ответ от LLM
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    
    # Симулируем успешный проход модерации
    fake_response = ModerationResponse(is_relevant=True, reason="Запрос по теме умного дома")
    mock_structured_llm.invoke.return_value = fake_response
    
    # Связываем цепочку вызовов: llm.with_structured_output() -> structured_llm
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_llm.return_value = mock_llm

    #Формируем входное состояние
    state = {"messages": [HumanMessage(content="Как починить лампу?")]}

    # Вызываем узел
    output = moderation_node(state)

    # Проверяем, что флаг безопасности выставился в True (в соответствии с result.is_relevant)
    assert output == {"is_safe": True}
    mock_structured_llm.invoke.assert_called_once()


@patch("builtins.open", new_callable=mock_open, read_data=FAKE_PROMPT_CONTENT)
@patch("src.application.agents.moderation_agent.get_llm")
def test_moderation_node_fallback_on_exception(mock_get_llm, mock_file):
    """Если Llama или синтаксис упали с ошибкой, узел должен пропустить запрос ради стабильности."""
    
    # Заставляем вызов модели выбросить исключение (например, сбой сети)
    mock_get_llm.side_effect = Exception("LLM Timeout")
    state = {"messages": [HumanMessage(content="Критический запрос")]}

    output = moderation_node(state)

    # Убеждаемся, что сработал блок `except Exception` и вернул True, сохранив стабильность приложения
    assert output == {"is_safe": True}


@patch("builtins.open", new_callable=mock_open, read_data=FAKE_PROMPT_CONTENT)
@patch("src.application.agents.rag_agent.get_llm")
def test_rag_router_node_success(mock_get_llm, mock_file):
    """Проверяет, что RAG-агент корректно привязывает инструменты и возвращает ответ LLM."""
    
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    
    # Мок ответа от Llama (будто она решила вызвать инструмент поиска)
    fake_ai_message = AIMessage(content="", tool_calls=[{"name": "search_smart_home_docs", "args": {"query": "тест"}, "id": "1"}])
    mock_llm_with_tools.invoke.return_value = fake_ai_message
    
    # Имитируем вызов llm.bind_tools(...)
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_get_llm.return_value = mock_llm

    state = {"messages": [HumanMessage(content="Где инструкция к пылесосу?")]}

    output = rag_router_node(state)

    # Проверяем, что узел вернул новое сообщение в историю сообщений графа
    assert "messages" in output
    assert output["messages"][0] == fake_ai_message
    # Проверяем, что инструменты были привязаны перед вызовом
    mock_llm.bind_tools.assert_called_once()


@patch("builtins.open", new_callable=mock_open, read_data="Промпт шаблона: {context}")
@patch("src.application.agents.writer.get_llm")
def test_writer_node_success(mock_get_llm, mock_file):
    """Проверяет сборку контекста из relevant_docs и генерацию финального ответа."""
    
    mock_llm = MagicMock()
    final_answer = AIMessage(content="Вот как это починить: ...")
    mock_llm.invoke.return_value = final_answer
    mock_get_llm.return_value = mock_llm

    # Имитируем состояние, когда в графе уже лежат найденные документы из Qdrant
    state = {
        "messages": [HumanMessage(content="Как починить?")],
        "relevant_docs": ["Документ 1: Нажмите кнопку сброса."]
    }

    output = writer_node(state)

    # Проверяем, что писатель вернул финальный сгенерированный ответ
    assert "messages" in output
    assert output["messages"][0].content == "Вот как это починить: ..."
    
    # Дополнительно проверяем, что в вызов модели ушел системный промпт (индекс 0 во втором аргументе mock_file)
    mock_llm.invoke.assert_called_once()