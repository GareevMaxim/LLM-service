FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы конфигурации проекта (включая uv.lock, если используем uv, или просто requirements)
COPY pyproject.toml uv.lock ./

# Устанавливаем пакетный менеджер uv глобально внутри контейнера для мгновенной сборки
RUN pip install --no-cache-dir uv

RUN uv sync --frozen

# Копируем весь исходный код и текстовые промпты в контейнер
COPY src/ ./src
COPY data_source/ ./data_source
COPY .env ./

EXPOSE 8000

# Запускаем наш сервер uvicorn через окружение .venv, созданное uv
CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]