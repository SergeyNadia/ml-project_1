# Базовый образ Python 3.10 slim для минимального размера
FROM python:3.10-slim

# Устанавливаем метаданные образа (опционально)
LABEL maintainer="your-email@example.com"
LABEL description="ML Model API with Flask"
LABEL version="1.0"

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем системные пакеты и устанавливаем необходимые системные зависимости
# ca-certificates - для SSL сертификатов
# && \ - объединяет команды в один слой для уменьшения размера
# apt-get clean - очищает кэш пакетов
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем ТОЛЬКО файл зависимостей сначала
# Это позволяет Docker кэшировать слой с установленными пакетами
COPY requirements.txt .

# Устанавливаем Python зависимости с оптимизацией
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
# .dockerignore должен исключать ненужные файлы (логи, кэш и т.д.)
COPY . .
COPY models/ ./models/

# Создаем необходимые директории для работы приложения
RUN mkdir -p models data logs

# Создаем непривилегированного пользователя для безопасности
# Это предотвращает запуск приложения от root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Открываем порт для API
EXPOSE 8000

# Переменные окружения для конфигурации приложения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check для мониторинга состояния контейнера
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Команда запуска приложения
# Используем waitress для production-ready сервера
CMD ["python", "api.py", "script_1.py"]