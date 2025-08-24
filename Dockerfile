# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p checkpoints uploads logs

# Открываем порт
EXPOSE 8000

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

# Команда запуска
CMD ["python", "run.py"]
