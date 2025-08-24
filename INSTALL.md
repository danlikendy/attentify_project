# 🚀 Инструкция по установке и запуску Attentify

## 📋 Требования

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM (для работы с трансформером)
- Git

## 🛠 Установка

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd attentify_project
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 🚀 Запуск приложения

### Способ 1: Прямой запуск Python

```bash
python run.py
```

Приложение будет доступно по адресу: http://localhost:8000

### Способ 2: Запуск через uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Способ 3: Запуск через Docker

```bash
# Сборка и запуск
docker-compose up --build

# Или только основное приложение
docker build -t attentify .
docker run -p 8000:8000 attentify
```

## 🌐 Доступные endpoints

### Веб-интерфейс
- **Главная страница**: http://localhost:8000/
- **Интерактивный интерфейс**: http://localhost:8000/static/index.html
- **API документация**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API endpoints
- `GET /health` - Проверка статуса системы
- `POST /api/translate` - Перевод текста
- `POST /api/summarize` - Резюмирование текста
- `POST /api/attention` - Получение весов внимания
- `POST /api/process` - Обработка текста
- `GET /api/model-info` - Информация о модели

## 🔧 Настройка

### Переменные окружения

Создайте файл `.env` в корневой директории:

```env
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://user:password@localhost:5432/attentify
MODEL_PATH=./checkpoints/best_model.pt
```

### Конфигурация модели

Параметры модели можно изменить в файле `app/main.py`:

```python
MODEL_CONFIG = {
    'd_model': 512,      # Размерность модели
    'n_layers': 6,       # Количество слоев
    'n_heads': 8,        # Количество голов внимания
    'd_ff': 2048,        # Размерность FFN
    'dropout': 0.1       # Dropout
}
```

## 📚 Использование

### 1. Открытие веб-интерфейса

1. Перейдите по адресу http://localhost:8000
2. Нажмите "Попробовать API" для доступа к интерактивному интерфейсу

### 2. Обработка текста

1. Выберите тип задачи (перевод, резюмирование, упрощение)
2. Введите исходный текст
3. Настройте параметры (максимальная длина результата)
4. Нажмите "Обработать текст"

### 3. Визуализация внимания

После обработки текста автоматически отобразится:
- Результат обработки
- Heatmap внимания для каждого слоя модели
- Интерактивные графики

### 4. Настройка параметров модели

В правой панели можно изменить:
- Количество слоев
- Количество голов внимания
- Размерность модели
- Размерность FFN

## 🧪 Тестирование

### Запуск тестов

```bash
# Установка тестовых зависимостей
pip install pytest pytest-asyncio httpx

# Запуск тестов
pytest tests/
```

### Тестирование API

```bash
# Проверка статуса
curl http://localhost:8000/health

# Перевод текста
curl -X POST "http://localhost:8000/api/translate" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Hello world&source_lang=en&target_lang=ru&max_length=50"
```

## 🐛 Устранение неполадок

### Проблема: "Model not loaded"

**Решение**: Проверьте, что модель инициализирована корректно. В логах должно быть сообщение "Model initialization completed successfully!"

### Проблема: CUDA out of memory

**Решение**: Уменьшите размер модели в `MODEL_CONFIG` или используйте CPU:
```python
device = 'cpu'  # Вместо 'cuda'
```

### Проблема: Port already in use

**Решение**: Измените порт в переменной окружения или остановите процесс, использующий порт 8000.

### Проблема: Import errors

**Решение**: Убедитесь, что виртуальное окружение активировано и все зависимости установлены:
```bash
pip install -r requirements.txt --force-reinstall
```

## 📁 Структура проекта

```
attentify_project/
├── app/                    # Основной код приложения
│   ├── core/              # Ядро трансформера
│   │   ├── transformer.py # Реализация архитектуры
│   │   ├── text_processor.py # Обработка текста
│   │   └── trainer.py     # Обучение модели
│   ├── static/            # Статические файлы
│   │   └── index.html     # Веб-интерфейс
│   └── main.py            # FastAPI приложение
├── checkpoints/           # Сохраненные модели
├── uploads/               # Загруженные файлы
├── logs/                  # Логи
├── requirements.txt       # Зависимости
├── run.py                 # Скрипт запуска
├── Dockerfile             # Docker конфигурация
├── docker-compose.yml     # Docker Compose
└── README.md              # Документация
```

## 🔄 Обновление

### Обновление кода

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Обновление модели

```bash
# Остановите приложение
# Замените файл модели в ./checkpoints/
# Перезапустите приложение
```

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи приложения
2. Убедитесь, что все зависимости установлены
3. Проверьте версии Python и PyTorch
4. Создайте issue в репозитории

## 🎯 Следующие шаги

После успешного запуска:

1. **Изучите API**: Откройте http://localhost:8000/docs
2. **Попробуйте интерфейс**: http://localhost:8000/static/index.html
3. **Настройте модель**: Измените параметры под ваши задачи
4. **Обучите на своих данных**: Используйте API для загрузки собственных данных
5. **Интегрируйте в проекты**: Используйте API endpoints в своих приложениях

---

**Удачного использования Attentify! 🧠✨**
