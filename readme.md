```markdown
# ML Project - Credit Risk Prediction

Проект машинного обучения для предсказания кредитного риска с использованием Random Forest. Включает полный pipeline: от обучения модели до deployment с Flask API и Docker.

## 📊 О проекте

Этот проект решает задачу бинарной классификации для предсказания кредитного риска на основе исторических данных транзакций и поведения кошельков в DeFi.

### 🎯 Основные метрики модели
- **Accuracy**: ~89%
- **Precision**: ~93%
- **Recall**: ~79%
- **F1-Score**: ~85%
- **ROC AUC**: ~95%

## 🏗️ Архитектура проекта

```
ml-project/
├── 📁 models/              # Сохраненные модели (в .gitignore)
├── 📁 data/                # Данные (в .gitignore)
├── 📁 logs/                # Логи приложения
├── 🐳 Dockerfile           # Конфигурация Docker образа
├── 🐳 docker-compose.yml   # Оркестрация сервисов
├── 🔧 Makefile             # Утилита для управления
├── 📋 requirements.txt     # Зависимости Python
├── 🐍 script_1.py          # Обучение и оценка модели
├── 🌐 api.py               # Flask API сервер
├── 📖 README.md            # Документация
└── ⚙️ .gitignore           # Игнорируемые файлы
```

## 🚀 Быстрый старт

### Предварительные требования
- Docker & Docker Compose
- Git

### 1. Клонирование репозитория
```bash
git clone https://github.com/username/ml-project.git
cd ml-project
```

### 2. Запуск проекта
```bash
make build
```

### 3. Проверка работы
```bash
make status
```

## 🛠️ Управление проектом

### Команды Makefile
```bash
make build      # Собрать и запустить контейнеры
make up         # Запустить сервисы
make down       # Остановить сервисы
make logs       # Просмотр логов в реальном времени
make clean      # Полная очистка (контейнеры, образы, volumes)
make restart    # Перезапустить сервисы
make status     # Проверить статус сервисов и API
```

### Ручное управление Docker
```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Просмотр логов
docker-compose logs -f ml-api
```

## 📡 API Endpoints

### 🏠 Главная страница
```http
GET /
```
Возвращает информацию о API и доступные endpoints.

### ❤️ Health Check
```http
GET /health
```
Проверяет статус API и загруженной модели.

### ℹ️ Информация о модели
```http
GET /model/info
```
Возвращает информацию о загруженной модели.

### 🔮 Предсказание
```http
POST /predict
Content-Type: application/json

{
    "feature1": 0.5,
    "feature2": 1.2,
    "feature3": -0.3
}
```

**Ответ:**
```json
{
    "result": {
        "prediction": 1,
        "probability_class_0": 0.23,
        "probability_class_1": 0.77,
        "confidence": 0.77
    },
    "status": "success"
}
```

## 🐳 Docker конфигурация

### Сервисы
- **ml-api**: Flask API на порту 8000
- **jupyter**: Jupyter Notebook на порту 8888 (опционально)

### Переменные окружения
- `PYTHONUNBUFFERED=1` - Небуферизованный вывод Python
- `PYTHONPATH=/app` - Путь к модулям Python

## 📊 Модель машинного обучения

### Алгоритм
- **Random Forest Classifier** с 100 деревьями
- Автоматическая обработка числовых признаков
- Проверка на переобучение

### Признаки
Модель использует 75+ признаков включая:
- Историю транзакций кошелька
- Временные метки активности
- Статистику по газам
- Рисковые транзакции
- Рыночные индикаторы
- История займов и депозитов

### Обучение модели
```bash
python script_1.py
```
Скрипт автоматически:
1. Загружает данные из `dataset.parquet`
2. Обрабатывает и масштабирует признаки
3. Обучает Random Forest модель
4. Проверяет на переобучение
5. Сохраняет модель в `models/`

## 🔧 Разработка

### Локальная разработка
```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели
python script_1.py

# Запуск API
python api.py
```

### Структура кода
- **DataPreprocessor**: Предобработка и масштабирование данных
- **MLPipeline**: Обучение и оценка модели
- **InferenceEngine**: Предсказания на новой модели
- **ModelMonitor**: Мониторинг дрейфа данных

## 📈 Мониторинг и логи

### Логирование
- Уровень: INFO
- Формат: структурированные логи
- Выход: консоль и файлы в `logs/`

### Health Checks
- Автоматические проверки каждые 30 секунд
- Мониторинг загруженности модели
- Проверка дрейфа данных

## 🐛 Устранение проблем

### Модель не загружается
```bash
# Переобучить модель
python script_1.py

# Проверить файлы моделей
ls -la models/
```

### Ошибки Docker
```bash
# Полная пересборка
make clean
make build
```

### Проблемы с портами
```bash
# Проверить занятые порты
netstat -tulpn | grep :8000

# Остановить conflicting процессы
sudo lsof -ti:8000 | xargs kill -9
```

## 📄 Лицензия

MIT License

## 👥 Контакты

- Автор: [Your Name]
- Email: your.email@example.com
- GitHub: [username](https://github.com/username)

---

**🚀 Happy Coding!**
```

Этот README покрывает все аспекты проекта: от быстрого старта до troubleshooting. Теперь можно смело пушить на GitHub! 🎯