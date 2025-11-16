```markdown
# ML Project - Credit Risk Prediction

Проект для предсказания кредитного риска с использованием машинного обучения. Включает обучение модели, API для предсказаний и контейнеризацию с Docker.

## 🚀 Быстрый старт

### 1. Клонирование и запуск
```bash
git clone <repository-url>
cd ml-project
make build
```

### 2. Проверка работы
```bash
make status
```

API будет доступно по адресу: `http://localhost:8000`

## 📁 Структура проекта

```
ml-project/
├── 📁 models/           # Сохраненные модели
├── 📁 data/            # Данные для обучения
├── 🐳 Dockerfile       # Конфигурация Docker
├── 🐳 docker-compose.yml
├── 🔧 Makefile         # Команды управления
├── 📋 requirements.txt # Зависимости Python
├── 🐍 main.py         # Основной скрипт обучения
├── 🌐 api.py          # Flask API
└── 📖 README.md
```

## 🛠️ Команды управления

```bash
make build      # Собрать и запустить
make up         # Запустить сервисы
make down       # Остановить
make logs       # Просмотр логов
make clean      # Полная очистка
make restart    # Перезапустить
make status     # Проверить статус
```

## 📡 API Endpoints

### Основные endpoints:

- `GET /` - Информация о API
- `GET /health` - Проверка здоровья
- `GET /model/info` - Информация о модели
- `POST /predict` - Предсказание

### Пример предсказания:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": 1.2, "feature3": -0.3}'
```

## 🤖 Модель машинного обучения

**Алгоритм:** Random Forest Classifier  
**Метрики:**
- Accuracy: ~89%
- Precision: ~93% 
- ROC AUC: ~95%

### Обучение модели:
```bash
python main.py
```

## 🔧 Разработка

### Локальный запуск:
```bash
pip install -r requirements.txt
python main.py    # Обучение модели
python api.py     # Запуск API
```

### Модули проекта:
- `config.py` - Настройки
- `data_preprocessor.py` - Обработка данных
- `model_pipeline.py` - Обучение модели
- `inference.py` - Предсказания
- `monitoring.py` - Мониторинг

## 🐛 Решение проблем

### Переобучение модели:
```bash
python main.py
```

### Очистка и перезапуск:
```bash
make clean
make build
```

### Проверка портов:
```bash
# Проверить занятость порта 8000
netstat -tulpn | grep :8000
```
