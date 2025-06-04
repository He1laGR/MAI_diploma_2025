# Интеллектуальный сервис для подготовки к собеседованиям

## Обзор
Сервис генерирует вопросы для IT-собеседований. Использует комбинацию Elasticsearch и Qdrant для поиска и языковую модель Vikhr для генерации обратной связи. Поддерживает различные категории, уровни сложности и типы вопросов (теория, код, кейсы).

## Возможности
- **Генерация вопросов**: Создает вопросы для собеседований на основе категории, сложности и типа.
- **Поиск**: Использует Elasticsearch и Qdrant для эффективного поиска вопросов.
- **Эмбеддинги**: Генерирует векторные представления вопросов с помощью языковой модели.

## Требования
- Python 3.12
- Docker и Docker Compose (для запуска через Docker)
- Необходимые Python пакеты (установка через `pip install -r requirements.txt`)

## Установка

### Локальный запуск

1. **Клонировать репозиторий:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Создать и активировать виртуальное окружение:**
   ```bash
   # Создание виртуального окружения
   python -m venv venv

   # Активация в Linux/MacOS
   source venv/bin/activate

   # Активация в Windows
   .\venv\Scripts\activate
   ```

3. **Установить зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Запустить Elasticsearch и Qdrant:**
   ```bash
   docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.13.0
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

5. **Сгенерировать эмбеддинги:**
   ```bash
   python embeddings/generate_embeddings.py
   ```

6. **Загрузить данные в Elasticsearch:**
   ```bash
   cd elasticsearch
   python setup_index.py
   python load_to_elastic.py
   ```

7. **Загрузить данные в Qdrant:**
   ```bash
   cd ../qdrant
   python load_to_qdrant.py
   ```

8. **Запустить сервисы:**
   ```bash
   # Запуск менеджера моделей
   cd retrieval
   ./start_model_cache.sh

   # Запуск сервиса эмбеддингов
   ./start_embedding_service.sh

   # Запуск сервиса генерации
   cd ../generation
   ./start_model_service.sh

   # Запуск основного приложения
   cd ..
   python app.py
   ```

### Запуск через Docker

1. **Клонировать репозиторий:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Создать необходимые директории:**
   ```bash
   mkdir -p data qdrant elasticsearch model
   ```

3. **Собрать и запустить все сервисы:**
   ```bash
   docker-compose up --build
   ```

   Это запустит:
   - Основное приложение (порт 7860) - Gradio интерфейс
   - Elasticsearch (порты 9200, 9300) - поисковый движок
   - Qdrant (порты 6333, 6334) - векторная база данных
   - Model Manager Service (порт 5000) - управление моделями
   - Embedding Service (порт 5001) - генерация эмбеддингов
   - Generation Service (порт 5002) - генерация текста

4. **Доступ к приложению:**
   - Откройте http://localhost:7860 в браузере

## Архитектура сервисов

Система состоит из нескольких сервисов:

1. **Main Application** (`app.py`):
   - Gradio интерфейс для взаимодействия с пользователем
   - Управление сессиями интервью
   - Оценка ответов и генерация фидбека

2. **Model Manager Service** (`retrieval/model_manager.py`):
   - Управление моделями и их кэширование
   - API для генерации эмбеддингов
   - API для реранкинга результатов поиска

3. **Embedding Service** (`retrieval/embedding_service.py`):
   - Генерация векторных представлений для вопросов
   - Интеграция с векторной базой Qdrant
   - Управление эмбеддингами

4. **Generation Service** (`generation/generation_service.py`):
   - Генерация текста с помощью модели Qwen
   - Обработка запросов на генерацию
   - Управление контекстом генерации

## Использование
- Выберите категорию (Python, Data Science)
- Ответьте на 5 вопросов:
  - 3 теоретических вопроса (разной сложности)
  - 1 case вопрос (практический кейс)
  - 1 code вопрос (задача на написание кода)
- Получите фидбек после каждого ответа:
  - Оценка ответа
  - Сильные стороны
  - Что можно улучшить
  - Рекомендации

## Решение проблем

### Локальный запуск
- Проверьте, что все сервисы запущены и работают
- Проверьте логи на наличие ошибок
- Убедитесь, что порты не заняты другими приложениями
- Проверьте доступность Elasticsearch и Qdrant

### Docker
- Проверьте статус контейнеров: `docker-compose ps`
- Посмотрите логи: `docker-compose logs`
- Проверьте доступность портов
- Убедитесь, что volumes созданы правильно
- Проверьте сетевые настройки Docker
