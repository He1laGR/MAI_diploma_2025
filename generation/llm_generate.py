import sys
import os
import json
from datetime import datetime
import time
from functools import wraps, lru_cache
import requests
import logging

# Модуль для проверки работоспособности генерации вопросов с помощью LLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../retrieval')))
from hybrid_search import hybrid_search

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация сервиса
MODEL_SERVICE_URL = "http://localhost:5001"

# Проверка доступности сервиса
def check_service_health():
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/health")
        return response.status_code == 200
    except requests.RequestException:
        return False

if not check_service_health():
    raise RuntimeError(
        "Model service is not available. Please start the service first using:\n"
        "python generation/model_service.py"
    )

# Few-shot примеры для разных категорий
FEW_SHOT_EXAMPLES = {
    'data science': [
        {
            'example': 'Что такое переобучение в машинном обучении?',
            'category': 'data science',
            'difficulty': 'medium',
            'explanation': 'Вопрос о базовом понятии ML, средней сложности'
        },
        {
            'example': 'Как работает метод опорных векторов?',
            'category': 'data science',
            'difficulty': 'high',
            'explanation': 'Вопрос о конкретном алгоритме, высокой сложности'
        }
    ],
    'python': [
        {
            'example': 'Что такое декораторы в Python?',
            'category': 'python',
            'difficulty': 'medium',
            'explanation': 'Вопрос о важной концепции Python, средней сложности'
        }
    ]
}

# Специализированные промпты для разных типов вопросов
'''THEORY_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    "Создай ОДИН теоретический вопрос для it-собеседования.\n"
    "\n"
    "Требования к вопросу:\n"
    "1. Язык: русский\n"
    "2. Длина: 5-8 слов\n"
    "3. Начало: 'Что', 'Как', 'Объясните', 'Опишите'\n"
    "4. Окончание: знак вопроса\n"
    "5. Стиль: простой и понятный\n"
    "\n"
    "Примеры хороших вопросов:\n"
    "- 'Что такое списки в Python?'\n"
    "- 'Как работает сборщик мусора?'\n"
    "- 'Объясните принцип работы декораторов?'\n"
    "- 'Что такое корутины в Python?'\n"
    "- 'Как Python ищет модули при импорте?'\n"
    "Создай теоретический вопрос для категории '{category}' со сложностью '{difficulty}' на основе примера: '{question}'\n"
)'''

THEORY_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    "Представь, что ты проводишь собеседование для it-специалиста. Создай ОДИН теоретический вопрос.\n"
    "Примеры хороших вопросов:\n"
    "- 'Что такое списки в Python?'\n"
    "- 'Как работает сборщик мусора?'\n"
    "- 'Объясните принцип работы декораторов?'\n"
    "- 'Что такое корутины в Python?'\n"
    "- 'Как Python ищет модули при импорте?'\n"
    "Создай теоретический вопрос для собеседования по '{category}' со сложностью вопроса '{difficulty}' на основе примера: '{question}'\n"
)

CODE_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    "Твоя задача — ВЕРНУТЬ ИСХОДНУЮ ЗАДАЧУ БЕЗ ИЗМЕНЕНИЙ.\n"
    "\n"
    "Правила:\n"
    "1. ВЕРНИ ЗАДАЧУ ТОЧНО КАК ЕСТЬ\n"
    "2. НЕ МЕНЯЙ ФОРМУЛИРОВКУ\n"
    "3. НЕ ДОБАВЛЯЙ ПРИМЕРЫ\n"
    "4. НЕ ДОБАВЛЯЙ ПОЯСНЕНИЯ\n"
    "Верни эту задачу для категории '{category}' со сложностью '{difficulty}' без изменений: '{question}'\n"
)

'''CASE_PROMPT = (
    "<|im_start|>system\n"
    "Ты — ассистент, который помогает готовиться к IT-собеседованиям. ВСЕГДА отвечай ТОЛЬКО на РУССКОМ языке.\n"
    "Создай ОДИН кейс-вопрос для собеседования на основе примера.\n"
    "Вопрос должен быть:\n"
    "- На русском языке\n"
    "- Четким и понятным\n"
    "\n"
    "Примеры хороших кейс-вопросов:\n"
    "- 'Как бы вы оптимизировали производительность веб-приложения, которое медленно загружается?'\n"
    "- 'Предложите решение для масштабирования базы данных при росте нагрузки'\n"
    "- 'Как бы вы организовали систему кэширования для часто запрашиваемых данных?'\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Создай кейс-вопрос для категории '{category}' со сложностью '{difficulty}' на основе примера: '{question}'\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)'''

CASE_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    "Создай ОДИН кейс-вопрос для собеседования на основе примера.\n"
    "Вопрос должен быть:\n"
    "- Четким и понятным\n"
    "\n"
    "Создай кейс-вопрос для категории '{category}' со сложностью '{difficulty}' на основе примера: '{question}'\n"
)

# Словарь с промптами для разных типов вопросов
QUESTION_PROMPTS = {
    'theory': THEORY_PROMPT,
    'code': CODE_PROMPT,
    'case': CASE_PROMPT
}

def load_history():
    """Загрузка истории генерации"""
    history_file = 'generation_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_to_history(original_question, paraphrased_question, category, difficulty, question_type):
    """Сохранение в историю"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().isoformat(),
        'original_question': original_question,
        'paraphrased_question': paraphrased_question,
        'category': category,
        'difficulty': difficulty,
        'question_type': question_type
    })
    with open('generation_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def is_russian_text(text):
    """Проверяет, содержит ли текст русские буквы"""
    russian_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    return any(char in russian_chars for char in text)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nВремя выполнения: {execution_time:.2f} секунд")
        return result
    return wrapper

def post_process_question(question: str) -> str:
    """Минимальная пост-обработка вопроса"""
    question = ' '.join(question.split())
    
    question = question.strip('«»""')

    if not question.endswith('?'):
        question = question.rstrip('.') + '?'
    
    return question

@timing_decorator
@lru_cache(maxsize=1000)
def generate_question(category: str, difficulty: str, question_type: str) -> str:
    """Основная функция генерации вопроса"""
    # Формируем поисковый запрос с учетом всех параметров
    search_query = f"{category} {difficulty} {question_type}"
    
    # Получаем похожие вопросы
    similar_questions = hybrid_search(
        query=search_query,
        top_k=15
    )
    
    if not similar_questions:
        print("Не найдено похожих вопросов")
        return None
        
    # Фильтруем вопросы по категории, сложности и типу
    filtered_questions = []
    for q in similar_questions:
        payload = q['payload']
        # Приводим все к нижнему регистру для сравнения
        if (payload['category'].lower() == category.lower() and
            payload['difficulty'].lower() == difficulty.lower() and
            payload['question_type'].lower() == question_type.lower()):
            filtered_questions.append(q)
    
    if not filtered_questions:
        print(f"Не найдено вопросов для категории '{category}', сложности '{difficulty}' и типа '{question_type}'")
        return None
        
    # Выводим информацию о найденных вопросах
    print("\nТоп вопросы из поиска:")
    print("-" * 50)
    print(f"Всего найдено: {len(similar_questions)}")
    print(f"После фильтрации: {len(filtered_questions)}")
    print("\nОтфильтрованные вопросы:")
    for i, q in enumerate(filtered_questions, 1):
        print(f"{i}. {q['payload']['question']}")
        print(f"   Категория: {q['payload']['category']}")
        print(f"   Сложность: {q['payload']['difficulty']}")
        print(f"   Тип: {q['payload']['question_type']}")
        print(f"   Релевантность: {q['score']:.3f}")
        print()
    
    # Выбираем случайный вопрос из отфильтрованных
    import random
    selected_question = random.choice(filtered_questions)
    
    # Если это кодовая задача или кейс, возвращаем её как есть
    if question_type.lower() in ['code', 'case']:
        return selected_question['payload']['question']
    
    # Для теоретических вопросов используем генерацию
    prompt_template = QUESTION_PROMPTS.get(question_type.lower(), THEORY_PROMPT)
    
    # Формируем промпт для генерации
    prompt = prompt_template.format(
        category=category,
        difficulty=difficulty,
        question=selected_question['payload']['question']
    )

    print("\nГенерация нового вопроса...")
    
    try:
        # Отправляем запрос к сервису
        response = requests.post(
            f"{MODEL_SERVICE_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Service error: {response.text}")
            
        result = response.json()
        generated_question = result["content"]
        
        # Проверяем, что вопрос на русском языке
        if not is_russian_text(generated_question):
            logger.warning("Generated question is not in Russian")
            return None
        
        # Минимальная пост-обработка
        generated_question = post_process_question(generated_question)
        
        # Сохраняем в историю
        save_to_history(selected_question['payload']['question'], generated_question, category, difficulty, question_type)
        
        return generated_question
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return None

if __name__ == "__main__":
    print("Генерация вопросов для собеседования")
    print("-" * 50)
    
    cat = input("Категория: ")
    diff = input("Сложность (Easy/Medium/High): ")
    qtype = input("Тип вопроса (theory/case/code): ")
    
    print("\nГенерация вопроса...")
    result = generate_question(cat, diff, qtype)
    print("\nРезультат:")
    print("-" * 50)
    print(result)
