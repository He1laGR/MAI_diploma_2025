import sys
import os
import random
from datetime import datetime
import json
from functools import wraps
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../retrieval')))
from hybrid_search import hybrid_search

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

def load_history():
    """Загрузка истории"""
    history_file = 'retrieval_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_to_history(question, category, difficulty, question_type):
    """Сохранение в историю"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'category': category,
        'difficulty': difficulty,
        'question_type': question_type
    })
    with open('retrieval_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@timing_decorator
def get_random_question(category: str, difficulty: str, question_type: str) -> dict:
    """Получение случайного вопроса из базы"""
    category = category.strip().lower()
    difficulty = difficulty.strip().lower()
    question_type = question_type.strip().lower()
    
    difficulty_map = {
        'easy': 'easy',
        'medium': 'medium',
        'high': 'high',
        'hard': 'high'
    }
    difficulty = difficulty_map.get(difficulty, difficulty)

    type_map = {
        'theory': 'theory',
        'case': 'case',
        'code': 'code'
    }
    question_type = type_map.get(question_type, question_type)
    
    print(f"\nПоиск вопросов с параметрами:")
    print(f"Категория: {category}")
    print(f"Сложность: {difficulty}")
    print(f"Тип: {question_type}")
    
    # Получаем случайные вопросы
    questions = hybrid_search(
        top_k=10,   
        category=category,
        question_type=question_type,
        difficulty=difficulty
    )
    
    if not questions:
        print(f"Не найдено вопросов для категории '{category}', сложности '{difficulty}' и типа '{question_type}'")
        return None
        
    # Выводим информацию о найденных вопросах
    print("\nДоступные вопросы:")
    print("-" * 50)
    print(f"Всего найдено: {len(questions)}")
    print("\nПримеры вопросов:")
    for i, q in enumerate(questions[:5], 1):  # Показываем только первые 5 для примера
        print(f"{i}. {q['payload']['question']}")
        print(f"   Категория: {q['payload']['category']}")
        print(f"   Сложность: {q['payload']['difficulty']}")
        print(f"   Тип: {q['payload']['question_type']}")
        print()
    
    selected_question = random.choice(questions)
    question_data = selected_question['payload']
    
    save_to_history(question_data['question'], category, difficulty, question_type)
    
    return question_data

if __name__ == "__main__":
    print("Получение случайного вопроса из базы")
    print("-" * 50)
    
    cat = input("Категория: ")
    diff = input("Сложность (Easy/Medium/High): ")
    qtype = input("Тип вопроса (theory/case/code): ")
    
    print("\nПоиск вопроса...")
    result = get_random_question(cat, diff, qtype)
    print("\nРезультат:")
    print("-" * 50)
    print(result) 