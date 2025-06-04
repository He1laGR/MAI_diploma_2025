from evaluation.llm_feedback import LLMFeedbackGenerator
from evaluation.answer_evaluation import evaluate_answer
from generation.question_retriever import get_random_question
import logging
import sys
import os
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_feedback.log')
    ]
)

def initialize_feedback_generator():
    """Инициализация генератора фидбека с прогревом модели"""
    start_time = time.time()
    logging.info("Начало инициализации генератора фидбека...")
    
    try:
        # Инициализация генератора фидбека без указания пути
        feedback_generator = LLMFeedbackGenerator()
        
        # Прогрев модели
        logging.info("Прогрев модели...")
        warmup_prompt = "Привет, как дела?"
        _ = feedback_generator.generate_feedback(
            answer=warmup_prompt,
            reference_answer="Привет! У меня всё хорошо, спасибо!",
            key_terms=["приветствие"],
            category="Общение",
            question_type="theory",
            scores={"bert_f1": 0.8, "tfidf_coverage": 0.8, "final_score": 0.8}
        )
        
        end_time = time.time()
        logging.info(f"Инициализация генератора фидбека завершена за {end_time - start_time:.2f} секунд")
        
        return feedback_generator
    except Exception as e:
        logging.error(f"Ошибка при инициализации генератора фидбека: {e}")
        return None

def test_feedback():
    # Инициализация генератора фидбека с прогревом
    feedback_generator = initialize_feedback_generator()
    if feedback_generator is None:
        return
    
    logging.info("Получение случайного вопроса")
    # Получаем случайный вопрос из базы
    question = get_random_question(
        category="Data Science",  
        difficulty="Medium",  
        question_type="case" 
    )
    
    if not question:
        logging.error("Не удалось получить вопрос из базы данных")
        return
    
    print("\n" + "="*50)
    print("Вопрос:")
    print(question['question'])
    print("\n" + "="*50 + "\n")
    
    # Получаем ответ от пользователя
    print("Введите ваш ответ (для завершения ввода нажмите Ctrl+D):")
    user_answer = ""
    try:
        while True:
            line = input()
            user_answer += line + "\n"
    except EOFError:
        pass
    
    print("\n" + "="*50)
    print("Ваш ответ:")
    print(user_answer)
    print("\n" + "="*50 + "\n")
    
    logging.info("Начало оценки ответа")
    # Оценка ответа
    scores = evaluate_answer(user_answer, question['reference_answer'])
    
    print("Метрики оценки:")
    print(f"- Семантическое сходство: {scores['bert_f1']:.2f}")
    print(f"- Покрытие терминов: {scores['tfidf_coverage']:.2f}")
    print(f"- Итоговая оценка: {scores['final_score']:.2f}")
    print("\n" + "="*50 + "\n")
    
    logging.info("Начало генерации фидбека")
    # Генерация фидбека
    feedback = feedback_generator.generate_feedback(
        answer=user_answer,
        reference_answer=question['reference_answer'],
        key_terms=question.get('key_terms', []),  # Добавляем key_terms, если их нет, используем пустой список
        category=question['category'],
        question_type=question['question_type'],
        scores=scores
    )
    
    logging.info("Фидбек сгенерирован")
    print("Сгенерированный фидбек:")
    print(feedback)

if __name__ == "__main__":
    test_feedback() 