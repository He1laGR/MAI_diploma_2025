from typing import Dict, List, Optional
from evaluation.llm_feedback import LLMFeedbackGenerator

# Инициализация генератора фидбека
llm_feedback_generator = None

def initialize_llm_feedback(model_path: str):
    """Initialize LLM feedback generator
    
    Args:
        model_path: Path to the Qwen model file
    """
    global llm_feedback_generator
    llm_feedback_generator = LLMFeedbackGenerator(model_path)

def generate_category_specific_feedback(category: str, question_type: str, scores: Dict[str, float]) -> List[str]:
    """Генерация специфичного фидбека для категории и типа вопроса"""
    feedback = []
    
    if category.lower() == 'python':
        if question_type.lower() == 'theory':
            if scores['bert_f1'] < 0.7:
                feedback.append("В ответе не хватает технических деталей Python. Постарайтесь упомянуть конкретные механизмы языка.")
            if scores['tfidf_coverage'] < 0.6:
                feedback.append("Используйте больше специальных терминов Python (например, GIL, декораторы, генераторы).")
                
        elif question_type.lower() == 'code':
            if scores['tfidf_similarity'] < 0.6:
                feedback.append("Опишите сложность алгоритма и возможные оптимизации.")
                
    elif category.lower() == 'data science':
        if question_type.lower() == 'theory':
            if scores['bert_f1'] < 0.7:
                feedback.append("Раскройте математические основы и принципы работы алгоритмов.")
            if scores['tfidf_coverage'] < 0.6:
                feedback.append("Используйте больше терминов из области машинного обучения и статистики.")
                
        elif question_type.lower() == 'code':
            if scores['tfidf_similarity'] < 0.6:
                feedback.append("Упомяните метрики качества и способы их оптимизации.")
    
    return feedback

def generate_feedback(
    answer: str,
    reference_answer: str,
    category: str,
    question_type: str,
    scores: Dict[str, float],
    key_terms: Optional[List[str]] = None
) -> str:
    """
    Генерация комплексного фидбека
    
    Args:
        answer: Ответ пользователя
        reference_answer: Эталонный ответ
        category: Категория вопроса (python/data science)
        question_type: Тип вопроса (theory/code)
        scores: Словарь с оценками
        key_terms: Список ключевых терминов
    """
    if llm_feedback_generator is None:
        raise RuntimeError("LLM feedback generator not initialized. Call initialize_llm_feedback first.")
    
    # Генерация фидбека с помощью LLM
    feedback = llm_feedback_generator.generate_feedback(
        answer=answer,
        reference_answer=reference_answer,
        category=category,
        question_type=question_type,
        scores=scores
    )
    
    # Добавление ключевых терминов, если они предоставлены
    if key_terms:
        missing_terms = [term for term in key_terms if term.lower() not in answer.lower()]
        if missing_terms:
            feedback += f"\n\nРекомендуется использовать следующие термины: {', '.join(missing_terms)}"
    
    return feedback

if __name__ == "__main__":
    # Пример использования
    test_answer = "Я использовал профилирование для оптимизации алгоритма и улучшил производительность на 50%."
    
    test_reference = "Применил профилирование, нашел узкие места и переписал алгоритм, достигнув улучшения производительности на 50%."
    
    test_scores = {
        'bert_precision': 0.85,
        'bert_recall': 0.82,
        'bert_f1': 0.83,
        'tfidf_similarity': 0.75,
        'tfidf_coverage': 0.8,
        'final_score': 0.85
    }
    
    test_key_terms = ["оптимизация", "производительность", "профилирование", "алгоритм"]
    
    # Инициализация генератора фидбека
    initialize_llm_feedback("path/to/qwen/model.gguf")
    
    feedback = generate_feedback(
        test_answer,
        test_reference,
        "python",
        "code",
        test_scores,
        test_key_terms
    )
    
    print(feedback) 