from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def bert_score_eval(
    user_answer: str,
    reference_answer: str,
    lang: str = 'ru',
    return_all: bool = False
) -> Union[float, Tuple[float, float, float]]:
    """
    Оценка ответа с помощью BERTScore
    
    Args:
        user_answer: Ответ пользователя
        reference_answer: Эталонный ответ
        lang: Язык текста
        return_all: Возвращать ли все метрики (P, R, F1)
        
    Returns:
        F1-score или кортеж (Precision, Recall, F1)
    """
    # Проверка на пустые или слишком короткие ответы
    if not user_answer or len(user_answer.strip()) < 3:
        return (0.0, 0.0, 0.0) if return_all else 0.0
    
    if not reference_answer or len(reference_answer.strip()) < 3:
        raise ValueError("Reference answer is too short or empty")
    
    P, R, F1 = bert_score(
        [user_answer],
        [reference_answer],
        lang=lang,
        rescale_with_baseline=True
    )
    
    if return_all:
        return float(P[0]), float(R[0]), float(F1[0])
    return float(F1[0])

def extract_terms(text: str, vocabulary: List[str]) -> set:
    tokens = re.findall(r'\w+', text.lower())
    return set(token for token in tokens if token in vocabulary)

def tfidf_evaluation(
    user_answer: str,
    reference_answer: str,
    key_terms: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Оценка с помощью TF-IDF
    
    Args:
        user_answer: Ответ пользователя
        reference_answer: Эталонный ответ
        key_terms: Список ключевых терминов 
        
    Returns:
        Словарь с различными метриками TF-IDF
    """
    # Проверка на пустые или слишком короткие ответы
    if not user_answer or len(user_answer.strip()) < 3:
        return {'similarity': 0.0, 'term_coverage': 0.0, 'tfidf_sum': 0.0}
    
    # Если ключевые термины не указаны, извлекаем их из эталонного ответа
    if not key_terms:
        vectorizer = TfidfVectorizer(max_features=20)
        vectorizer.fit([reference_answer])
        key_terms = vectorizer.get_feature_names_out().tolist()
    
    if not key_terms:
        return {'similarity': 0.0, 'term_coverage': 0.0, 'tfidf_sum': 0.0}
    
    # Очищаем ключевые термины от пробелов и дубликатов
    key_terms = [term.strip() for term in key_terms if term.strip()]
    key_terms = list(set(key_terms))  
    
    if not key_terms:
        return {'similarity': 0.0, 'term_coverage': 0.0, 'tfidf_sum': 0.0}
    
    try:
        # Создаем векторизатор с ключевыми терминами
        vectorizer = TfidfVectorizer(vocabulary=key_terms)
        
        # Получаем TF-IDF матрицы
        tfidf_matrix = vectorizer.fit_transform([user_answer, reference_answer])
        
        # Вычисляем косинусное сходство
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Оцениваем покрытие ключевых терминов
        user_terms = extract_terms(user_answer, key_terms)
        key_terms_set = set(key_terms)
        term_coverage = len(user_terms.intersection(key_terms_set)) / len(key_terms_set) if key_terms_set else 0.0

        tfidf_sum = float(tfidf_matrix[0].sum() / len(key_terms)) if key_terms else 0.0
        
        return {
            'similarity': similarity,
            'term_coverage': term_coverage,
            'tfidf_sum': tfidf_sum
        }
    except ValueError as e:
        # В случае ошибки возвращаем нулевые метрики
        print(f"Ошибка при TF-IDF оценке: {str(e)}")
        return {'similarity': 0.0, 'term_coverage': 0.0, 'tfidf_sum': 0.0}

def evaluate_answer(
    user_answer: str,
    reference_answer: str,
    question: str
) -> Dict[str, float]:
    """
    Комплексная оценка ответа пользователя
    
    Args:
        user_answer: Ответ пользователя
        reference_answer: Эталонный ответ
        question: Вопрос пользователя
        
    Returns:
        Словарь с оценками по различным метрикам
    """
    # Проверка на пустые или слишком короткие ответы
    if not user_answer or len(user_answer.strip()) < 3:
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0,
            'final_score': 0.0
        }
    
    if not reference_answer or len(reference_answer.strip()) < 3:
        raise ValueError("Reference answer is too short or empty")
    
    # BERTScore
    bert_precision, bert_recall, bert_f1 = bert_score_eval(
        user_answer,
        reference_answer,
        return_all=True
    )
    
    # TF-IDF для оценки технических терминов
    tfidf_metrics = tfidf_evaluation(
        user_answer=user_answer,
        reference_answer=reference_answer
    )
    
    # Комбинированная оценка
    final_score = (
        0.4 * bert_f1 +                      # Семантическое сходство
        0.2 * tfidf_metrics['similarity'] +  # Сходство по терминам
        0.2 * tfidf_metrics['term_coverage'] # Покрытие ключевых терминов
    )
    
    return {
        'bert_precision': bert_precision,
        'bert_recall': bert_recall,
        'bert_f1': bert_f1,
        'tfidf_similarity': tfidf_metrics['similarity'],
        'term_coverage': tfidf_metrics['term_coverage'],
        'final_score': final_score
    } 