from typing import Dict
import bert_score
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_llm_generation(
    generated_feedback: str,
    user_answer: str,
    reference_answer: str
) -> Dict[str, float]:
    """
    Оценивает качество генерации фидбека LLM.
    
    Args:
        generated_feedback: Сгенерированный LLM фидбек
        user_answer: Ответ пользователя
        reference_answer: Эталонный ответ
        
    Returns:
        Dict с метриками оценки генерации:
        - answer_similarity: сходство ответа пользователя с эталонным
        - feedback_uniqueness: уникальность фидбека относительно ответа пользователя
        - feedback_faithfulness: насколько фидбек отражает информацию из эталонного ответа
        - rouge_faithfulness: дополнительная оценка достоверности фидбека
        - generation_quality: итоговая оценка качества генерации
    """
    if not all([generated_feedback, user_answer, reference_answer]):
        logger.warning("Пустые входные данные")
        return {
            'answer_similarity': 0.0,
            'feedback_uniqueness': 0.0,
            'feedback_faithfulness': 0.0,
            'rouge_faithfulness': 0.0,
            'generation_quality': 0.0
        }
    
    try:
        # 1. Оценка сходства ответа с эталоном (Answer Similarity)
        P, R, F1 = bert_score.score(
            [user_answer],
            [reference_answer],
            lang='ru',
            rescale_with_baseline=True
        )
        answer_similarity = float(F1[0])
        
        # 2. Оценка уникальности фидбека (Feedback Uniqueness)
        P, R, F1 = bert_score.score(
            [generated_feedback],
            [user_answer],
            lang='ru',
            rescale_with_baseline=True
        )
        feedback_user_similarity = float(F1[0])
        feedback_uniqueness = 1 - feedback_user_similarity
        
        # 3. Оценка достоверности фидбека (Feedback Faithfulness)
        P, R, F1 = bert_score.score(
            [generated_feedback],
            [reference_answer],
            lang='ru',
            rescale_with_baseline=True
        )
        bert_faithfulness = float(F1[0])
        
        feedback_faithfulness = bert_faithfulness
        
        # Итоговое качество генерации
        generation_quality = (
            0.3 * answer_similarity +           # качество ответа
            0.3 * feedback_faithfulness +       # достоверность фидбека
            0.4 * feedback_uniqueness           # уникальность фидбека
        )
        
        # Нормализация итоговой оценки
        generation_quality = min(1.0, max(0.0, generation_quality))
        
        metrics = {
            'answer_similarity': answer_similarity,
            'feedback_uniqueness': feedback_uniqueness,
            'feedback_faithfulness': feedback_faithfulness,
            'generation_quality': generation_quality
        }
        
        logger.info(f"Метрики оценки: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Ошибка при оценке генерации: {str(e)}", exc_info=True)
        return {
            'answer_similarity': 0.0,
            'feedback_uniqueness': 0.0,
            'feedback_faithfulness': 0.0,
            'generation_quality': 0.0
        } 