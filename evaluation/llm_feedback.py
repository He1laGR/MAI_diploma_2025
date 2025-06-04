from typing import Dict, List, Optional
from llama_cpp import Llama
import json
import logging
import os

class LLMFeedbackGenerator:
    _instance = None
    _model = None
    _model_path = None

    def __new__(cls, model_path: str = None):
        if cls._instance is None:
            cls._instance = super(LLMFeedbackGenerator, cls).__new__(cls)
            cls._model_path = model_path
        return cls._instance

    def __init__(self, model_path: str = None):
        """Инициализация модели Vikhr
        Args:
            model_path: Путь к модели (опционально)
        """
        if self._model is None:
            logging.info("Инициализация модели Vikhr")
            try:
                # Сначала пробуем загрузить локально, если путь указан
                if model_path and os.path.exists(model_path):
                    logging.info(f"Попытка загрузки локальной модели из {model_path}")
                    self._model = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_batch=512,
                        n_threads=4, 
                        n_gpu_layers=0,
                        verbose=False
                    )
                    logging.info("Модель Vikhr загружена локально")
                else:
                    # Если локальная модель не найдена, пробуем через from_pretrained
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    logging.info(f"Попытка загрузки модели через from_pretrained (кэш: {cache_dir})")
                    self._model = Llama.from_pretrained(
                        repo_id="oblivious/Vikhr-7B-instruct-GGUF",
                        filename="Vikhr-7B-instruct-Q4_K_M.gguf",
                        cache_dir=cache_dir,
                        n_ctx=2048,
                        n_batch=512,
                        n_threads=4,  
                        n_gpu_layers=0,
                        verbose=False
                    )
                    logging.info("Модель Vikhr загружена через from_pretrained")
            except Exception as e:
                logging.error(f"Ошибка при загрузке модели: {str(e)}")
                raise ValueError(f"Не удалось загрузить модель: {str(e)}")
        
    def _create_prompt(
        self,
        answer: str,
        reference_answer: str,
        key_terms: List[str],
        category: str,
        question_type: str,
        scores: Dict[str, float]
    ) -> str:
        """Создание промпта для генерации фидбека"""
        
        system_prompt = """Ты — эксперт по проведению технических собеседований. Твоя задача - давать краткий и конкретный фидбек на русском языке, сравнивая ответ с эталонным ответом.
Требования к фидбеку:
- Всегда сравнивай ответ кандидата с эталонным ответом
- Кратко (не более 2 предложений в каждом пункте)
- Конкретно (не используй общие слова, а конкретизируй)
- НЕ повторяй одни и те же мысли в разных формулировках
- НЕ генерируй больше 2 пунктов в каждом разделе
- НЕ приписывай того, чего нет в его ответе
- Оценивай ТОЛЬКО то, что реально написано в ответе
- Если ответ слишком краткий, укажи это в фидбеке
- НЕ используй нумерацию пунктов
- Используй простые короткие предложения
- НЕ используй скобки и дополнительные пояснения
- НЕ приписывай термины, которых нет в ответе
- НЕ используй сложные технические термины
- Рекомендации должны быть основаны на реальном ответе"""

        user_prompt = f"""Ответ: {answer}
Эталонный ответ: {reference_answer}
Ключевые термины: {', '.join(key_terms)}

Структура фидбека:
Сильные стороны:
- Какие аспекты из эталонного ответа затронуты

Что улучшить:
- Какие неточности в сравнении с эталонным ответом

Рекомендации:
- Как дополнить ответ, опираясь на эталонный ответ

Сгенерируй фидбек согласно структуре, сравнивая ответ с эталонным ответом."""

        # Форматируем промпт в формате Vikhr
        prompt = f"<s>system\n{system_prompt}</s>\n<s>user\n{user_prompt}</s>\n<s>bot\n"
        return prompt

    def generate_feedback(
        self,
        answer: str,
        reference_answer: str,
        key_terms: List[str],
        category: str,
        question_type: str,
        scores: Dict[str, float]
    ) -> str:
        """Генерация фидбека с помощью LLM"""
        logging.info("Создание промпта для генерации фидбека")
        prompt = self._create_prompt(
            answer=answer,
            reference_answer=reference_answer,
            key_terms=key_terms,
            category=category,
            question_type=question_type,
            scores=scores
        )
        
        logging.info("Начало генерации фидбека")
        response = self._model.create_completion(
            prompt=prompt,
            max_tokens=512,  
            temperature=0.1,  
            top_p=0.9,
            top_k=20, 
            min_p=0,
            stop=["</s>", "###", "END", "Конец", "```", "```python", "```python\n"], 
            repeat_penalty=1.2,  
            presence_penalty=0.2,  
            frequency_penalty=0.2 
        )
        
        logging.info("Фидбек сгенерирован")
        feedback = response["choices"][0]["text"].strip()
        
        # Проверка на пустой фидбек
        if not feedback:
            feedback = """Сильные стороны:
- Ответ слишком краткий для оценки

Что улучшить:
- Необходимо дать более развернутый ответ

Рекомендации:
- Объяснить разницу между методами
- Привести примеры использования"""
            
        return feedback 