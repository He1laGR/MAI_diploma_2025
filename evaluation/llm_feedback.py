from typing import Dict, List, Optional
from llama_cpp import Llama
import json
import logging
import os
import re
from evaluation.answer_evaluation import tfidf_evaluation

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
            model_path: Путь к модели
        """
        if self._model is None:
            logging.info("Инициализация модели Vikhr")
            try:
                # Сначала пробуем загрузить локально
                if model_path and os.path.exists(model_path):
                    logging.info(f"Попытка загрузки локальной модели из {model_path}")
                    self._model = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_batch=512,
                        n_threads=8, 
                        n_gpu_layers=24,
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
                        n_ctx=4096,
                        n_batch=512,
                        n_threads=8,  
                        n_gpu_layers=24,
                        verbose=False
                    )
                    logging.info("Модель Vikhr загружена через from_pretrained")
            except Exception as e:
                logging.error(f"Ошибка при загрузке модели: {str(e)}")
                raise ValueError(f"Не удалось загрузить модель: {str(e)}")
        
    def _extract_key_phrases(self, text: str) -> set:
        """Извлекает ключевые фразы и термины из текста"""
        # Убираем знаки препинания и приводим к нижнему регистру
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Извлекаем слова длиннее 2 символов
        words = [word for word in text.split() if len(word) > 2]
        
        # Создаем множество уникальных слов
        word_set = set(words)
        
        # Извлекаем биграммы
        bigrams = set()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.add(bigram)
        
        return word_set.union(bigrams)
    
    def _check_content_relevance(self, user_answer: str, reference_answer: str) -> Dict[str, any]:
        """Проверяет релевантность ответа пользователя эталонному ответу на основе TF-IDF метрик"""
        
        # Извлекаем ключевые фразы для базового анализа
        user_phrases = self._extract_key_phrases(user_answer)
        ref_phrases = self._extract_key_phrases(reference_answer)
        
        # Находим пересечение фраз
        common_phrases = user_phrases.intersection(ref_phrases)
        
        # Вычисляем коэффициент пересечения фраз
        if len(ref_phrases) == 0:
            phrase_overlap_ratio = 0.0
        else:
            phrase_overlap_ratio = len(common_phrases) / len(ref_phrases)
        
        # Получаем TF-IDF метрики
        try:
            tfidf_metrics = tfidf_evaluation(user_answer, reference_answer)
            tfidf_similarity = tfidf_metrics['similarity']
            term_coverage = tfidf_metrics['term_coverage']
        except:
            tfidf_similarity = 0.0
            term_coverage = 0.0
        
        # Проверяем длину ответа
        word_count = len(user_answer.split())
        
        # Определяем тип ответа на основе метрик
        is_too_short = word_count < 5
        is_no_technical_terms = tfidf_similarity < 0.2 and term_coverage < 0.2
        is_low_relevance = phrase_overlap_ratio < 0.1 and tfidf_similarity < 0.1
        
        # Определяем нужен ли специальный промпт
        needs_special_prompt = is_too_short or is_no_technical_terms or is_low_relevance
        
        return {
            'word_count': word_count,
            'phrase_overlap_ratio': phrase_overlap_ratio,
            'tfidf_similarity': tfidf_similarity,
            'term_coverage': term_coverage,
            'common_phrases': common_phrases,
            'is_too_short': is_too_short,
            'is_no_technical_terms': is_no_technical_terms,
            'is_low_relevance': is_low_relevance,
            'needs_special_prompt': needs_special_prompt
        }

    def _create_prompt(
        self,
        answer: str,
        reference_answer: str,
        category: str,
        question_type: str,
        scores: Dict[str, float]
    ) -> str:
        """Создание промпта для генерации фидбека"""
        
        # Анализируем релевантность ответа
        relevance_check = self._check_content_relevance(answer, reference_answer)
        
        if relevance_check['needs_special_prompt']:
            # Промпт для нерелевантных/бессмысленных ответов
            system_prompt = """Ты — эксперт по проведению технических собеседований. 
Ответ кандидата не содержит релевантной технической информации по теме вопроса.
Твоя задача - дать честный и конструктивный фидбек.

Требования:
- Честно укажи, что ответ не соответствует теме вопроса
- Не придумывай технические термины, которых нет в ответе
- Будь конкретным и конструктивным
- Дай практические рекомендации для улучшения"""
            
            if relevance_check['is_too_short']:
                feedback_type = "слишком краткий"
            elif relevance_check['is_no_technical_terms']:
                feedback_type = "без технических терминов"
            else:
                feedback_type = "не по теме"
            
            user_prompt = f"""Ответ кандидата: "{answer}"
Эталонный ответ: "{reference_answer}"
Тип проблемы: ответ {feedback_type}
TF-IDF сходство: {relevance_check['tfidf_similarity']:.2f}
Покрытие терминов: {relevance_check['term_coverage']:.2f}

Сгенерируй честный фидбек:

Сильные стороны:
- Нет технически релевантных аспектов для оценки

Что улучшить:
- Ответ не содержит технической информации по теме вопроса
- Необходимо дать развернутый ответ, соответствующий теме

Рекомендации:
- Изучить основные концепции темы
- Дать конкретный технический ответ с примерами и объяснениями"""
        
        else:
            # Промпт для релевантных ответов
            system_prompt = """Ты — эксперт по проведению технических собеседований. 
Твоя задача - давать краткий и конкретный фидбек на русском языке, сравнивая ответ с эталонным ответом.

Требования к фидбеку:
- Всегда сравнивай ответ кандидата с эталонным ответом
- Кратко (не более 2 предложений в каждом пункте)  
- Конкретно (не используй общие слова, а конкретизируй)
- НЕ повторяй одни и те же мысли в разных формулировках
- НЕ генерируй больше 2 пунктов в каждом разделе
- НЕ приписывай того, чего нет в ответе кандидата
- Оценивай ТОЛЬКО то, что реально написано в ответе
- Если ответ неполный, укажи конкретно чего не хватает
- НЕ используй нумерацию пунктов
- Используй простые короткие предложения
- Рекомендации должны быть основаны на сравнении с эталонным ответом"""

            overlap_info = f"""Пересечение с эталоном: {relevance_check['phrase_overlap_ratio']:.1%}
TF-IDF сходство: {relevance_check['tfidf_similarity']:.2f}
Покрытие терминов: {relevance_check['term_coverage']:.2f}"""
            
            user_prompt = f"""Ответ кандидата: "{answer}"
Эталонный ответ: "{reference_answer}"
{overlap_info}

Структура фидбека:
Сильные стороны:
- Какие конкретные аспекты из эталонного ответа правильно отражены в ответе кандидата

Что улучшить:
- Какие важные концепции из эталонного ответа отсутствуют в ответе кандидата

Рекомендации:
- Как конкретно дополнить ответ, основываясь на эталонном ответе

Сгенерируй фидбек согласно структуре, сравнивая ответ кандидата с эталонным ответом."""

        prompt = f"<s>system\n{system_prompt}</s>\n<s>user\n{user_prompt}</s>\n<s>bot\n"
        return prompt

    def generate_feedback(
        self,
        answer: str,
        reference_answer: str,
        category: str,
        question_type: str,
        scores: Dict[str, float]
    ) -> str:
        """Генерация фидбека с помощью LLM"""
        logging.info("Создание промпта для генерации фидбека")
        prompt = self._create_prompt(
            answer=answer,
            reference_answer=reference_answer,
            category=category,
            question_type=question_type,
            scores=scores
        )
        
        logging.info("Начало генерации фидбека")
        response = self._model.create_completion(
            prompt=prompt,
            max_tokens=256,  
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