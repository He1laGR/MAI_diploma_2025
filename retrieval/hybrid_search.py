import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
import json
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from requests.exceptions import RequestException

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройки
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
ELASTIC_URL = 'http://localhost:9200'
MODEL_SERVICE_URL = 'http://localhost:5000'
COLLECTION_NAME = 'interview_questions'
INDEX_NAME = 'interview_questions'
TOP_K = 10
BATCH_SIZE = 8

class ServiceClients:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Инициализация клиентов"""
        try:
            start_time = time.time()
            
            # Проверка доступности сервиса моделей
            try:
                response = requests.get(f"{MODEL_SERVICE_URL}/health")
                if response.status_code != 200:
                    raise ConnectionError("Сервис моделей недоступен. Запустите его командой: ./retrieval/start_model_manager.sh")
            except RequestException as e:
                raise ConnectionError(f"Сервис моделей недоступен. Запустите его командой: ./retrieval/start_model_manager.sh\nОшибка: {str(e)}")
            
            # Инициализация клиентов
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.elastic = Elasticsearch(ELASTIC_URL)
            
            logger.info(f"Service clients initialized successfully in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing service clients: {str(e)}")
            raise

def get_embedding(text: str) -> np.ndarray:
    """Получение эмбеддинга через сервис моделей"""
    try:
        response = requests.post(
            f"{MODEL_SERVICE_URL}/encode",
            json={"text": text},
            timeout=5
        )
        response.raise_for_status()
        return np.array(response.json()['embedding'])
    except RequestException as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ранжирование вопросов через сервис моделей"""
    if not candidates:
        return []
    
    try:
        candidate_texts = [c['payload']['question'] for c in candidates]
        response = requests.post(
            f"{MODEL_SERVICE_URL}/rerank",
            json={
                "query": query,
                "candidates": candidate_texts
            },
            timeout=10
        )
        response.raise_for_status()
        scores = response.json()['scores']
        
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = score
        
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    except RequestException as e:
        logger.error(f"Error during reranking: {str(e)}")
        return candidates

def get_elastic_query(query: str, category: Optional[str], question_type: Optional[str], difficulty: Optional[str]):
    """Генерация Elasticsearch запроса"""
    must_clauses = [
        {
            "multi_match": {
                "query": query,
                "fields": ["question^3", "category^2", "explanation"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
    ]
    
    filter_clauses = []
    if category:
        filter_clauses.append({"term": {"category.keyword": category.lower()}})
    if question_type:
        filter_clauses.append({"term": {"question_type.keyword": question_type.lower()}})
    if difficulty:
        filter_clauses.append({"term": {"difficulty.keyword": difficulty.lower()}})
    
    return {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        },
        "size": TOP_K,
        "_source": ["question", "category", "question_type", "difficulty"],
        "track_total_hits": False  
    }

def parallel_search(query: str, query_emb: np.ndarray, category: str, question_type: str, difficulty: str):
    """Параллельное выполнение поиска"""
    clients = ServiceClients()
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Qdrant поиск
        qdrant_future = executor.submit(
            clients.qdrant.search,
            collection_name=COLLECTION_NAME,
            query_vector=query_emb,
            limit=TOP_K,
            with_payload=True
        )
        
        # Elasticsearch поиск
        elastic_future = executor.submit(
            clients.elastic.search,
            index=INDEX_NAME,
            body=get_elastic_query(query, category, question_type, difficulty)
        )
        
        return qdrant_future.result(), elastic_future.result()

def validate_inputs(query: str, category: Optional[str], question_type: Optional[str], difficulty: Optional[str]) -> bool:
    """Валидация входных параметров"""
    if not query or not query.strip():
        logger.error("Пустой поисковый запрос")
        return False
        
    valid_difficulties = {'easy', 'medium', 'hard', 'high'}
    if difficulty and difficulty.lower() not in valid_difficulties:
        logger.error(f"Некорректная сложность. Допустимые значения: {valid_difficulties}")
        return False
        
    return True

def get_random_questions(
    top_k: int = TOP_K,
    category: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Получение случайных вопросов с учетом фильтров"""
    try:
        clients = ServiceClients()
        
        # Формируем базовый запрос
        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": []
                }
            }
        }
        
        # Добавляем фильтры с учетом регистра
        if category:
            query["query"]["bool"]["must"].append({
                "match": {
                    "category": category.title()  
                }
            })
        if question_type:
            query["query"]["bool"]["must"].append({
                "match": {
                    "question_type": question_type.lower() 
                }
            })
        if difficulty:
            difficulty = difficulty.title()
            query["query"]["bool"]["must"].append({
                "match": {
                    "difficulty": difficulty
                }
            })
            
        # Если нет фильтров, используем match_all
        if not query["query"]["bool"]["must"]:
            query["query"] = {"match_all": {}}
            
        # Добавляем сортировку по случайному значению
        query["sort"] = [{"_script": {
            "script": "Math.random()",
            "type": "number",
            "order": "asc"
        }}]
        
        logger.info(f"Elasticsearch query: {json.dumps(query, indent=2)}")
        results = clients.elastic.search(index=INDEX_NAME, body=query)
        
        # Преобразуем результаты в нужный формат
        random_questions = [
            {
                'id': hit['_id'],
                'score': hit['_score'],
                'payload': hit['_source']
            }
            for hit in results['hits']['hits']
        ]
        
        logger.info(f"Found {len(random_questions)} random questions")
        return random_questions
        
    except Exception as e:
        logger.error(f"Error getting random questions: {str(e)}", exc_info=True)
        return []

def hybrid_search(
    query: Optional[str] = None,
    top_k: int = TOP_K,
    category: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Гибридный поиск с поддержкой случайной выборки"""
    try:
        # Если запрос пустой, возвращаем случайные вопросы
        if not query:
            return get_random_questions(top_k, category, question_type, difficulty)
        
        # Валидация
        if not validate_inputs(query, category, question_type, difficulty):
            return []
        
        # Получаем эмбеддинг запроса
        query_emb = get_embedding(query)
        
        # Параллельный поиск
        qdrant_res, elastic_res = parallel_search(query, query_emb, category, question_type, difficulty)
        
        # Обработка результатов Qdrant
        qdrant_results = [
            {'id': hit.id, 'score': hit.score, 'payload': hit.payload}
            for hit in qdrant_res
        ]
        
        # Обработка результатов Elasticsearch
        elastic_results = [
            {'id': hit['_id'], 'score': hit['_score'], 'payload': hit['_source']}
            for hit in elastic_res['hits']['hits']
        ]
        
        # Объединение результатов
        merged = {r['id']: r for r in qdrant_results + elastic_results}
        merged_results = list(merged.values())
        
        # Фильтрация по метаданным с учетом релевантности
        filtered = []
        for r in merged_results:
            score = r['score']
            payload = r['payload']
            
            # Проверяем соответствие категории
            if category and payload.get('category', '').lower() != category.lower():
                continue
            
            # Проверяем соответствие типа вопроса
            if question_type and payload.get('question_type', '').lower() != question_type.lower():
                continue
            
            # Проверяем соответствие сложности
            if difficulty and payload.get('difficulty', '').lower() != difficulty.lower():
                continue
            
            # Увеличиваем score для более точных совпадений
            if category and payload.get('category', '').lower() == category.lower():
                score *= 1.5  # Буст за точное совпадение категории
            if difficulty and payload.get('difficulty', '').lower() == difficulty.lower():
                score *= 1.5  # Буст за точное совпадение сложности
            
            r['score'] = score
            filtered.append(r)
        
        # Ранжирование
        return rerank(query, filtered[:top_k])
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return []

if __name__ == "__main__":
    try:
        query = input("Введите поисковый запрос: ")
        category = input("Категория (оставьте пустым для всех): ") or None
        question_type = input("Тип вопроса (оставьте пустым для всех): ") or None
        difficulty = input("Сложность (оставьте пустым для всех): ") or None
        
        results = hybrid_search(query, category=category, question_type=question_type, difficulty=difficulty)
        
        if not results:
            print("Результаты не найдены")
            sys.exit(0)
            
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['payload']['question']} (Категория: {r['payload'].get('category', '')})")
            print(f"   Тип: {r['payload'].get('question_type', '')}, Сложность: {r['payload'].get('difficulty', '')}")
            print(f"   Rerank score: {r['rerank_score']:.3f}\n")
            
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)
