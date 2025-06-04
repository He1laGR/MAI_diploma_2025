from qdrant_client import QdrantClient
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(embeddings: np.ndarray, meta: List[Dict[str, Any]]) -> bool:
    """Проверка корректности данных перед загрузкой"""
    if len(embeddings) != len(meta):
        logger.error(f"Несоответствие размеров: {len(embeddings)} эмбеддингов и {len(meta)} метаданных")
        return False
    
    # Проверка обязательных полей в метаданных
    required_fields = {'question', 'category', 'question_type', 'difficulty'}
    for i, item in enumerate(meta):
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            logger.error(f"Отсутствуют обязательные поля {missing_fields} в элементе {i}")
            return False
    
    return True

def load_to_qdrant(embeddings_path: str, meta_path: str, collection_name: str = 'interview_questions'):
    """Загрузка данных в Qdrant с обработкой ошибок"""
    try:
        # Проверка существования файлов
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Файл с эмбеддингами не найден: {embeddings_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Файл с метаданными не найден: {meta_path}")

        # Загрузка данных
        logger.info("Загрузка эмбеддингов и метаданных...")
        embeddings = np.load(embeddings_path)
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)

        # Валидация данных
        if not validate_data(embeddings, meta):
            raise ValueError("Ошибка валидации данных")

        # Подключение к Qdrant
        logger.info("Подключение к Qdrant...")
        client = QdrantClient(host='localhost', port=6333)

        # Создание коллекции
        logger.info(f"Создание коллекции {collection_name}...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "size": int(embeddings.shape[1]),
                "distance": "Cosine"
            }
        )

        # Загрузка данных
        logger.info("Загрузка данных в Qdrant...")
        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=meta,
            ids=None,
            batch_size=64
        )
        
        logger.info("Загрузка успешно завершена!")
        return True

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return False

if __name__ == "__main__":
    embeddings_path = '../embeddings/embeddings.npy'
    meta_path = '../embeddings/meta.json'
    
    success = load_to_qdrant(embeddings_path, meta_path)
    sys.exit(0 if success else 1) 