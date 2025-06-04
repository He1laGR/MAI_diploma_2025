import os
import logging
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from flask import Flask, request, jsonify
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройки
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
EMBEDDING_MODEL = 'BAAI/bge-m3'
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

app = Flask(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Инициализация моделей"""
        try:
            start_time = time.time()
            logger.info("Starting model initialization...")
            
            # Определение устройства
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Инициализация эмбеддера
            logger.info(f"Loading embedder model: {EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(
                EMBEDDING_MODEL,
                device=self.device,
                cache_folder=CACHE_DIR
            )
            logger.info(f"Embedder initialized on {self.embedder.device}")
            
            # Инициализация реранкера
            logger.info(f"Loading reranker model: {RERANKER_MODEL}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL,
                cache_dir=CACHE_DIR
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL,
                cache_dir=CACHE_DIR
            ).to(self.device)
            logger.info(f"Reranker initialized on {next(self.reranker_model.parameters()).device}")
            
            # Прогрев моделей
            logger.info("Warming up models...")
            self.embedder.encode("warming up")
            warmup_inputs = self.reranker_tokenizer("warming up", return_tensors="pt")
            warmup_inputs = {k: v.to(self.device) for k, v in warmup_inputs.items()}
            self.reranker_model(**warmup_inputs)
            
            init_time = time.time() - start_time
            logger.info(f"Models initialized in {init_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

# Создаем глобальный экземпляр менеджера моделей
logger.info("Creating ModelManager instance...")
model_manager = ModelManager()
logger.info("ModelManager instance created successfully")

@app.route('/encode', methods=['POST'])
def encode():
    """Эндпоинт для получения эмбеддинга"""
    try:
        data = request.get_json()
        text = data['text']
        embedding = model_manager.embedder.encode(text, normalize_embeddings=True)
        return jsonify({'embedding': embedding.tolist()})
    except Exception as e:
        logger.error(f"Error in encode endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/rerank', methods=['POST'])
def rerank_endpoint():
    """Эндпоинт для ранжирования"""
    try:
        data = request.get_json()
        query = data['query']
        candidates = data['candidates']
        
        pairs = [(query, c) for c in candidates]
        inputs = model_manager.reranker_tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            scores = model_manager.reranker_model(**inputs).logits.squeeze(-1).cpu().tolist()
            
        return jsonify({'scores': scores})
    except Exception as e:
        logger.error(f"Error in rerank endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
    logger.info("Flask server started successfully") 