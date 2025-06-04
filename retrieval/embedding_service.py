import os
import logging
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from flask import Flask, request, jsonify

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

class ModelService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Инициализация моделей
        start_time = time.time()
        
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device,
            cache_folder=CACHE_DIR
        )
        logger.info(f"Embedder initialized on {self.embedder.device}")
        
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
        self.embedder.encode("warming up")
        warmup_inputs = self.reranker_tokenizer("warming up", return_tensors="pt")
        warmup_inputs = {k: v.to(self.device) for k, v in warmup_inputs.items()}
        self.reranker_model(**warmup_inputs)
        
        init_time = time.time() - start_time
        logger.info(f"Models initialized in {init_time:.2f} seconds")

model_service = ModelService()

@app.route('/encode', methods=['POST'])
def encode():
    text = request.json['text']
    embedding = model_service.embedder.encode(text, normalize_embeddings=True)
    return jsonify({'embedding': embedding.tolist()})

@app.route('/rerank', methods=['POST'])
def rerank():
    query = request.json['query']
    candidates = request.json['candidates']
    
    # Подготовка батчей
    pairs = [(query, c) for c in candidates]
    inputs = model_service.reranker_tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Перенос на устройство
    inputs = {k: v.to(model_service.device) for k, v in inputs.items()}
    
    # Инференс
    with torch.no_grad(), torch.amp.autocast('cuda'):
        scores = model_service.reranker_model(**inputs).logits.squeeze(-1)
    
    return jsonify({'scores': scores.cpu().tolist()})

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='localhost', port=5000) 