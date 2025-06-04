import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и специальных символов"""
    if not isinstance(text, str):
        return ""
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    # Удаляем специальные символы, оставляя пунктуацию
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def prepare_text_for_embedding(row: Dict[str, Any]) -> str:
    """Подготовка текста для эмбеддинга"""
    # Основной текст (вопрос + ключевые термины)
    main_text = f"{clean_text(row['question'])} {clean_text(row['key_terms'])}"
    
    # Дополнительный контекст (ответ + код)
    context = f"{clean_text(row['reference_answer'])} {clean_text(row['code_snippet'])}"
    
    # Метаданные для лучшего поиска
    metadata = f"{clean_text(row['category'])} {clean_text(row['question_type'])} {clean_text(row['difficulty'])}"
    
    # Комбинируем все части с разными весами
    return f"{main_text} {context} {metadata}"

# Загрузка данных
input_path = 'data/interview_questions.xlsx'
df = pd.read_excel(input_path)
df = df.astype(str).replace("nan", "")

# Гарантируем наличие всех нужных столбцов
required_fields = [
    'question', 'category', 'question_type', 'difficulty',
    'reference_answer', 'code_snippet', 'key_terms'
]

# Проверяем наличие всех полей
missing_fields = [field for field in required_fields if field not in df.columns]
if missing_fields:
    raise ValueError(f"Отсутствуют обязательные поля: {missing_fields}")

print("Загрузка модели для эмбеддингов...")
model = SentenceTransformer('BAAI/bge-m3')

print("Подготовка текстов...")
texts = df.apply(prepare_text_for_embedding, axis=1).tolist()

# Генерируем эмбеддинги
print("Генерация эмбеддингов...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=32
)

# Сохраняем эмбеддинги
print("Сохранение эмбеддингов...")
os.makedirs('embeddings', exist_ok=True)
np.save('embeddings/embeddings.npy', embeddings)

# Сохраняем метаданные
print("Сохранение метаданных...")
meta = df[required_fields].to_dict(orient='records')
with open('embeddings/meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Готово!") 