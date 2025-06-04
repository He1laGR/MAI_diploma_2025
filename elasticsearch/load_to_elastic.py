import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch('http://localhost:9200')

with open('../embeddings/meta.json', encoding='utf-8') as f:
    meta = json.load(f)

def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, float):
        try:
            import math
            if math.isnan(val):
                return ""
        except Exception:
            pass
    if str(val).lower() == "nan":
        return ""
    return str(val)

actions = [
    {
        "_index": "interview_questions",
        "_id": i,
        "_source": {
            "question": safe_str(item.get("question")),
            "category": safe_str(item.get("category")),
            "question_type": safe_str(item.get("question_type")),
            "difficulty": safe_str(item.get("difficulty")),
            "reference_answer": safe_str(item.get("reference_answer")),
            "key_terms": safe_str(item.get("key_terms")),
            "code_snippet": safe_str(item.get("code_snippet"))
        }
    }
    for i, item in enumerate(meta)
]

try:
    helpers.bulk(es, actions)
except Exception as e:
    print("Ошибка bulk-загрузки:", e)
    if hasattr(e, 'errors'):
        for error in e.errors:
            print(json.dumps(error, ensure_ascii=False, indent=2))