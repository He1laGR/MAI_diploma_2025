from elasticsearch import Elasticsearch

es = Elasticsearch('http://localhost:9200')

# Удаляем индекс если он существует
if es.indices.exists(index="interview_questions"):
    es.indices.delete(index="interview_questions")

# Создаем индекс с правильными маппингами
mapping = {
    "mappings": {
        "properties": {
            "question": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "category": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "question_type": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "difficulty": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "reference_answer": {
                "type": "text",
                "analyzer": "standard"
            },
            "key_terms": {
                "type": "text",
                "analyzer": "standard"
            },
            "code_snippet": {
                "type": "text",
                "analyzer": "standard"
            }
        }
    },
    "settings": {
        "analysis": {
            "analyzer": {
                "standard": {
                    "type": "standard",
                    "stopwords": "_none_"
                }
            }
        }
    }
}

# Создаем индекс
es.indices.create(index="interview_questions", body=mapping)
print("Индекс успешно создан с правильными маппингами") 