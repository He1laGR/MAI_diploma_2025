from answer_evaluation import (
    star_analysis,
    evaluate_star_completeness,
    bert_score_eval,
    tfidf_evaluation,
    evaluate_answer
)

def test_star_analysis():
    """Тест STAR-анализа"""
    print("\n=== Тест STAR-анализа ===")
    
    # Тестовый ответ с разными форматами
    test_answer = """
    Situation: В проекте была проблема с производительностью базы данных.
    Task - Необходимо было оптимизировать запросы и индексы.
    Action • Переписал сложные запросы, добавил индексы.
    Result: Улучшили время отклика на 50%.
    """
    
    star_parts = star_analysis(test_answer)
    print("\nВыделенные секции:")
    for label, content in star_parts.items():
        print(f"{label}: {content}")
    
    completeness = evaluate_star_completeness(star_parts)
    print(f"\nОценка полноты STAR: {completeness:.2f}")

def test_bert_score():
    """Тест BERTScore"""
    print("\n=== Тест BERTScore ===")
    
    user_answer = "Python использует сборщик мусора для автоматического освобождения памяти. Он отслеживает ссылки на объекты и удаляет те, на которые больше нет ссылок."
    reference_answer = "В Python управление памятью осуществляется через сборщик мусора, который автоматически освобождает память, когда объекты больше не используются."
    
    precision, recall, f1 = bert_score_eval(user_answer, reference_answer, return_all=True)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

def test_tfidf():
    """Тест TF-IDF оценки"""
    print("\n=== Тест TF-IDF оценки ===")
    
    user_answer = "В Python декораторы позволяют модифицировать функции. Они начинаются с @ и могут принимать аргументы."
    reference_answer = "Декораторы в Python - это функции, которые принимают другую функцию и расширяют её функциональность. Они обозначаются символом @."
    key_terms = ["декоратор", "функция", "python", "модификация"]
    
    scores = tfidf_evaluation(user_answer, reference_answer, key_terms)
    print("\nМетрики TF-IDF:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.2f}")

def test_full_evaluation():
    """Тест полной оценки ответа"""
    print("\n=== Тест полной оценки ===")
    
    user_answer = """
    Situation: В проекте была проблема с медленной загрузкой данных.
    Task: Необходимо оптимизировать процесс загрузки.
    Action: Реализовал кэширование и пагинацию.
    Result: Время загрузки уменьшилось на 70%.
    """
    
    reference_answer = """
    Situation: Проект страдал от медленной загрузки данных.
    Task: Требовалось оптимизировать процесс загрузки данных.
    Action: Внедрил систему кэширования и пагинацию результатов.
    Result: Удалось сократить время загрузки на 70%.
    """
    
    key_terms = ["оптимизация", "загрузка", "кэширование", "пагинация", "производительность"]
    
    evaluation = evaluate_answer(user_answer, reference_answer, key_terms)
    print("\nРезультаты оценки:")
    for metric, score in evaluation.items():
        print(f"{metric}: {score:.2f}")

if __name__ == "__main__":
    print("Тестирование модуля оценки ответов")
    print("=" * 50)
    
    test_star_analysis()
    test_bert_score()
    test_tfidf()
    test_full_evaluation() 