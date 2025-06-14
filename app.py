import gradio as gr
from typing import Dict, List, Optional, Tuple
from evaluation.answer_evaluation import evaluate_answer
from evaluation.llm_feedback import LLMFeedbackGenerator
from evaluation.llm_generation_evaluation import evaluate_llm_generation
from retrieval.hybrid_search import hybrid_search
import random
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация генератора фидбека
feedback_generator = LLMFeedbackGenerator("model/Vikhr-7B-instruct-Q4_K_M.gguf")

class InterviewSession:
    def __init__(self):
        self.category = None
        self.questions = []
        self.current_question_index = 0
        self.answers = []
        self.reference_answers = []
        self.feedback = []
        self.answer_scores = []  
        self.feedback_metrics = [] 
        self.is_completed = False
        self.passed_threshold = 0.7

    def start_session(self, category: str) -> str:
        """Начало сессии интервью"""
        self.category = category
        self.questions = []
        self.current_question_index = 0
        self.answers = []
        self.feedback = []
        self.answer_scores = []
        self.feedback_metrics = []
        self.is_completed = False
        
        # Генерируем вопросы
        # 3 теоретических вопроса
        for _ in range(3):
            difficulty = random.choice(["Easy", "Medium", "High"])
            questions = hybrid_search(
                query=None,
                top_k=1,
                category=category,
                difficulty=difficulty,
                question_type="theory"
            )
            if questions:
                self.questions.append(questions[0])
        
        # 1 case вопрос
        difficulty = random.choice(["Easy", "Medium", "High"])
        case_questions = hybrid_search(
            query=None,
            top_k=1,
            category=category,
            difficulty=difficulty,
            question_type="case"
        )
        if case_questions:
            self.questions.append(case_questions[0])
            
        # 1 code вопрос
        difficulty = random.choice(["Easy", "Medium", "High"])
        code_questions = hybrid_search(
            query=None,
            top_k=1,
            category=category,
            difficulty=difficulty,
            question_type="code"
        )
        if code_questions:
            self.questions.append(code_questions[0])
            
        if not self.questions:
            return "Не удалось сгенерировать вопросы. Попробуйте позже."
        return f"Вопрос 1 из {len(self.questions)}:\n{self.questions[0]['payload']['question']}"
    
    def get_next_question(self) -> Optional[str]:
        self.current_question_index += 1
        if self.current_question_index < len(self.questions):
            q_idx = self.current_question_index + 1
            total = len(self.questions)
            return f"Вопрос {q_idx} из {total}:\n{self.questions[self.current_question_index]['payload']['question']}"
        return None
    
    def check_completion(self) -> Dict:
        """Проверяет, прошел ли пользователь интервью"""
        if not self.is_completed:
            return {
                "is_completed": False,
                "message": "Интервью еще не завершено"
            }
            
         # Считаем средние оценки ответов
        avg_bert_score = sum(s['bert_f1'] for s in self.answer_scores) / len(self.answer_scores)
        avg_tfidf_score = sum(s['tfidf_similarity'] for s in self.answer_scores) / len(self.answer_scores)
        avg_term_coverage = sum(s['term_coverage'] for s in self.answer_scores) / len(self.answer_scores)
        avg_final_score = sum(s['final_score'] for s in self.answer_scores) / len(self.answer_scores)
        
        # Считаем средние метрики фидбека
        avg_feedback_uniqueness = sum(m['feedback_uniqueness'] for m in self.feedback_metrics) / len(self.feedback_metrics)
        avg_feedback_faithfulness = sum(m['feedback_faithfulness'] for m in self.feedback_metrics) / len(self.feedback_metrics)
        avg_generation_quality = sum(m['generation_quality'] for m in self.feedback_metrics) / len(self.feedback_metrics)
        
        # Общая оценка фидбека
        avg_feedback = (
            avg_feedback_uniqueness + 
            avg_feedback_faithfulness +
            avg_generation_quality
        ) / 3
        
        # Проверяем, прошел ли пользователь интервью
        passed = avg_score >= self.passed_threshold
        
        return {
            "is_completed": True,
            "passed": passed,
            "avg_bert_score": avg_bert_score,
            "avg_tfidf_score": avg_tfidf_score,
            "avg_term_coverage": avg_term_coverage,
            "avg_final_score": avg_final_score,
            "avg_feedback": avg_feedback,
            "message": "Поздравляем! Вы успешно прошли интервью!" if passed else "К сожалению, вы не прошли интервью. Попробуйте еще раз!"
        }
    
    def submit_answer(self, answer: str) -> Dict:
        if not answer:
            return {"error": "Ответ не может быть пустым"}
            
        self.answers.append(answer)
        current_question = self.questions[self.current_question_index]['payload']
        reference_answer = current_question.get('reference_answer', 'Эталонный ответ')
        
        # Определяем тип вопроса
        question_type = "theory"
        if self.current_question_index == 3:
            question_type = "case"
        elif self.current_question_index == 4:
            question_type = "code"
            
        # Оцениваем ответ пользователя
        scores = evaluate_answer(
            user_answer=answer,
            reference_answer=reference_answer,
            question=current_question['question']
        )
        self.answer_scores.append(scores)
        
        # Генерируем фидбек
        feedback = feedback_generator.generate_feedback(
            answer=answer,
            reference_answer=reference_answer,
            category=self.category,
            question_type=question_type,
            scores=scores
        )
        self.feedback.append(feedback)
        
        # Оцениваем качество фидбека
        feedback_metrics = evaluate_llm_generation(
            generated_feedback=feedback,
            user_answer=answer,
            reference_answer=reference_answer
        )
        
        self.feedback_metrics.append({**feedback_metrics})
        
        # Формируем текст с оценками
        score_text = f"""
Оценка ответа:
- Семантическое сходство (BERTScore): {scores['bert_f1']:.2f}
- Сходство терминов (TF-IDF): {scores['tfidf_similarity']:.2f}
- Покрытие ключевых терминов: {scores['term_coverage']:.2f}
- Итоговая оценка: {scores['final_score']:.2f}

Оценка фидбека:
- Уникальность фидбека: {feedback_metrics['feedback_uniqueness']:.2f}
- Достоверность фидбека: {feedback_metrics['feedback_faithfulness']:.2f}
- Общее качество: {feedback_metrics['generation_quality']:.2f}
"""
        # Последний вопрос
        if self.current_question_index >= len(self.questions) - 1:
            self.is_completed = True  # Отмечаем интервью как завершенное
            completion_status = self.check_completion()
            
            score_text += f"""

Итоговый отчет:
- Средняя семантическая оценка: {completion_status['avg_bert_score']:.2f}
- Среднее сходство терминов: {completion_status['avg_tfidf_score']:.2f}
- Среднее покрытие терминов: {completion_status['avg_term_coverage']:.2f}
- Средняя итоговая оценка: {completion_status['avg_final_score']:.2f}
- Среднее качество фидбека: {completion_status['avg_feedback']:.2f}
- Статус: {completion_status['message']}
"""
            return {
                "is_last": True,
                "feedback": f"Интервью завершено!\n\n{feedback}\n\n{score_text}",
                "next_question": ""
            }
            
        next_question = self.get_next_question()
        return {
            "is_last": False,
            "feedback": f"{feedback}\n\n{score_text}",
            "next_question": next_question
        }

# Глобальная переменная для хранения сессии
current_session = None

def welcome_message() -> str:
    """Приветственное сообщение"""
    return """
    Добро пожаловать в систему подготовки к интервью!
    
    Выберите специализацию и нажмите "Начать интервью".
    Вам будет предложено ответить на 5 вопросов:
    
    После каждого ответа вы получите подробный фидбек.
    """

def select_category(category: str) -> tuple:
    """Обработка выбора категории"""
    global current_session
    current_session = InterviewSession()
    first_question = current_session.start_session(category)
    return "", first_question

def submit_answer(answer: str) -> tuple:
    """Обработка отправки ответа"""
    global current_session
    if not current_session:
        return "Ошибка: сессия не найдена. Пожалуйста, начните интервью заново.", ""
        
    # Проверяем, не завершено ли уже интервью
    if current_session.is_completed:
        return "Интервью уже завершено. Начните новую сессию.", ""
        
    result = current_session.submit_answer(answer)
    feedback = result["feedback"]
    next_question = result["next_question"]
    return feedback, next_question

# Создаем интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Система подготовки к интервью")
    gr.Markdown(welcome_message())
    
    with gr.Row():
        category = gr.Dropdown(
            choices=["Python", "Data Science"],
            label="Выберите специализацию"
        )
        start_btn = gr.Button("Начать интервью")
    
    with gr.Row():
        question = gr.Textbox(label="Вопрос", interactive=False)
        answer = gr.Textbox(label="Ваш ответ", lines=5)
    
    submit_btn = gr.Button("Отправить ответ")
    feedback = gr.Textbox(label="Фидбек", lines=10)
    
    start_btn.click(
        fn=select_category,
        inputs=[category],
        outputs=[feedback, question]
    )
    
    submit_btn.click(
        fn=submit_answer,
        inputs=[answer],
        outputs=[feedback, question]
    )

if __name__ == "__main__":
    demo.launch() 