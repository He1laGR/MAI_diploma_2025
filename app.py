import gradio as gr
from typing import Dict, List, Optional
from generation.question_retriever import get_random_question
from evaluation.answer_evaluation import evaluate_answer
from evaluation.llm_feedback import LLMFeedbackGenerator
import random

# Инициализация генератора фидбека
feedback_generator = LLMFeedbackGenerator("model/Vikhr-7B-instruct-Q4_K_M.gguf")

class InterviewSession:
    def __init__(self):
        self.category = None
        self.questions = []
        self.current_question_index = 0
        self.answers = []
        self.reference_answers = []  # TODO: добавить эталонные ответы
        self.feedback = []  # Сохраняем фидбек для каждого ответа
        
    def start_session(self, category: str) -> str:
        """Начало сессии интервью"""
        self.category = category
        self.questions = []
        self.current_question_index = 0
        self.answers = []
        self.feedback = []
        
        # Генерируем вопросы
        # 3 теоретических вопроса
        for _ in range(3):
            difficulty = random.choice(["Easy", "Medium", "High"])
            question_data = get_random_question(
                category=category,
                difficulty=difficulty,
                question_type="theory"
            )
            if question_data:
                self.questions.append(question_data['question'])
        
        # 1 case вопрос
        difficulty = random.choice(["Easy", "Medium", "High"])
        case_question_data = get_random_question(
            category=category,
            difficulty=difficulty,
            question_type="case"
        )
        if case_question_data:
            self.questions.append(case_question_data['question'])
            
        # 1 code вопрос
        difficulty = random.choice(["Easy", "Medium", "High"])
        code_question_data = get_random_question(
            category=category,
            difficulty=difficulty,
            question_type="code"
        )
        if code_question_data:
            self.questions.append(code_question_data['question'])
            
        if not self.questions:
            return "Не удалось сгенерировать вопросы. Попробуйте позже."
        return f"Вопрос 1 из {len(self.questions)}:\n{self.questions[0]}"
    
    def get_next_question(self) -> Optional[str]:
        self.current_question_index += 1
        if self.current_question_index < len(self.questions):
            q_idx = self.current_question_index + 1
            total = len(self.questions)
            return f"Вопрос {q_idx} из {total}:\n{self.questions[self.current_question_index]}"
        return None
    
    def submit_answer(self, answer: str) -> Dict:
        if not answer:
            return {"error": "Ответ не может быть пустым"}
        self.answers.append(answer)
        reference_answer = "Эталонный ответ"  # Временное решение
        
        # Определяем тип вопроса
        question_type = "theory"
        if self.current_question_index == 3:
            question_type = "case"
        elif self.current_question_index == 4:
            question_type = "code"
            
        scores = evaluate_answer(answer, reference_answer)
        feedback = feedback_generator.generate_feedback(
            answer=answer,
            reference_answer=reference_answer,
            key_terms=[],  # TODO: добавить ключевые термины
            category=self.category,
            question_type=question_type,
            scores=scores
        )
        self.feedback.append(feedback)
        
        # Если это последний вопрос, возвращаем все фидбеки
        if self.current_question_index >= len(self.questions) - 1:
            score_text = f"""
Оценка ответа:
- Семантическое сходство: {scores['bert_f1']:.2f}
- Покрытие терминов: {scores['tfidf_coverage']:.2f}
- Итоговая оценка: {scores['final_score']:.2f}
"""
            return {
                "is_last": True,
                "feedback": f"Интервью завершено!\n\n{feedback}\n\n{score_text}",
                "next_question": ""
            }
            
        # Иначе возвращаем следующий вопрос
        next_question = self.get_next_question()
        score_text = f"""
Оценка ответа:
- Семантическое сходство: {scores['bert_f1']:.2f}
- Покрытие терминов: {scores['tfidf_coverage']:.2f}
- Итоговая оценка: {scores['final_score']:.2f}
"""
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