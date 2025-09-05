from src.llm.ask_llm import LLMWrapper


def ask_question_zero_shot(prompt, question, model="llama3.2"):
    llm = LLMWrapper(model=model)
    full_prompt = f"{prompt}\n\nPregunta: {question}\nRespuesta:"
    return llm.ask(full_prompt)


