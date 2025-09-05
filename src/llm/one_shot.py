from src.llm.ask_llm import LLMWrapper


def ask_question_one_shot(prompt, question, model="llama3.2"):
    llm = LLMWrapper(model=model)

    example = (
        "Example:\n"
        "x1 is on 10.0.0.1/24, x2 on 10.0.0.2/24 → both in 10.0.0.0/24\n\n"
        "Question: Which is the subnetwork that connects x1 to x2?\n"
        "Answer: 10.0.0.0/24\n"
    )

    full_prompt = f"{example}\n\nNow, based on the following topology:\n{prompt}\n\nQuestion: {question}\nAnswer:"
    return llm.ask(full_prompt)

