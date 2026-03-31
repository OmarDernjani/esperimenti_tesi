from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import extract_code, evaluate_code


def _rewrite_chain(model: str):
    template = ChatPromptTemplate([
        ("system",
         "You are a prompt engineer. Rewrite the following competitive programming problem "
         "to be clearer and more precise, preserving all constraints, examples, and requirements. "
         "Return only the rewritten problem description, no code, no explanations."),
        ("human", "{question}"),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def run_baseline(
    question: str,
    io_data: dict,
    solver_chain,
    model_optimizer: str,
) -> dict:
    print("\n[Baseline] Riscrittura prompt …")
    rewritten_prompt = _rewrite_chain(model_optimizer).invoke({"question": question})

    print("[Baseline] Generazione codice …")
    code = extract_code(solver_chain.invoke({"user_prompt": rewritten_prompt}))
    accuracy = evaluate_code(code, io_data)

    print(f"[Baseline] Accuracy: {accuracy:.3f}")
    return {"rewritten_prompt": rewritten_prompt, "code": code, "accuracy": accuracy}
