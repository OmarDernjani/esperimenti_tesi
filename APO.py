from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import build_solver_chain, resampling
import os


def generate_socratic_chain(model: str = 'llama3.1:8b'):
    """Critica il codice e suggerisce miglioramenti"""
    template = ChatPromptTemplate([
        ('system',
         'You are an LLM who critiques the code you receive as a prompt, '
         'constructing solid arguments that would lead the user to solve the problem. '
         'Address and compile a clear and concise list of all the problems you see in the code.'),
        ('human', '{code}')
    ])

    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def generate_improved_prompt_chain(delta_model: str = 'mistral-nemo'):
    """Genera un prompt migliorato a partire dal problema originale, il codice baseline e la critica"""
    template = ChatPromptTemplate([
        ('system',
         'You are a prompt engineer. Given a competitive programming problem, a baseline solution, '
         'and a critique of that solution, rewrite the problem description to be clearer and more precise, '
         'so that an LLM would be guided to avoid the issues highlighted in the critique. '
         'Return only the rewritten problem description, no code, no explanations.'),
        ('human', 'Problem:\n{user_prompt}\n\nBaseline code:\n{code}\n\nCritique:\n{critique}')
    ])

    llm = ChatOllama(model=delta_model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def automatic_prompt_engineering(
        user_prompt: str,
        baseline_code: str,
        target_model: str = 'mistral-nemo',
        socratic_model: str = 'llama3.1:8b',
        n_variants: int = 5
    ):

    solver_chain = build_solver_chain(target_model)

    socratic_chain = generate_socratic_chain(socratic_model)
    critique = socratic_chain.invoke({'code': baseline_code})

    delta_chain = generate_improved_prompt_chain(target_model)
    improved_prompt = delta_chain.invoke({
        'user_prompt': user_prompt,
        'code': baseline_code,
        'critique': critique
    })

    resampled_variants = resampling(
        question=improved_prompt,
        solver_chain=solver_chain,
        model=os.getenv("MODEL_OPTIMIZER", socratic_model),
        n_variants=n_variants
    )

    return resampled_variants, critique