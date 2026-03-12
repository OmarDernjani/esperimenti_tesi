from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import build_solver_chain, resampling, extract_code
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


def generate_delta_chain(delta_model: str = 'mistral-nemo'):
    """Genera la soluzione migliorata a partire dal codice + critica"""
    template = ChatPromptTemplate([
        ('system',
         'You are a code-savvy LLM who solves the problem denoted in the prompt '
         'and follows the instructions exactly to build the solution. '
         'put | before and after the script'),
        ('human', 'Original code:\n{code}\n\nCritique:\n{critique}')
    ])

    llm = ChatOllama(model=delta_model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def automatic_prompt_engineering(
        user_prompt: str,
        target_model: str = 'mistral-nemo',
        socratic_model: str = 'llama3.1:8b'
    ):

    solver_chain = build_solver_chain(target_model)
    baseline_response = solver_chain.invoke({'user_prompt': user_prompt})

    
    socratic_chain = generate_socratic_chain(socratic_model)
    critique = socratic_chain.invoke({'code': baseline_response})

    
    delta_chain = generate_delta_chain(target_model)
    improved_response = delta_chain.invoke({
        'code': baseline_response,
        'critique': critique
    })

    improved_code = extract_code(improved_response)

    resampled_variants = resampling(
        question = improved_response,
        solver_chain = solver_chain,
        model = os.getenv("MODEL_OPTIMIZER", socratic_model)
    )

    return resampled_variants, critique 