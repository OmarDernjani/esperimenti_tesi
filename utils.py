import re
import math
import sys
from datasets import load_dataset
from sqlalchemy.orm import Session
from schemas import ProblemSchema, SolutionSchema
from models import Problem, Solution
from db import engine
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import subprocess

def load_apps_dataset():
    
    """Carica i file direttamente da Json (Dataset deprecato su HF)"""

    APPS = load_dataset(
        "json",
        data_files={
            "train": "hf://datasets/codeparrot/apps/train.jsonl",
            "test": "hf://datasets/codeparrot/apps/test.jsonl"
        }
    )
    return APPS['train'], APPS['test']



def get_problem(data, index: int = 0) -> dict:
    return {
        "question": data['question'][index],
        "difficulty": data['difficulty'][index],
        "input_output": data['input_output'][index]
    }



def build_solver_chain(model: str = 'mistral-nemo'):
    
    """Logica della chain per il modello target"""

    template = ChatPromptTemplate([
        ('system',
         'You are a helpful assistant for coding. '
         'Write a complete, standalone Python script that solves the competitive programming problem described by the user. '
         'The script must read input from stdin (using input() or sys.stdin) and print the result to stdout. '
         'Do not include test cases, example usage, or explanations. '
         'Wrap the code in a markdown Python code block: ```python ... ```'),
        ('human', '{user_prompt}')
    ])
    
    llm = ChatOllama(
        model=model,
        num_ctx=4096,
        num_keep = 0
    )

    return template | llm | StrOutputParser()



def extract_code(response: str) -> str:

    """Estrae il codice dal prompt"""

    # Prova markdown code block con python/py o generico
    for pattern in [r'```python(.*?)```', r'```py(.*?)```', r'```(.*?)```']:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    # Fallback: pipe delimiter (meno affidabile se il codice usa |)
    pipe_match = re.findall(r'\|([^|]+)\|', response, re.DOTALL)
    if pipe_match:
        candidates = [m.strip() for m in pipe_match if len(m.strip()) > 20]
        if candidates:
            return max(candidates, key=len)

    return response.strip()



def saving_data(
        variant_response: list,
        dataset: str,
        question: str,
        model_optimizer: str,
        model_target: str,
        difficulty: str,
        input_output: str,
        baseline_code: str,
        algorithm: str,
        accuracy: float = 0.0,
        critique: str | None = None
    ):

    """Salva i dati sulla sessione del DB"""

    with Session(engine) as session:
        try:
            problem_data = ProblemSchema(
                dataset=dataset,
                question=question,
                model_optimizer=model_optimizer,
                difficulty=difficulty,
                input_output=str(input_output),
                code=baseline_code,
                critique=critique
            )

            db_problem = Problem(**problem_data.model_dump())
            session.add(db_problem)
            session.flush()

            for item in variant_response:
                sol_data = SolutionSchema(
                    problem_id = db_problem.problem_id,
                    algorithm = algorithm,
                    prompt = item['prompt'],
                    model_target = model_target,
                    code = item["code"],
                    accuracy = accuracy
                )
                db_solution = Solution(**sol_data.model_dump())
                session.add(db_solution)

            session.commit()
            print(f" Salvato: problem_id={db_problem.problem_id}, {len(variant_response)} soluzioni")

        except Exception as e:
            session.rollback()
            print(f" Errore: {e}")
            raise


def resampling(question: str, solver_chain, model: str = 'llama3.1:8b', n_variants: int = 5) -> list:

    """Fa resempling del prompt in ingresso per n_varianti"""

    template_resampling = ChatPromptTemplate([
        ('system',
         f'You are a prompt resampler. Given a competitive programming problem, generate EXACTLY {n_variants} '
         'complete rewordings of the ENTIRE problem. '
         'Each rewriting must:\n'
         '- Contain ALL the original information (constraints, input format, output format, examples)\n'
         '- Be a COMPLETE standalone problem description\n'
         '- Use different wording and sentence structure\n\n'
         'FORMAT (mandatory):\n'
         '|complete reworded problem 1|\n'
         '|complete reworded problem 2|\n'
         '...\n'
         'Output NOTHING outside the | | delimiters.'),
        ('human', '{user_prompt}')
    ])

    llm_resampling = ChatOllama(
        model=model,
        num_ctx=4096,
        num_keep = 0
    )


    chain_res = template_resampling | llm_resampling | StrOutputParser()
    response_resampled = chain_res.invoke({'user_prompt': question})

    resampled_list = [
        item.strip()
        for item in response_resampled.split('|')
        if item.strip() and item.strip() != '\n\n'
    ]

    variant_response = []
    for variant in resampled_list:
        raw_response = solver_chain.invoke({"user_prompt": variant})
        code = extract_code(raw_response)
        variant_response.append({
            "variant": variant,
            "prompt" : variant,
            "raw": raw_response,
            "code": code
        })

    return variant_response

def get_minibatch(data, n_per_difficulty: int = 50) -> list:
    
    difficulties = ['introductory', 'interview', 'competition']
    samples = []

    for diff in difficulties:
        indices = [
            i for i, d in enumerate(data['difficulty']) 
            if d == diff
        ][:n_per_difficulty]
        
        for i in indices:
            samples.append(get_problem(data, index=i))
    
    return samples


def run_and_test_code(file_path: str, test_input: str) -> dict:
    """Esegue lo script passando lo standard input e catturando l'output."""
    try:
        proc = subprocess.Popen(
            [sys.executable, file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
        )
        try:
            stdout, stderr = proc.communicate(input=test_input, timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return {"success": False, "output": None, "error": "Timeout"}

        if proc.returncode == 0:
            return {"success": True, "output": stdout, "error": None}
        else:
            return {"success": False, "output": None, "error": stderr}

    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calcola la metrica pass@k.
    n: numero totale di campioni (soluzioni) generati per il problema.
    c: numero di campioni corretti (che passano tutti i test case).
    k: il valore 'k' per cui calcolare la probabilità (es. 1, 3, 5, 10).
    """

    if n - c < k:
        return 1.0
    
    if k > n:
        return 1.0

    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def AUTOMATIC_PROMPT_ENGINEERING():
    #use resampler
    #select top K% based with accuracy
    #return subset of K%
    pass