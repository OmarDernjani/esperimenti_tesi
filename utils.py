import re
import sys
import os
import math
import subprocess
import tempfile
from datasets import load_dataset
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_apps_dataset():
    APPS = load_dataset("json", data_files={
        "train": "hf://datasets/codeparrot/apps/train.jsonl",
        "test":  "hf://datasets/codeparrot/apps/test.jsonl",
    })
    return APPS["train"], APPS["test"]


def get_minibatch(data, n_per_difficulty: int = 10) -> list:
    difficulties = ["introductory", "interview", "competition"]
    samples = []
    for diff in difficulties:
        indices = [i for i, d in enumerate(data["difficulty"]) if d == diff][:n_per_difficulty]
        for i in indices:
            samples.append({
                "question":     data["question"][i],
                "difficulty":   data["difficulty"][i],
                "input_output": data["input_output"][i],
            })
    return samples


def build_solver_chain(model: str = "mistral-nemo"):
    template = ChatPromptTemplate([
        ("system",
         "You are a helpful assistant for coding. "
         "Write a complete, standalone Python script that solves the competitive programming problem described by the user. "
         "The script must read input from stdin (using input() or sys.stdin) and print the result to stdout. "
         "Do not include test cases, example usage, or explanations. "
         "Wrap the code in a markdown Python code block: ```python ... ```"),
        ("human", "{user_prompt}"),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def extract_code(response: str) -> str:
    """Estrae il blocco di codice Python da una risposta in markdown."""
    for pattern in [r"```python(.*?)```", r"```py(.*?)```", r"```(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    pipe_match = re.findall(r"\|([^|]+)\|", response, re.DOTALL)
    if pipe_match:
        candidates = [m.strip() for m in pipe_match if len(m.strip()) > 20]
        if candidates:
            return max(candidates, key=len)
    # fallback: rimuovi eventuali fence residue che il modello non ha chiuso
    code = response.strip()
    code = re.sub(r'^```(?:python|py)?\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    return code


def resampling(question: str, solver_chain, model: str = "llama3.1:8b", n_variants: int = 5) -> list:
    """Genera n_variants riformulazioni del problema e per ognuna invoca solver_chain."""
    template = ChatPromptTemplate([
        ("system",
         f"You are a prompt resampler. Given a competitive programming problem, generate EXACTLY {n_variants} "
         "complete rewordings of the ENTIRE problem. "
         "Each rewriting must:\n"
         "- Contain ALL the original information (constraints, input format, output format, examples)\n"
         "- Be a COMPLETE standalone problem description\n"
         "- Use different wording and sentence structure\n\n"
         "FORMAT (mandatory):\n"
         "|complete reworded problem 1|\n"
         "|complete reworded problem 2|\n"
         "...\n"
         "Output NOTHING outside the | | delimiters."),
        ("human", "{user_prompt}"),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    chain = template | llm | StrOutputParser()
    response = chain.invoke({"user_prompt": question})
    resampled = [item.strip() for item in response.split("|") if item.strip() and item.strip() != "\n\n"]
    # fallback: se il modello non ha usato i delimitatori, prova a splittare su numerazione (1. 2. ...)
    if len(resampled) < 2:
        resampled = re.split(r'\n\s*\d+\.\s+', response)
        resampled = [item.strip() for item in resampled if len(item.strip()) > 50]
    return [
        {"prompt": prompt, "code": extract_code(solver_chain.invoke({"user_prompt": prompt}))}
        for prompt in resampled
    ]


def _run_script(file_path: str, test_input: str) -> dict:
    try:
        proc = subprocess.Popen(
            [sys.executable, file_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        try:
            stdout, stderr = proc.communicate(input=test_input, timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return {"success": False, "error": "Timeout"}
        if proc.returncode == 0:
            return {"success": True, "output": stdout}
        return {"success": False, "error": stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def evaluate_code(code: str, io_data: dict) -> float:
    """Esegue il codice sui test case e restituisce la frazione di test passati."""
    inputs = io_data.get("inputs", [])
    if not inputs:
        return 0.0
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        passed = 0
        for inp, exp in zip(inputs, io_data.get("outputs", [])):
            result = _run_script(path, inp)
            if result["success"] and _normalize(result["output"]) == _normalize(exp):
                passed += 1
        return passed / len(inputs)
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def get_failing_tests(code: str, io_data: dict, max_failures: int = 3) -> list[dict]:
    """Restituisce i primi max_failures test case falliti (usati come segnale di errore in APO)."""
    inputs  = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    failures = []
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        for inp, exp in zip(inputs, outputs):
            if len(failures) >= max_failures:
                break
            result = _run_script(path, inp)
            if result["success"]:
                actual = _normalize(result["output"])
                if actual != _normalize(exp):
                    failures.append({"input": inp[:300], "expected": exp[:300], "actual": actual[:300]})
            else:
                failures.append({"input": inp[:300], "expected": exp[:300], "error": result["error"][:300]})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return failures


def compute_pass_at_k(accuracies: list[float], k: int) -> float:
    """pass@k = 1 - C(n-c, k) / C(n, k), dove c = campioni con accuracy == 1.0."""
    n = len(accuracies)
    c = sum(1 for a in accuracies if a == 1.0)
    if n == 0 or k > n:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
