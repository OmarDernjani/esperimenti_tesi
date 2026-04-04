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
from langchain_core.runnables import RunnableLambda


def load_apps_dataset():
    APPS = load_dataset("json", data_files={
        "train": "hf://datasets/codeparrot/apps/train.jsonl",
        "test":  "hf://datasets/codeparrot/apps/test.jsonl",
    })
    return APPS["train"], APPS["test"]


def load_humaneval_dataset():
    ds = load_dataset("openai/openai_humaneval")
    return ds["test"]


def get_humaneval_sample(data, n: int = 20, seed: int = 42) -> list:
    import random as _random
    _random.seed(seed)
    indices = list(range(len(data)))
    chosen = _random.sample(indices, min(n, len(indices)))
    return [
        {
            "task_id":      data[i]["task_id"],
            "prompt":       data[i]["prompt"],
            "test":         data[i]["test"],
            "entry_point":  data[i]["entry_point"],
        }
        for i in chosen
    ]


def get_minibatch(
    data,
    n_per_difficulty: dict | int = 10,
    min_test_cases: int = 5,
    seed: int = 42,
) -> list:
    
    import json as _json
    import random as _random
    _random.seed(seed)

    #mappa i differenti valori per difficoltà
    difficulties = ["introductory", "interview", "competition"]

    if isinstance(n_per_difficulty, int):

        #se n_per_difficulty è un int allora assegna a ogni key difficoltà il valore di int
        n_map = {d: n_per_difficulty for d in difficulties}
    else:

        # se è un dizionario assegna i valori personalizzati (utilizzo nel caso di sample sbilanciati)
        n_map = n_per_difficulty

    samples = []
    for diff in difficulties:

        #se trova la difficoltà ritorna il numero di sample associato alla difficoltà es: {'introductory': 10}
        #altrimenti, se non trova la difficoltà, ritorna 10 come output standard
        n = n_map.get(diff, 10)

        candidates = []

        for i, d in enumerate(data["difficulty"]):

            #blocco che gestisce il caso in cui una key non è una difficulty, ricarica i dati
            if d != diff:
                continue
            try:
                io = _json.loads(data["input_output"][i])
                n_tc = len(io.get("inputs", []))
            except Exception:
                n_tc = 0
            
            #se il numero dei test case delle singole entrate del dataset >= min_test_cases allora aggiungo ai candidati
            if n_tc >= min_test_cases:
                candidates.append(i)
        
        #candidati scelti randomicamente nella lista dei candidati
        chosen = _random.sample(candidates, min(n, len(candidates)))

        #aggiunge ai sample gli scelti randomicamente, dove sample è del tipo:
        #sample = {'question': question, difficulty = 'introductory/...', input_output = 'IO' (test cases)}
        for i in chosen:
            samples.append({
                "question":     data["question"][i],
                "difficulty":   data["difficulty"][i],
                "input_output": data["input_output"][i],
            })
    return samples


def build_direct_chain(model: str = "llama3.1:8b", call_based: bool = False, fn_name: str = ""):

    if call_based:
        system_msg = (
            "You are a helpful assistant for coding. "
            "Implement ONLY the Python function described by the user. "
            f"The function MUST be named exactly `{fn_name}`. "
            "Do NOT read from stdin or print anything. "
            "Do NOT include test cases, example usage, or explanations. "
            "Wrap the code in a markdown Python code block: ```python ... ```"
        )
    else:
        system_msg = (
            "You are a helpful assistant for coding. "
            "Write a complete, standalone Python script that solves the competitive programming problem described by the user. "
            "Read input with `import sys` and `data = sys.stdin.read()`, then parse it "
            "(e.g. `tokens = data.split()` for numeric problems, or `lines = data.splitlines()` for line-oriented problems). "
            "NEVER call `sys.stdin.read()`, `sys.stdin.readline()`, `sys.stdin.readlines()`, or `input()` more than once. "
            "Process the data, print the output, and exit cleanly. "
            "Do not include test cases, example usage, or explanations. "
            "Wrap the code in a markdown Python code block: ```python ... ```"
        )

    template = ChatPromptTemplate([
        ("system", system_msg),
        ("human", "{user_prompt}"),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def build_solver_chain(model_optimizer: str = "mistral-nemo", model_target: str = "llama3.1:8b", call_based: bool = False, fn_name: str = ""):

    if call_based:
        strict_rules = (
            "- Instruct the target LLM to implement ONLY the Python function described.\n"
            f"- Instruct the target LLM that the function MUST be named exactly `{fn_name}`.\n"
            "- Instruct the target LLM to NOT read from stdin or print anything.\n"
        )
    else:
        strict_rules = (
            "- Instruct the target LLM to write a complete, standalone Python script.\n"
            "- Instruct the target LLM to read input with `import sys` and `data = sys.stdin.read()`, then parse it.\n"
            "- Instruct the target LLM to NEVER call `sys.stdin.read()`, `sys.stdin.readline()`, `sys.stdin.readlines()`, or `input()` more than once.\n"
            "- Instruct the target LLM to process the data, print the output, and exit cleanly.\n"
        )
    
    
    common_rules = (
        "- Instruct the target LLM to NOT include test cases, example usage, or explanations in the final output.\n"
        "- Instruct the target LLM to wrap the code in a markdown Python code block: ```python ... ```\n"
    )

    
    system_msg = (
        "You are an expert prompt engineer.\n\n"
        "Your task is to write a prompt that will help a target LLM to solve a coding problem correctly.\n\n"
        "The prompt you write MUST:\n"
        "- Clearly restate the problem to be solved\n"
        "- Encourage step-by-step reasoning\n"
        "- Ask for efficient code\n\n"
        "CRITICAL: To fit our execution pipeline, the prompt you generate MUST ALSO strictly command the target LLM to follow these rules:\n"
        f"{strict_rules}"
        f"{common_rules}\n"
        "Return ONLY the final prompt."
    )

    template = ChatPromptTemplate([
        ("system", system_msg),
        ("human", "Problem:\n{user_prompt}"),
    ])

    llm_opt = ChatOllama(model=model_optimizer, num_ctx=4096, num_keep=0)
    prompt_chain = template | llm_opt | StrOutputParser()

    target_template = ChatPromptTemplate([
        ("human", "{generated_prompt}"),
    ])
    llm_tgt = ChatOllama(model=model_target, num_ctx=4096, num_keep=0)
    target_chain = target_template | llm_tgt | StrOutputParser()

    return prompt_chain | RunnableLambda(lambda p: target_chain.invoke({"generated_prompt": p}))


def extract_code(response: str) -> str:
    
    for pattern in [r"```python(.*?)```", r"```py(.*?)```", r"```(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    pipe_match = re.findall(r"\|([^|]+)\|", response, re.DOTALL)
    if pipe_match:
        candidates = [m.strip() for m in pipe_match if len(m.strip()) > 20]
        if candidates:
            return max(candidates, key=len)
        
    code = response.strip()
    code = re.sub(r'^```(?:python|py)?\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    return code


def resampling(question: str, solver_chain, model: str = "llama3.1:8b", n_variants: int = 5) -> list:
    
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

    if len(resampled) < 2:
        resampled = re.split(r'\n\s*\d+\.\s+', response)
        resampled = [item.strip() for item in resampled if len(item.strip()) > 50]

    resampled = resampled[:n_variants]
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


def _run_call_based(code: str, fn_name: str, args: list) -> dict:
    import json as _json
    harness = code + "\n\n"
    harness += f"import json as __json, sys as __sys\n"
    harness += f"__args = __json.loads(__sys.stdin.read())\n"
    harness += f"__result = {fn_name}(*__args)\n"
    harness += f"print(__json.dumps(__result))\n"
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(harness)
        proc = subprocess.Popen(
            [sys.executable, path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        try:
            stdout, stderr = proc.communicate(input=_json.dumps(args), timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate()
            return {"success": False, "error": "Timeout"}
        if proc.returncode == 0:
            try:
                return {"success": True, "output": _json.loads(stdout.strip())}
            except _json.JSONDecodeError:
                return {"success": True, "output": stdout.strip()}
        return {"success": False, "error": stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


# ── HumanEval assertion helpers ──────────────────────────────────────

def _extract_assertions(test_code: str) -> list[str]:
    """Extract individual assert lines from HumanEval check(candidate) function."""
    assertions = []
    in_check = False
    for line in test_code.split('\n'):
        stripped = line.strip()
        if 'def check(' in stripped:
            in_check = True
            continue
        if in_check and stripped.startswith('assert '):
            assertions.append(stripped)
    return assertions


def _run_single_assertion(code: str, assertion: str, entry_point: str) -> dict:
    """Run one HumanEval assertion against generated code."""
    assertion_fixed = assertion.replace('candidate', entry_point)
    script = code + "\n\n" + assertion_fixed + "\n"
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)
        proc = subprocess.Popen(
            [sys.executable, path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate()
            return {"success": False, "error": "Timeout"}
        if proc.returncode == 0:
            return {"success": True}
        return {"success": False, "error": stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def is_call_based(io_data: dict) -> bool:
    return bool(io_data.get("fn_name"))


def _unwrap_expected(expected):
    """APPS wraps call-based outputs in a single-element list. Unwrap it."""
    if isinstance(expected, list) and len(expected) == 1:
        return expected[0]
    return expected


def evaluate_code(code: str, io_data: dict) -> float:

    # ── HumanEval (assertion-based) ──
    if io_data.get("humaneval"):
        assertions = _extract_assertions(io_data["test_code"])
        if not assertions:
            return 0.0
        passed = sum(
            1 for a in assertions
            if _run_single_assertion(code, a, io_data["entry_point"])["success"]
        )
        return passed / len(assertions)

    # ── APPS ──
    inputs = io_data.get("inputs", [])

    if not inputs:
        return 0.0

    fn_name = io_data.get("fn_name")

    if fn_name:
        # call-based: inputs/outputs are lists of arguments/return values
        passed = 0
        for args, expected in zip(inputs, io_data.get("outputs", [])):
            result = _run_call_based(code, fn_name, args)
            if result["success"] and result["output"] == _unwrap_expected(expected):
                passed += 1
        return passed / len(inputs)

    # stdin-based
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

    # ── HumanEval (assertion-based) ──
    if io_data.get("humaneval"):
        assertions = _extract_assertions(io_data["test_code"])
        failures = []
        for assertion in assertions:
            if len(failures) >= max_failures:
                break
            result = _run_single_assertion(code, assertion, io_data["entry_point"])
            if not result["success"]:
                assertion_display = assertion.replace('candidate', io_data["entry_point"])
                failures.append({
                    "input":    assertion_display[:300],
                    "expected": "assertion pass",
                    "error":    result.get("error", "AssertionError")[:300],
                })
        return failures

    # ── APPS ──
    inputs  = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    fn_name = io_data.get("fn_name")
    failures = []

    if fn_name:
        # call-based
        for args, expected_raw in zip(inputs, outputs):
            if len(failures) >= max_failures:
                break
            expected = _unwrap_expected(expected_raw)
            result = _run_call_based(code, fn_name, args)
            if result["success"]:
                if result["output"] != expected:
                    failures.append({
                        "input": str(args)[:300],
                        "expected": str(expected)[:300],
                        "actual": str(result["output"])[:300],
                    })
            else:
        
                failures.append({
                    "input": str(args)[:300],
                    "expected": str(expected)[:300],
                    "error": result["error"][:300],
                })
        return failures

    # stdin-based
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
    n = len(accuracies)
    c = sum(1 for a in accuracies if a == 1.0)
    if n == 0 or k > n:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
