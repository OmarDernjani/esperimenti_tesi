import re
import sys
import os
import ast
import math
import subprocess
import tempfile
from datasets import load_dataset
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Global execution / model knobs (env-tunable) ────────────────────────────
NUM_CTX        = int(os.environ.get("NUM_CTX", "8192"))
EXEC_TIMEOUT   = int(os.environ.get("EXEC_TIMEOUT", "10"))


def load_apps_dataset():
    APPS = load_dataset("json", data_files={
        "train": "hf://datasets/codeparrot/apps/train.jsonl",
        "test":  "hf://datasets/codeparrot/apps/test.jsonl",
    })
    return APPS["train"], APPS["test"]


def load_humaneval_dataset():
    ds = load_dataset("openai/openai_humaneval")
    return ds["test"]


def get_humaneval_sample(data, n: int = 20, seed: int = 42, min_assertions: int = 6) -> list:
    import random as _random
    _random.seed(seed)
    indices = [
        i for i in range(len(data))
        if len(_extract_assertions(data[i]["test"])) >= min_assertions
    ]
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


def load_humaneval_plus_dataset():
    """HumanEval+ (evalplus) — 164 problemi con molti più test per funzione."""
    ds = load_dataset("evalplus/humanevalplus", split="test")
    return ds


def _extract_plus_tests(test_code: str) -> tuple[list, list]:
    """
    Parsa il `test` field di evalplus/humanevalplus e ritorna (inputs, results).
    Il formato atteso è una funzione `check(candidate)` con assegnazioni
    letterali `inputs = [...]` e `results = [...]` nel corpo.
    """
    import ast as _ast
    tree = _ast.parse(test_code)
    check_fn = next((n for n in tree.body
                     if isinstance(n, _ast.FunctionDef) and n.name == "check"), None)
    if check_fn is None:
        raise ValueError("check() function not found")
    inputs = results = None
    for stmt in check_fn.body:
        if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if isinstance(t, _ast.Name):
                if t.id == "inputs":
                    inputs = _ast.literal_eval(stmt.value)
                elif t.id == "results":
                    results = _ast.literal_eval(stmt.value)
    if inputs is None or results is None:
        raise ValueError("inputs/results not found in check()")
    return inputs, results


def get_humaneval_plus_sample(data, n: int = 20, seed: int = 42,
                              min_tests: int = 6) -> list:
    """
    Campiona n problemi HumanEval+ con almeno `min_tests` test case ciascuno,
    ritornando dict pronti per il flow call-based (chiavi: task_id, prompt,
    entry_point, inputs, results).
    """
    import random as _random
    _random.seed(seed)
    eligible = []
    for i in range(len(data)):
        try:
            inputs, results = _extract_plus_tests(data[i]["test"])
        except Exception:
            continue
        if len(inputs) >= min_tests and len(inputs) == len(results):
            eligible.append((i, inputs, results))
    chosen = _random.sample(eligible, min(n, len(eligible)))
    return [
        {
            "task_id":      data[i]["task_id"],
            "prompt":       data[i]["prompt"],
            "entry_point":  data[i]["entry_point"],
            "inputs":       inputs,
            "results":      results,
        }
        for (i, inputs, results) in chosen
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
            sample = {
                "question":     data["question"][i],
                "difficulty":   data["difficulty"][i],
                "input_output": data["input_output"][i],
            }
            if "solutions" in data.column_names:
                sample["solutions"] = data["solutions"][i]
            samples.append(sample)
    return samples


def load_augmented_dev(path: str, samples: list | None = None) -> dict:
    """
    Carica un file prodotto da augment_apps.py. Ritorna una mappa
    {int problem_idx: {"inputs": [...], "outputs": [...]}} limitata ai problemi
    con status == 'ok' e almeno un (input, output). Se il file non esiste o è
    corrotto ritorna {} (fallback trasparente allo split classico).

    Se `samples` è fornito (il minibatch che il runner sta per usare), controlla
    che il fingerprint salvato nell'augment corrisponda — altrimenti stampa un
    warning e ritorna {} (meglio fallback che dev disallineato).
    """
    import os as _os, json as _json, sys as _sys
    if not _os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return {}

    if samples is not None:
        meta = data.get("_meta", {}) if isinstance(data, dict) else {}
        saved_fp = meta.get("minibatch_fingerprint")
        if saved_fp:
            current_fp = []
            for i, s in enumerate(samples):
                try:
                    io = _json.loads(s["input_output"])
                    n_tc = len(io.get("inputs", []))
                except Exception:
                    n_tc = 0
                current_fp.append((i, s.get("difficulty"), n_tc,
                                   (s.get("question", "") or "")[:80]))
            saved_tuples = [(e.get("idx"), e.get("difficulty"),
                             e.get("n_test_cases"), e.get("question_head"))
                            for e in saved_fp]
            if saved_tuples != current_fp:
                print(f"[load_augmented_dev] WARNING: fingerprint mismatch su {path}. "
                      f"Il minibatch attuale differisce da quello usato in augment. "
                      f"Ignoro l'augmented dev e ricado sullo split classico.",
                      file=_sys.stderr)
                return {}

    problems = data.get("problems", {}) if isinstance(data, dict) else {}
    out: dict = {}
    for k, v in problems.items():
        if not isinstance(v, dict):
            continue
        if v.get("status") != "ok":
            continue
        ins, outs = v.get("inputs"), v.get("outputs")
        if not ins or not outs or len(ins) != len(outs):
            continue
        try:
            out[int(k)] = {"inputs": list(ins), "outputs": list(outs)}
        except (TypeError, ValueError):
            continue
    return out


def inject_augmented(io_data: dict, aug_entry: dict | None) -> dict:
    """Restituisce una copia di io_data con augmented_inputs/outputs iniettati
    se aug_entry è valido. Se None, restituisce io_data inalterato."""
    if not aug_entry:
        return io_data
    merged = dict(io_data)
    merged["augmented_inputs"]  = aug_entry["inputs"]
    merged["augmented_outputs"] = aug_entry["outputs"]
    return merged


def build_target_chain(model: str = "llama3.1:8b", call_based: bool = False, fn_name: str = ""):

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
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def build_optimizer_chain(model: str = "mistral-nemo"):
    """
    Optimizer chain only. Takes a raw coding problem and produces an enriched
    prompt that wraps the ORIGINAL problem with role + solution scaffolding.
    Pipeline rules (function name, I/O, markdown) are NOT injected here:
    they live in the target chain's system prompt.
    """
    model_target = os.environ.get("MODEL_TARGET", "")
    target_mention = (
        f"The target LLM you are optimizing for is `{model_target}`.\n"
        "Tailor your scaffolding to the strengths and weaknesses of this model.\n\n"
    ) if model_target else ""

    system_msg = (
        "You are an expert prompt engineer specializing in code generation.\n\n"
        f"{target_mention}"
        "You will receive a coding problem. Your task is NOT to rewrite or rephrase it.\n"
        "Your task is to produce an enriched prompt that wraps the ORIGINAL problem with\n"
        "scaffolding that helps the target LLM solve it correctly.\n\n"
        "The prompt you produce MUST:\n\n"
        "1. Assign the target LLM an expert role relevant to the problem\n"
        "   (e.g. \"expert competitive programmer\", \"Python algorithms specialist\").\n\n"
        "2. Include the ORIGINAL problem text VERBATIM — copy it exactly as given.\n"
        "   Do NOT paraphrase, summarize, shorten, or omit examples, constraints,\n"
        "   or input/output format.\n\n"
        "3. After the problem, add a \"## Solution guidance\" section containing:\n"
        "   - A short decomposition of the problem into 2-4 algorithmic sub-steps\n"
        "   - 1-2 non-trivial edge cases the model should explicitly handle\n"
        "   - The expected algorithmic approach and a target time complexity, if inferable\n"
        "   - Any data-structure hints that would simplify the implementation\n\n"
        "4. Instruct the target LLM to reason step-by-step BEFORE writing code,\n"
        "   then produce the final implementation.\n\n"
        "Do NOT add execution rules (function name, I/O handling, markdown wrapping).\n"
        "Those are appended automatically downstream.\n\n"
        "Return ONLY the final prompt for the target LLM. No preamble, no commentary."
    )

    template = ChatPromptTemplate([
        ("system", system_msg),
        ("human", "Problem:\n{problem}"),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def extract_code(response: str) -> str:
    """
    Extract a Python code block from an LLM response.

    Strategy:
      1. Look for fenced blocks ```python ... ``` (or ```py / plain ```), pick
         the longest match. Handles multiple fenced blocks in one response.
      2. If no closing fence, but the response opens with ```python / ```py /
         ```, take everything after the opening fence.
      3. Otherwise return the whole response stripped — let the executor fail
         on malformed output rather than guessing with fragile heuristics.
    """
    # ── 1. Properly fenced blocks ──
    for pattern in [r"```python\s*\n?(.*?)```", r"```py\s*\n?(.*?)```", r"```\s*\n?(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    # ── 2. Unclosed opening fence ──
    open_match = re.search(r"```(?:python|py)?\s*\n?(.*)$", response, re.DOTALL)
    if open_match:
        return open_match.group(1).strip()

    # ── 3. Plain text fallback ──
    return response.strip()


def build_variation_chain(model: str = "mistral-nemo"):
    """
    Iterative APE Monte Carlo step: takes a previous enriched prompt and produces
    a SEMANTIC VARIATION of its scaffolding. The original problem text must remain
    verbatim — only the role, decomposition, and hints vary.
    """
    model_target = os.environ.get("MODEL_TARGET", "")
    target_mention = (
        f"The target LLM you are optimizing for is `{model_target}`.\n"
        "Tailor your scaffolding to the strengths and weaknesses of this model.\n\n"
    ) if model_target else ""

    system_msg = (
        "You are an expert prompt engineer specializing in code generation.\n\n"
        f"{target_mention}"
        "You will receive a coding problem AND a previous enriched prompt that wraps\n"
        "the problem with a role and a '## Solution guidance' section. Your task is\n"
        "to produce a SEMANTIC VARIATION of that enriched prompt: same goal, different\n"
        "framing.\n\n"
        "The variation MUST:\n\n"
        "1. Include the ORIGINAL problem text VERBATIM — copy it exactly. Do NOT\n"
        "   paraphrase, summarize, or omit examples, constraints, or I/O format.\n\n"
        "2. Use a DIFFERENT expert role from the previous prompt.\n\n"
        "3. In the '## Solution guidance' section, use a DIFFERENT decomposition\n"
        "   strategy, DIFFERENT edge cases, and/or a DIFFERENT algorithmic angle\n"
        "   than the previous prompt.\n\n"
        "4. Keep the same overall structure: role + verbatim problem + solution\n"
        "   guidance + step-by-step instruction.\n\n"
        "Do NOT add execution rules (function name, I/O handling, markdown wrapping).\n"
        "Return ONLY the new enriched prompt. No preamble, no commentary."
    )
    template = ChatPromptTemplate([
        ("system", system_msg),
        ("human", "Problem:\n{problem}\n\nPrevious enriched prompt:\n{previous}"),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def population_size(io_data: dict) -> int:
    """Number of test cases / assertions in an io_data dict."""
    if io_data.get("humaneval"):
        if io_data.get("assertions") is not None:
            return len(io_data["assertions"])
        return len(_extract_assertions(io_data.get("test_code", "")))
    return len(io_data.get("inputs", []))


def split_io_data(io_data: dict, dev_frac: float = 0.3) -> tuple[dict, dict]:
    """
    Split a problem's evaluation data into a dev portion (used by APE/APO for
    candidate scoring) and a held-out test portion (used to report final accuracy).

    Returns (dev_io, test_io). Both retain all metadata (fn_name, humaneval flag,
    entry_point, ...). Only the test cases / assertions are partitioned.

    Edge cases:
    - If only 1 test case is available, dev == test (degenerate but non-breaking).
    - dev always has at least 1 item; test always has at least 1 item if possible.
    """
    # ── HumanEval (assertion-based) ──
    if io_data.get("humaneval"):
        import random as _r
        all_assertions = _extract_assertions(io_data["test_code"])
        _r.Random(42).shuffle(all_assertions)
        n = len(all_assertions)
        if n <= 1:
            dev_io  = {**io_data, "assertions": all_assertions}
            test_io = {**io_data, "assertions": all_assertions}
            return dev_io, test_io
        n_dev = max(1, min(n - 1, int(round(n * dev_frac))))
        dev_io  = {**io_data, "assertions": all_assertions[:n_dev]}
        test_io = {**io_data, "assertions": all_assertions[n_dev:]}
        return dev_io, test_io

    # ── APPS ──
    inputs  = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])

    # Augmented dev: if the caller injected augmented_inputs/outputs (produced
    # offline by augment_apps.py from the reference solution), use them as the
    # dev pool and keep ALL official inputs/outputs as the held-out test.
    aug_in  = io_data.get("augmented_inputs")
    aug_out = io_data.get("augmented_outputs")
    if aug_in and aug_out and len(aug_in) == len(aug_out) and len(inputs) >= 1:
        dev_io  = {**io_data, "inputs": list(aug_in),  "outputs": list(aug_out)}
        test_io = {**io_data, "inputs": list(inputs),  "outputs": list(outputs)}
        # Strip augmented fields from the inner dicts so evaluate_code sees clean IO.
        for k in ("augmented_inputs", "augmented_outputs"):
            dev_io.pop(k, None)
            test_io.pop(k, None)
        return dev_io, test_io

    n = len(inputs)
    if n <= 1:
        return dict(io_data), dict(io_data)
    n_dev = max(1, min(n - 1, int(round(n * dev_frac))))
    dev_io  = {**io_data, "inputs": inputs[:n_dev],  "outputs": outputs[:n_dev]}
    test_io = {**io_data, "inputs": inputs[n_dev:], "outputs": outputs[n_dev:]}
    return dev_io, test_io


def _run_script(file_path: str, test_input: str, timeout: int = EXEC_TIMEOUT) -> dict:
    try:
        proc = subprocess.Popen(
            [sys.executable, file_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        try:
            stdout, stderr = proc.communicate(input=test_input, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return {"success": False, "error": "Timeout"}
        if proc.returncode == 0:
            return {"success": True, "output": stdout}
        return {"success": False, "error": stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _normalize_value(x):
    """
    Recursively convert tuples → lists and sets → sorted lists. Used to make
    function return values comparable to JSON-loaded expected outputs (which
    never contain tuples or sets).
    """
    if isinstance(x, tuple):
        return [_normalize_value(e) for e in x]
    if isinstance(x, list):
        return [_normalize_value(e) for e in x]
    if isinstance(x, dict):
        return {k: _normalize_value(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted([_normalize_value(e) for e in x], key=repr)
    return x


def _run_call_based(code: str, fn_name: str, args, timeout: int = EXEC_TIMEOUT) -> dict:
    """
    Run `fn_name(*args)` against the user's code in a subprocess.

    Uses ast.literal_eval/repr instead of JSON to round-trip values, so tuples,
    sets, and other Python literals not supported by JSON pass through cleanly.
    Output is normalized (tuple→list, set→sorted list) so it can be compared
    against APPS expected values (which always come from json.loads).
    """
    # APPS occasionally stores a single arg unwrapped instead of as a list.
    if not isinstance(args, list):
        args = [args]

    harness = code + "\n\n"
    harness += "import sys as __sys, ast as __ast\n"
    harness += "__args = __ast.literal_eval(__sys.stdin.read())\n"
    harness += f"__result = {fn_name}(*__args)\n"
    harness += "print(repr(__result))\n"

    args_repr = repr(args)
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(harness)
        proc = subprocess.Popen(
            [sys.executable, path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        )
        try:
            stdout, stderr = proc.communicate(input=args_repr, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate()
            return {"success": False, "error": "Timeout"}
        if proc.returncode == 0:
            raw = stdout.strip()
            # The harness's repr(__result) is the LAST line; the user's code may
            # have printed debug output before it. Only parse the last line.
            last_line = raw.splitlines()[-1] if raw else ""
            try:
                value = ast.literal_eval(last_line)
                return {"success": True, "output": _normalize_value(value)}
            except (ValueError, SyntaxError):
                return {"success": True, "output": last_line}
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


def _run_single_assertion(code: str, assertion: str, entry_point: str, timeout: int = EXEC_TIMEOUT) -> dict:
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
            stdout, stderr = proc.communicate(timeout=timeout)
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
        assertions = io_data.get("assertions")
        if assertions is None:
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
        assertions = io_data.get("assertions")
        if assertions is None:
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
