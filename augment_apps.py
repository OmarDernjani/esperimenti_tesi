from __future__ import annotations
import os
import sys
import ast
import json
import time
import tempfile
import subprocess
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

import utils
from utils import _run_script, _run_call_based, _normalize, evaluate_code, EXEC_TIMEOUT

load_dotenv()


MODEL_AUGMENTER   = os.getenv("MODEL_AUGMENTER", os.getenv("MODEL_OPTIMIZER", "llama3.1:8b"))
N_PER_DIFFICULTY  = int(os.getenv("N_PER_DIFFICULTY", 5))
N_CANDIDATES      = int(os.getenv("N_CANDIDATES", 200))  
N_MIN_KEPT        = int(os.getenv("N_MIN_KEPT",    30))  
MAX_REFS_TO_TRY   = int(os.getenv("MAX_REFS_TO_TRY", 5))
OUTPUT_FILE       = os.getenv("AUG_OUTPUT", "augmented_dev.json")


AUG_TEMPERATURE = float(os.getenv("AUG_TEMPERATURE", 0.9))


def find_valid_reference(problem: dict, io_data: dict) -> tuple[int, str] | None:
    
    raw_sols = problem.get("solutions")
    if not raw_sols:
        return None
    try:
        sols = json.loads(raw_sols)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(sols, list) or not sols:
        return None

    for idx, code in enumerate(sols[:MAX_REFS_TO_TRY]):
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            score = evaluate_code(code, io_data)
        except Exception:
            continue
        if score >= 0.999:
            return idx, code
    return None



INPUT_DELIM = "<<<INPUT>>>"
ARGS_DELIM  = "<<<ARGS>>>"
END_DELIM   = "<<<END>>>"


GENERATION_ANGLES: list[tuple[str, str]] = [
    ("typical",
     "Typical, realistic inputs of medium size. Nominal usage — no edge cases, "
     "no extremes. What a normal user would pass."),
    ("boundary",
     "Boundary values: sizes and numeric values at the minimum and maximum allowed "
     "by the problem constraints, plus values at type edges (0, -1, 1, max int "
     "within bounds). One test case per distinct boundary when possible."),
    ("degenerate",
     "Degenerate cases: empty collections where allowed, single-element inputs, "
     "inputs where all elements are identical, length-1 containers, trivial cases "
     "that often break naive implementations."),
    ("stress",
     "Stress-size inputs: collections and numeric magnitudes NEAR (but not above) "
     "the maximum allowed by the constraints. Designed to exercise scalability."),
    ("ordering",
     "Ordering variations: inputs that are already sorted ascending, sorted "
     "descending / reverse, shuffled random, and nearly-sorted with just a few "
     "out-of-place elements."),
    ("duplicates",
     "Heavy duplication: inputs with many repeated values, entire collections of "
     "the same element, collections where each value appears exactly twice, and "
     "sparse-vs-dense duplication patterns."),
    ("adversarial",
     "Adversarial numerics (where the problem allows): negative values, zero-heavy "
     "inputs, values chosen to trigger off-by-one, overflow corners, or precision "
     "issues in naive solutions."),
    ("diverse_random",
     "Maximally diverse random inputs: a mix of sizes and value ranges that is NOT "
     "covered by the previous categories. Aim for variety over any particular "
     "structural property."),
]


_GENERATOR_SYSTEM_STDIN = (
    "You are a test-case designer for a competitive programming problem that reads from stdin.\n"
    "Your job is to produce NEW stdin inputs for a SPECIFIC category (the user tells you which).\n\n"
    "Rules:\n"
    "1. Follow the EXACT input format used in the problem statement and examples.\n"
    "2. Respect every constraint on ranges, sizes, types. Do not violate bounds.\n"
    "3. Stay STRICTLY within the category the user specifies. Do not mix categories.\n"
    "4. Output FORMAT — produce each input wrapped between markers exactly like this:\n"
    f"   {INPUT_DELIM}\n"
    "   <raw stdin content, possibly multi-line>\n"
    f"   {END_DELIM}\n"
    "5. Produce exactly N inputs (N is given by the user). No commentary, no numbering,\n"
    "   no explanations between or around the blocks."
)

_GENERATOR_SYSTEM_CALL = (
    "You are a test-case designer for a Python function-based problem.\n"
    "Your job is to produce NEW call arguments for a SPECIFIC category (the user tells you which).\n\n"
    "Rules:\n"
    "1. Each test case is a Python LIST of positional arguments to pass to the function,\n"
    "   in the same order and types as the examples. Example: if the function is\n"
    "   `solve(s, n)`, a test case is `[\"abc\", 3]`.\n"
    "2. Respect types and constraints stated in the problem. Do not invent types\n"
    "   not shown in the examples.\n"
    "3. Stay STRICTLY within the category the user specifies. Do not mix categories.\n"
    "4. Output FORMAT — each test case wrapped between markers exactly like this:\n"
    f"   {ARGS_DELIM}\n"
    "   [<python arg>, <python arg>, ...]\n"
    f"   {END_DELIM}\n"
    "   The content between the markers MUST be a single valid Python list literal\n"
    "   parseable by ast.literal_eval (strings quoted, no function calls, no imports).\n"
    "5. Produce exactly N test cases. No commentary, no numbering, no explanations\n"
    "   between or around the blocks."
)


def build_generator_chain(model: str, call_based: bool = False,
                          temperature: float = AUG_TEMPERATURE):
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    if call_based:
        system_msg = _GENERATOR_SYSTEM_CALL
        human_msg = (
            "Function name: `{fn_name}`\n\n"
            "Problem statement:\n{problem}\n\n"
            "Examples of valid official argument lists (each between markers):\n{examples}\n\n"
            "CATEGORY FOR THIS BATCH: **{angle_name}**\n"
            "{angle_hint}\n\n"
            "Now produce exactly {n} NEW argument lists in the above category, between "
            f"{ARGS_DELIM} / {END_DELIM} markers. Stay strictly within this category."
        )
    else:
        system_msg = _GENERATOR_SYSTEM_STDIN
        human_msg = (
            "Problem statement:\n{problem}\n\n"
            "Examples of valid official inputs (each between markers):\n{examples}\n\n"
            "CATEGORY FOR THIS BATCH: **{angle_name}**\n"
            "{angle_hint}\n\n"
            "Now produce exactly {n} NEW inputs in the above category, between "
            f"{INPUT_DELIM} / {END_DELIM} markers. Stay strictly within this category."
        )
    template = ChatPromptTemplate([("system", system_msg), ("human", human_msg)])
    llm = ChatOllama(
        model=model,
        num_ctx=int(os.environ.get("NUM_CTX", "8192")),
        num_keep=0,
        temperature=temperature,
    )
    return template | llm | StrOutputParser()


def _format_examples_stdin(inputs: list[str], k: int = 3) -> str:
    lines = []
    for inp in inputs[:k]:
        lines.append(INPUT_DELIM)
        lines.append(inp.rstrip("\n"))
        lines.append(END_DELIM)
    return "\n".join(lines) if lines else "(no examples available)"


def _format_examples_args(arg_lists: list, k: int = 3) -> str:
    lines = []
    for args in arg_lists[:k]:
        if not isinstance(args, list):
            args = [args]
        lines.append(ARGS_DELIM)
        lines.append(repr(args))
        lines.append(END_DELIM)
    return "\n".join(lines) if lines else "(no examples available)"


def _parse_blocks(text: str, delim: str) -> list[str]:
    out = []
    cursor = 0
    while True:
        start = text.find(delim, cursor)
        if start < 0:
            break
        end = text.find(END_DELIM, start + len(delim))
        if end < 0:
            break
        body = text[start + len(delim):end]
        body = body.strip("\r\n")
        if body.strip():
            out.append(body)
        cursor = end + len(END_DELIM)
    return out


def parse_generated(text: str) -> list[str]:
    """Stdin inputs: blocchi grezzi (stringa) tra <<<INPUT>>> e <<<END>>>."""
    raw = _parse_blocks(text, INPUT_DELIM)
    return [(b if b.endswith("\n") else b + "\n") for b in raw]


def parse_generated_args(text: str) -> list[list]:
    """Call-based args: blocchi Python literal tra <<<ARGS>>> e <<<END>>>."""
    out: list[list] = []
    for body in _parse_blocks(text, ARGS_DELIM):
        try:
            val = ast.literal_eval(body)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(val, list):
            val = [val]
        out.append(val)
    return out



def run_reference_on_input(ref_code: str, stdin_input: str) -> str | None:
    
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(ref_code)
        result = _run_script(path, stdin_input, timeout=EXEC_TIMEOUT)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if not result.get("success"):
        return None
    out = result.get("output", "")
    if out is None:
        return None
    norm = _normalize(out)
    if not norm:
        return None
    return norm


def run_reference_on_args(ref_code: str, fn_name: str, args: list):
    
    result = _run_call_based(ref_code, fn_name, args, timeout=EXEC_TIMEOUT)
    if not result.get("success"):
        return None
    return result.get("output")


def _invoke_with_retry(chain, payload: dict, max_attempts: int = 3,
                       base_delay: float = 2.0) -> str:
    
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return chain.invoke(payload)
        except Exception as e:
            name = type(e).__name__
            if "Connection" not in name and "Timeout" not in name:
                raise
            last_err = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                tqdm.write(f"    [retry] {name} — sleep {delay:.0f}s poi riprovo")
                time.sleep(delay)
    raise last_err if last_err else RuntimeError("retry exhausted")


def preflight_ollama(model: str) -> None:
    """
    Verifica prima del loop che Ollama risponda e che il modello risponda.
    Abort con messaggio chiaro se fallisce: meglio fermarsi subito che girare
    30 problemi tutti in errore di connessione.
    """
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model=model, num_ctx=1024, num_keep=0)
    try:
        out = llm.invoke("Say 'ok'.")
    except Exception as e:
        print(f"[preflight] Ollama non raggiungibile o modello '{model}' non caricabile: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        print("[preflight] Verifica: 'ollama serve' attivo, 'ollama list' mostra il modello.",
              file=sys.stderr)
        sys.exit(3)
    content = getattr(out, "content", str(out))
    print(f"[preflight] Ollama OK. Modello '{model}' risponde: {content.strip()[:60]!r}")


ù

def _dedupe_key(cand, call_based: bool) -> str:
    """
    Chiave di dedupe stabile per input generati.
    - stdin: la stringa stessa (già normalizzata in parse)
    - call-based: repr della lista argomenti (ordine e tipi preservati)
    """
    if call_based:
        return repr(cand)
    ù
    return "\n".join(l.rstrip() for l in cand.splitlines()).strip()



def augment_one(problem: dict, stdin_chain, call_chain,
                total_target: int = N_CANDIDATES) -> dict | None:
    """
    Genera fino a ~total_target test case per il problema, spalmati sui round
    definiti in GENERATION_ANGLES. Ogni round:
      1. chiama l'LLM con la categoria specifica
      2. parsa i blocchi generati
      3. dedupe progressivo (cross-round)
      4. valida eseguendo la reference → scarta i candidati invalidi/crash
    Ritorna dict con status/inputs/outputs/per-round stats.
    """
    try:
        io_data = json.loads(problem["input_output"])
    except (json.JSONDecodeError, TypeError):
        return {"status": "skipped", "reason": "bad_input_output"}

    if not io_data.get("inputs"):
        return {"status": "skipped", "reason": "no_official_tests"}

    ref = find_valid_reference(problem, io_data)
    if ref is None:
        return {"status": "skipped", "reason": "no_valid_reference"}
    ref_idx, ref_code = ref

    fn_name = io_data.get("fn_name") or ""
    call_based = bool(fn_name)
    chain = call_chain if call_based else stdin_chain

    n_angles = len(GENERATION_ANGLES)
    per_round = max(10, (total_target + n_angles - 1) // n_angles)  

    
    if call_based:
        base_payload = {
            "fn_name":  fn_name,
            "problem":  problem["question"],
            "examples": _format_examples_args(io_data["inputs"], k=3),
        }
    else:
        base_payload = {
            "problem":  problem["question"],
            "examples": _format_examples_stdin(io_data["inputs"], k=3),
        }

    aug_inputs:  list = []
    aug_outputs: list = []
    seen: set[str] = set()
    round_stats: list[dict] = []
    total_raw = 0
    total_errors = 0

    for angle_name, angle_hint in GENERATION_ANGLES:
        payload = {
            **base_payload,
            "angle_name": angle_name,
            "angle_hint": angle_hint,
            "n": per_round,
        }

        t0_round = time.time()
        try:
            raw = _invoke_with_retry(chain, payload)
        except Exception as e:
            round_stats.append({
                "angle": angle_name, "raw": 0, "kept": 0,
                "dups": 0, "errors": 0,
                "status": f"generator_error:{type(e).__name__}",
                "elapsed_sec": round(time.time() - t0_round, 2),
            })
            continue

        if call_based:
            candidates = parse_generated_args(raw)
        else:
            candidates = parse_generated(raw)
        total_raw += len(candidates)

        n_kept_round = 0
        n_dups_round = 0
        n_err_round  = 0
        for cand in candidates:
            key = _dedupe_key(cand, call_based)
            if key in seen:
                n_dups_round += 1
                continue
            seen.add(key)

            if call_based:
                out = run_reference_on_args(ref_code, fn_name, cand)
                if out is None:
                    n_err_round += 1
                    continue
                aug_inputs.append(cand)
                aug_outputs.append([out])
            else:
                out = run_reference_on_input(ref_code, cand)
                if out is None:
                    n_err_round += 1
                    continue
                aug_inputs.append(cand)
                aug_outputs.append(out)
            n_kept_round += 1

            
            if len(aug_inputs) >= total_target:
                break

        total_errors += n_err_round
        round_stats.append({
            "angle":       angle_name,
            "raw":         len(candidates),
            "kept":        n_kept_round,
            "dups":        n_dups_round,
            "errors":      n_err_round,
            "elapsed_sec": round(time.time() - t0_round, 2),
        })

        if len(aug_inputs) >= total_target:
            break

    return {
        "status":           "ok" if len(aug_inputs) >= N_MIN_KEPT else "too_few",
        "reference_index":  ref_idx,
        "call_based":       call_based,
        "fn_name":          fn_name,
        "inputs":           aug_inputs,
        "outputs":          aug_outputs,
        "n_candidates_raw": total_raw,
        "n_kept":           len(aug_inputs),
        "n_errors":         total_errors,
        "rounds":           round_stats,
    }



def main():
    print(f"[augment_apps] MODEL_AUGMENTER={MODEL_AUGMENTER}  "
          f"N_PER_DIFFICULTY={N_PER_DIFFICULTY}  N_CANDIDATES(total)={N_CANDIDATES}  "
          f"rounds={len(GENERATION_ANGLES)}  temp={AUG_TEMPERATURE}")

    print("[augment_apps] Loading APPS dataset...")
    train_data, _ = utils.load_apps_dataset()
    samples = utils.get_minibatch(train_data, n_per_difficulty=N_PER_DIFFICULTY)
    print(f"[augment_apps] Minibatch: {len(samples)} problemi")

    if not any("solutions" in s for s in samples):
        print("[augment_apps] ERRORE: il minibatch non contiene 'solutions'. "
              "Verifica che utils.get_minibatch sia la versione aggiornata e "
              "che il dataset esponga la colonna 'solutions'.", file=sys.stderr)
        sys.exit(2)

    preflight_ollama(MODEL_AUGMENTER)

    stdin_chain = build_generator_chain(MODEL_AUGMENTER, call_based=False)
    call_chain  = build_generator_chain(MODEL_AUGMENTER, call_based=True)


    fingerprint = []
    for i, s in enumerate(samples):
        try:
            io = json.loads(s["input_output"])
            n_tc = len(io.get("inputs", []))
        except Exception:
            n_tc = 0
        fingerprint.append({
            "idx":          i,
            "difficulty":   s.get("difficulty"),
            "n_test_cases": n_tc,
            "question_head": (s.get("question", "") or "")[:80],
        })

    output: dict = {
        "_meta": {
            "created_at":        datetime.now().isoformat(timespec="seconds"),
            "model_augmenter":   MODEL_AUGMENTER,
            "n_per_difficulty":  N_PER_DIFFICULTY,
            "n_candidates":      N_CANDIDATES,
            "n_min_kept":        N_MIN_KEPT,
            "temperature":       AUG_TEMPERATURE,
            "rounds":            [a[0] for a in GENERATION_ANGLES],
            "minibatch_fingerprint": fingerprint,
        },
        "problems": {},
    }
    stats = {"ok": 0, "too_few": 0, "skipped": 0}

    for idx, problem in enumerate(tqdm(samples, desc="Augmenting")):
        t0 = time.time()
        res = augment_one(problem, stdin_chain, call_chain)
        dt = time.time() - t0
        if res is None:
            stats["skipped"] += 1
            continue

        status = res.get("status", "skipped")
        stats[status] = stats.get(status, 0) + 1

        output["problems"][str(idx)] = {
            "difficulty":      problem.get("difficulty"),
            "status":          status,
            "reason":          res.get("reason"),
            "reference_index": res.get("reference_index"),
            "n_candidates":    res.get("n_candidates_raw", 0),
            "n_kept":          res.get("n_kept", 0),
            "n_errors":        res.get("n_errors", 0),
            "rounds":          res.get("rounds", []),
            "inputs":          res.get("inputs", []),
            "outputs":         res.get("outputs", []),
            "elapsed_sec":     round(dt, 2),
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        tqdm.write(f"  [{idx}] {status} — kept={res.get('n_kept', 0)}/"
                   f"{res.get('n_candidates_raw', 0)}  ref#{res.get('reference_index')}  "
                   f"({dt:.1f}s)")

    print(f"\n[augment_apps] Done. stats={stats}  →  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
