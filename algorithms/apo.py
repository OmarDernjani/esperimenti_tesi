from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import evaluate_code, get_failing_tests, extract_code

BEAM_WIDTH     = 2
MAX_ITERS      = 4
MAX_NO_IMPROVE = 2


def _critique_chain(model: str):
    template = ChatPromptTemplate([
        ("system",
         "You are an expert in competitive programming and prompt engineering. "
         "A language model was given a problem description and generated Python code "
         "that fails on some test cases. Identify weaknesses in the PROBLEM DESCRIPTION "
         "that could have caused the LLM to misunderstand the problem. "
         "Focus on ambiguities, missing constraints, or unclear input/output format — not on the code."),
        ("human",
         "Problem description:\n{prompt}\n\n"
         "Generated code:\n{code}\n\n"
         "Failing test cases:\n{failures}\n\n"
         "List the weaknesses in the problem description that likely led to these failures."),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def _edit_chain(model: str):
    template = ChatPromptTemplate([
        ("system",
         "You are a prompt engineer specializing in competitive programming. "
         "Rewrite the problem description to fix the weaknesses identified in the critique. "
         "Preserve all original constraints, examples, and input/output format. "
         "Return only the rewritten problem description, no code, no explanations."),
        ("human",
         "Original problem description:\n{prompt}\n\n"
         "Failing test cases:\n{failures}\n\n"
         "Critique:\n{critique}\n\n"
         "Rewritten problem description:"),
    ])
    llm = ChatOllama(model=model, num_ctx=4096, num_keep=0)
    return template | llm | StrOutputParser()


def _format_failures(failures: list[dict]) -> str:
    if not failures:
        return "None — the code passes all available test cases."
    lines = []
    for i, f in enumerate(failures, 1):
        if "error" in f:
            lines.append(f"Test {i}: input={f['input']!r} → runtime error: {f['error']!r}")
        else:
            lines.append(f"Test {i}: input={f['input']!r} → expected={f['expected']!r}, got={f['actual']!r}")
    return "\n".join(lines)


def _expand(item: dict, io_data: dict, solver_chain, num_gradients: int, critique_chain, edit_chain) -> list[dict]:
    """Per ogni gradiente: critica il prompt sui test falliti → edita → genera codice."""
    failure_str = _format_failures(get_failing_tests(item["code"], io_data))
    candidates = []
    for _ in range(num_gradients):
        critique        = critique_chain.invoke({"prompt": item["prompt"], "code": item["code"], "failures": failure_str})
        improved_prompt = edit_chain.invoke({"prompt": item["prompt"], "failures": failure_str, "critique": critique})
        code            = extract_code(solver_chain.invoke({"user_prompt": improved_prompt}))
        candidates.append({"prompt": improved_prompt, "code": code, "accuracy": evaluate_code(code, io_data), "critique": critique})
    return candidates


def run_apo(
    question: str,
    baseline_code: str,
    io_data: dict,
    solver_chain,
    model_target: str,
    model_optimizer: str,
    num_gradients: int = 2,
    beam_width: int = BEAM_WIDTH,
    max_iters: int = MAX_ITERS,
    max_no_improve: int = MAX_NO_IMPROVE,
) -> dict:
    """
    Beam search di larghezza beam_width per max_iters iterazioni (o early stopping).
    Ogni step espande ogni membro del beam con num_gradients candidati diretti,
    poi seleziona i top-b dalla pool combinata.
    """
    print(f"\n[APO] Avvio — beam_width={beam_width}, max_iters={max_iters}")

    crit_chain = _critique_chain(model_optimizer)
    edit_chain = _edit_chain(model_target)

    beam        = [{"prompt": question, "code": baseline_code, "accuracy": evaluate_code(baseline_code, io_data)}]
    best_so_far = beam[0]["accuracy"]
    no_improve  = 0
    iterations  = []

    for iter_num in range(1, max_iters + 1):
        print(f"  [APO] Iter {iter_num}/{max_iters} — espansione beam ({len(beam)} membri) …")

        pool = []
        for item in beam:
            pool.extend(_expand(item, io_data, solver_chain, num_gradients, crit_chain, edit_chain))

        pool.sort(key=lambda x: x["accuracy"], reverse=True)
        beam          = pool[:beam_width]
        best_accuracy = beam[0]["accuracy"]

        print(f"  [APO] Iter {iter_num}: best={best_accuracy:.3f}, candidati={len(pool)}")

        iterations.append({
            "iter":          iter_num,
            "best_accuracy": best_accuracy,
            "beam":          [{"prompt": c["prompt"], "code": c["code"], "accuracy": c["accuracy"]} for c in beam],
            "all_candidates": [{"prompt": c["prompt"], "accuracy": c["accuracy"]} for c in pool],
        })

        if best_accuracy >= 1.0:
            print(f"  [APO] Accuracy 1.0 raggiunta — stop anticipato.")
            break

        if best_accuracy > best_so_far:
            best_so_far = best_accuracy
            no_improve  = 0
            print(f"  [APO] Miglioramento — reset contatore.")
        else:
            no_improve += 1
            print(f"  [APO] Nessun miglioramento ({no_improve}/{max_no_improve}).")
            if no_improve >= max_no_improve:
                print(f"  [APO] Convergenza dopo {iter_num} iterazioni.")
                break

    print(f"[APO] Fine. Best: {beam[0]['accuracy']:.3f}")
    return {"best_accuracy": beam[0]["accuracy"], "iterations": iterations}
