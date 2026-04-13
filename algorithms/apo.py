"""
APO / ProTeGi (Pryzant et al. 2023, "Automatic Prompt Optimization with
'Gradient Descent' and Beam Search"), per-problem adaptation.

Algorithm 1 of the paper, scaled down for local execution and adapted to the
single-problem setting (each APPS / HumanEval task is its own optimization
instance, with its dev_io split as the training set).

Differences from the paper:
- Minibatch sampling collapses to "all of dev_io" (the dev split is already
  small, ~3 cases per problem).
- Successor selection is exhaustive evaluation on dev (no UCB / successive
  halving). At this pool size (~10 candidates) bandit methods give no benefit.
- The initial beam contains a single prompt p0 = raw question, with code from
  the zero-shot run (passed in as `baseline_code`).
- Strict no-elitism, as in the paper: the next beam is drawn only from the
  current iteration's pool; previous beam members are not carried over.
"""
import os
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import evaluate_code, get_failing_tests, extract_code, population_size, NUM_CTX
from typing import Optional

# Defaults inspired by Pryzant et al. 2023, scaled down for local Ollama.
NUM_GRADIENTS   = 2   # m: distinct critiques per beam member
NUM_EDITS       = 1   # q: improved prompts per critique
NUM_PARAPHRASES = 1   # r: paraphrases per gradient candidate
BEAM_WIDTH      = 2
MAX_ITERS       = 4
MAX_NO_IMPROVE  = 2


# ── Optimizer chains ─────────────────────────────────────────────────────────

def _gradient_chain(model: str):
    """ProTeGi step 1 — textual gradients (critiques) of a failing prompt."""
    model_target = os.environ.get("MODEL_TARGET", "")
    target_mention = (
        f"The target LLM is `{model_target}`. "
    ) if model_target else ""

    template = ChatPromptTemplate([
        ("system",
         "You are an expert in competitive programming and prompt engineering. "
         f"{target_mention}"
         "I am writing a prompt that asks a target language model to solve a coding "
         "problem. My current prompt led the model to produce code that fails on some "
         "test cases. Your job is to identify weaknesses in the PROMPT (not the code) "
         "that could have caused the failures.\n\n"
         "Generate EXACTLY {m} DISTINCT critiques. Each critique should focus on a "
         "DIFFERENT weakness, e.g.: ambiguity in problem statement, missing edge case, "
         "unclear input/output format, missing complexity hint, missing decomposition "
         "guidance, missing data-structure suggestion.\n\n"
         "Output format (mandatory, one critique per <critique> block):\n"
         "<critique>first weakness explained in 1-3 sentences</critique>\n"
         "<critique>second weakness explained in 1-3 sentences</critique>\n"
         "...\n"
         "Output NOTHING outside the <critique> blocks."),
        ("human",
         "Current prompt:\n{prompt}\n\n"
         "Generated code:\n{code}\n\n"
         "Failing test cases:\n{failures}\n\n"
         "Generate {m} distinct critiques."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _edit_chain(model: str):
    """ProTeGi step 2 — apply a textual gradient: write q improved prompts."""
    model_target = os.environ.get("MODEL_TARGET", "")
    target_mention = (
        f"The target LLM is `{model_target}`. "
    ) if model_target else ""

    template = ChatPromptTemplate([
        ("system",
         "You are an expert prompt engineer for code generation. "
         f"{target_mention}"
         "Given a current prompt "
         "that led the target LLM to produce failing code, the failing test cases, and a "
         "critique that identifies a weakness in the prompt, write {q} DIFFERENT improved "
         "prompts that address the critique.\n\n"
         "Each improved prompt MUST:\n"
         "- Include the ORIGINAL problem statement VERBATIM (do not omit examples,\n"
         "  constraints, or input/output format)\n"
         "- Specifically address the critique with new scaffolding (role, decomposition,\n"
         "  edge cases, complexity, data-structure hints, ...)\n"
         "- Be self-contained and ready to send to the target model\n\n"
         "Do NOT add execution rules (function name, I/O handling, markdown wrapping).\n\n"
         "Output format (mandatory, one prompt per <prompt> block):\n"
         "<prompt>first improved prompt</prompt>\n"
         "<prompt>second improved prompt</prompt>\n"
         "...\n"
         "Output NOTHING outside the <prompt> blocks."),
        ("human",
         "Current prompt:\n{prompt}\n\n"
         "Failing test cases:\n{failures}\n\n"
         "Critique:\n{critique}\n\n"
         "Write {q} improved prompts that address this critique."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _paraphrase_chain(model: str):
    """ProTeGi step 3 — Monte Carlo expansion via semantic paraphrase."""
    model_target = os.environ.get("MODEL_TARGET", "")
    target_mention = (
        f"The target LLM is `{model_target}`. "
    ) if model_target else ""

    template = ChatPromptTemplate([
        ("system",
         f"You are an expert prompt engineer. {target_mention}"
         "Generate a SEMANTIC PARAPHRASE of a given "
         "prompt while preserving its instructional intent. The paraphrase MUST:\n"
         "- Include the original problem statement VERBATIM (do NOT paraphrase the\n"
         "  problem itself — only the surrounding instructions)\n"
         "- Restructure or rephrase the role, hints, scaffolding, and decomposition\n"
         "- Be meaningfully different from the input but with the same goal\n\n"
         "Do NOT add execution rules. Return ONLY the paraphrased prompt, no preamble."),
        ("human", "Original prompt:\n{prompt}\n\nGenerate a paraphrase."),
    ])
    llm = ChatOllama(model=model, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _extract_tagged(text: str, tag: str, min_len: int = 10) -> list[str]:
    """
    Extract <tag>...</tag> blocks from an LLM response.
    Falls back to numbered-list parsing, then to whole-text, if tags are missing.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = [m.strip() for m in re.findall(pattern, text, re.DOTALL) if m.strip() and len(m.strip()) >= min_len]
    if matches:
        return matches
    # Fallback 1: numbered list "1." "2." ...
    items = re.split(r'\n\s*\d+[\.\)]\s+', text)
    items = [it.strip() for it in items if len(it.strip()) >= min_len]
    if len(items) > 1:
        return items
    # Fallback 2: whole response
    return [text.strip()] if len(text.strip()) >= min_len else []


# ── Main loop ────────────────────────────────────────────────────────────────

def run_apo(
    question: str,
    baseline_code: str,
    dev_io: dict,
    test_io: dict,
    target_chain,
    model_optimizer: str,
    optimizer_chain=None,
    num_gradients: int = NUM_GRADIENTS,
    num_edits: int = NUM_EDITS,
    num_paraphrases: int = NUM_PARAPHRASES,
    beam_width: int = BEAM_WIDTH,
    max_iters: int = MAX_ITERS,
    max_no_improve: int = MAX_NO_IMPROVE,
) -> dict:
    """
    APO / ProTeGi, per-problem version.

    Beam search of width `beam_width` over prompts. Each iteration:

      1. For every beam member with errors on dev:
         a. Generate `num_gradients` distinct textual gradients (critiques)
            in a single LLM call.
         b. For each gradient, generate `num_edits` improved prompts that
            address it (apply-gradient step).
      2. Monte Carlo expansion: for every gradient candidate, generate
         `num_paraphrases` semantic paraphrases.
      3. Evaluate the entire pool (gradient candidates ∪ paraphrases) on dev.
      4. Select top-`beam_width` as the next beam (no elitism — paper-faithful).
      5. Early stop on dev=1.0 or no improvement for `max_no_improve` iters.

    Final reporting: every member of the LAST iteration's pool is also
    evaluated on the held-out TEST split for downstream pass@k.
    """
    n_dev  = population_size(dev_io)
    n_test = population_size(test_io)
    print(f"\n[APO] Avvio — beam={beam_width}, m={num_gradients}, q={num_edits}, "
          f"r={num_paraphrases}, max_iters={max_iters}, dev={n_dev}, test={n_test}")

    grad_chain = _gradient_chain(model_optimizer)
    edit_chain = _edit_chain(model_optimizer)
    para_chain = _paraphrase_chain(model_optimizer)

    # ── Initial beam: p0 = enriched prompt via optimizer_chain (fallback: raw question) ──
    # Rationale: starting from the raw question causes APO to exit with a
    # "best_prompt" that is just a restatement of the problem whenever the
    # baseline already passes the dev split. We initialize with the same
    # enriched scaffolding APE uses, so the two branches are comparable and
    # the saved best_prompt always reflects an actual optimizer output.
    if optimizer_chain is not None:
        p0 = optimizer_chain.invoke({"problem": question})
        p0_code = extract_code(target_chain.invoke({"user_prompt": p0}))
    else:
        p0 = question
        p0_code = baseline_code
    initial_dev = evaluate_code(p0_code, dev_io)
    beam: list[dict] = [{
        "prompt":    p0,
        "code":      p0_code,
        "dev_score": initial_dev,
        "source":    "init",
    }]
    best_so_far = initial_dev
    no_improve  = 0
    iterations: list[dict] = []
    last_pool: list[dict] = list(beam)

    if initial_dev >= 1.0:
        print("[APO] Baseline già perfetto sul dev — nessuna iterazione necessaria.")

    for t in range(1, max_iters + 1):
        if all(m["dev_score"] >= 1.0 for m in beam):
            print(f"  [APO] Tutti i beam member sono al 1.0 — stop.")
            break

        print(f"  [APO] Iter {t}/{max_iters} — espansione beam ({len(beam)} membri) …")

        gradient_candidates: list[dict] = []
        for member in beam:
            if member["dev_score"] >= 1.0:
                # No errors on dev → no gradient. Paper just doesn't expand this member.
                continue

            failures    = get_failing_tests(member["code"], dev_io)
            failure_str = _format_failures(failures)

            # Step 1: m gradients (single LLM call)
            grad_resp = grad_chain.invoke({
                "prompt":   member["prompt"],
                "code":     member["code"],
                "failures": failure_str,
                "m":        num_gradients,
            })
            gradients = _extract_tagged(grad_resp, "critique")[:num_gradients]
            if not gradients:
                continue

            # Step 2: per ogni gradient, q edit (apply-gradient)
            for grad in gradients:
                edit_resp = edit_chain.invoke({
                    "prompt":   member["prompt"],
                    "failures": failure_str,
                    "critique": grad,
                    "q":        num_edits,
                })
                improved = _extract_tagged(edit_resp, "prompt")[:num_edits]
                for imp in improved:
                    code      = extract_code(target_chain.invoke({"user_prompt": imp}))
                    dev_score = evaluate_code(code, dev_io)
                    gradient_candidates.append({
                        "prompt":    imp,
                        "code":      code,
                        "dev_score": dev_score,
                        "source":    "gradient",
                        "critique":  grad,
                    })

        # Step 3: Monte Carlo paraphrase expansion
        paraphrase_candidates: list[dict] = []
        for cand in gradient_candidates:
            for _ in range(num_paraphrases):
                para      = para_chain.invoke({"prompt": cand["prompt"]})
                code      = extract_code(target_chain.invoke({"user_prompt": para}))
                dev_score = evaluate_code(code, dev_io)
                paraphrase_candidates.append({
                    "prompt":    para,
                    "code":      code,
                    "dev_score": dev_score,
                    "source":    "paraphrase",
                })

        full_pool = gradient_candidates + paraphrase_candidates
        if not full_pool:
            print("  [APO] Nessun candidato generato — stop.")
            break

        # Step 4: successor selection (top-b on dev, no elitism)
        full_pool.sort(key=lambda c: c["dev_score"], reverse=True)
        beam      = full_pool[:beam_width]
        last_pool = full_pool
        best_dev  = beam[0]["dev_score"]

        n_grad = sum(1 for c in full_pool if c["source"] == "gradient")
        n_para = sum(1 for c in full_pool if c["source"] == "paraphrase")
        print(f"  [APO] Iter {t}: best_dev={best_dev:.3f}  pool={len(full_pool)} (grad={n_grad}, para={n_para})")

        iterations.append({
            "iter":           t,
            "best_dev":       best_dev,
            "pool_size":      len(full_pool),
            "n_gradient":     n_grad,
            "n_paraphrase":   n_para,
            "beam":           [
                {"prompt": c["prompt"], "code": c["code"], "dev_score": c["dev_score"], "source": c["source"]}
                for c in beam
            ],
            "all_candidates": [
                {"prompt": c["prompt"], "dev_score": c["dev_score"], "source": c["source"]}
                for c in full_pool
            ],
        })

        if best_dev >= 1.0:
            print("  [APO] Dev score 1.0 raggiunto — stop anticipato.")
            break

        if best_dev > best_so_far:
            best_so_far = best_dev
            no_improve  = 0
            print("  [APO] Miglioramento — reset contatore.")
        else:
            no_improve += 1
            print(f"  [APO] Nessun miglioramento ({no_improve}/{max_no_improve}).")
            if no_improve >= max_no_improve:
                print(f"  [APO] Convergenza dopo {t} iterazioni.")
                break

    # ── Final test eval su tutto last_pool ──
    print(f"  [APO] Valutazione pool finale ({len(last_pool)}) sul test split …")
    for c in last_pool:
        c["test_score"] = evaluate_code(c["code"], test_io)

    best = max(last_pool, key=lambda c: c["dev_score"])
    print(f"[APO] Best: dev={best['dev_score']:.3f}  test={best['test_score']:.3f}")

    return {
        "best_prompt": best["prompt"],
        "best_code":   best["code"],
        "best_dev":    best["dev_score"],
        "best_test":   best["test_score"],
        "n_dev":       n_dev,
        "n_test":      n_test,
        "final_pool":  [
            {
                "prompt":     c["prompt"],
                "code":       c["code"],
                "dev_score":  c["dev_score"],
                "test_score": c["test_score"],
                "source":     c.get("source", "init"),
            }
            for c in last_pool
        ],
        "iterations":  iterations,
    }
