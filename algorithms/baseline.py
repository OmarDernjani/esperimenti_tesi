from utils import extract_code, evaluate_code


def run_baseline(
    question: str,
    test_io: dict,
    target_chain,
    optimizer_chain,
) -> dict:
    """
    Single-shot optimizer baseline:
      problem  →  optimizer (enriched prompt)  →  target (code).
    No iterations, no selection. Evaluated on the held-out TEST split for
    consistency with APE/APO reporting.
    """
    print("\n[Baseline] Generazione enriched prompt …")
    enriched_prompt = optimizer_chain.invoke({"problem": question})

    print("[Baseline] Generazione codice …")
    code = extract_code(target_chain.invoke({"user_prompt": enriched_prompt}))
    test_score = evaluate_code(code, test_io)

    print(f"[Baseline] test_score: {test_score:.3f}")
    return {"enriched_prompt": enriched_prompt, "code": code, "test_score": test_score}
