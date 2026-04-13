from utils import (
    build_variation_chain,
    evaluate_code,
    extract_code,
    population_size,
)

DEFAULT_N_PROPOSALS = 5
DEFAULT_N_ITERS     = 2
DEFAULT_N_KEEP      = 2


def run_ape(
    question: str,
    dev_io: dict,
    test_io: dict,
    target_chain,
    optimizer_chain,
    model_optimizer: str,
    n_proposals: int = DEFAULT_N_PROPOSALS,
    n_iters: int = DEFAULT_N_ITERS,
    n_keep: int = DEFAULT_N_KEEP,
) -> dict:
    """
    APE (Zhou et al. 2022), per-problem adaptation.

    1. Proposal: generate `n_proposals` enriched prompts via the optimizer.
    2. Score each on the DEV split (passed in by the caller).
    3. Iterative Monte Carlo: keep top-`n_keep`, generate semantic variations
       of each via the variation chain. Re-score on dev. Repeat for `n_iters`.
    4. Final eval: every candidate in the final population is also evaluated
       on the held-out TEST split for downstream pass@k computation.
    """
    n_dev  = population_size(dev_io)
    n_test = population_size(test_io)
    print(f"\n[APE] Avvio — n_proposals={n_proposals}, n_iters={n_iters}, n_keep={n_keep}, dev={n_dev}, test={n_test}")

    variation_chain = build_variation_chain(model_optimizer)

    iterations = []

    # ── Round 0: initial proposals ──
    print(f"  [APE] Round 0: {n_proposals} candidati iniziali …")
    candidates = []
    for _ in range(n_proposals):
        enriched = optimizer_chain.invoke({"problem": question})
        code     = extract_code(target_chain.invoke({"user_prompt": enriched}))
        dev_acc  = evaluate_code(code, dev_io)
        candidates.append({"prompt": enriched, "code": code, "dev_score": dev_acc})

    iterations.append({
        "round":      0,
        "candidates": [
            {"prompt": c["prompt"], "code": c["code"], "dev_score": c["dev_score"]}
            for c in candidates
        ],
        "best_dev":   max(c["dev_score"] for c in candidates),
    })
    print(f"  [APE] Round 0: best_dev={iterations[-1]['best_dev']:.3f}")

    # ── Iterative Monte Carlo refinement ──
    for it in range(1, n_iters + 1):
        candidates.sort(key=lambda c: c["dev_score"], reverse=True)
        top = candidates[:n_keep]

        if top[0]["dev_score"] >= 1.0:
            print(f"  [APE] Dev score 1.0 raggiunto — stop anticipato.")
            break

        n_variations_per_kept = max(1, n_proposals // n_keep)
        print(f"  [APE] Round {it}: variazioni dei top-{n_keep} ({n_variations_per_kept} per kept) …")

        new_candidates = []
        for parent in top:
            for _ in range(n_variations_per_kept):
                variation = variation_chain.invoke({
                    "problem":  question,
                    "previous": parent["prompt"],
                })
                code    = extract_code(target_chain.invoke({"user_prompt": variation}))
                dev_acc = evaluate_code(code, dev_io)
                new_candidates.append({"prompt": variation, "code": code, "dev_score": dev_acc})

        # Elitism: keep top-k from previous round, add new ones.
        candidates = top + new_candidates

        iterations.append({
            "round":      it,
            "candidates": [
                {"prompt": c["prompt"], "code": c["code"], "dev_score": c["dev_score"]}
                for c in new_candidates
            ],
            "best_dev":   max(c["dev_score"] for c in candidates),
        })
        print(f"  [APE] Round {it}: best_dev={iterations[-1]['best_dev']:.3f}")

    # ── Evaluate ALL final-population candidates on held-out TEST split ──
    print(f"  [APE] Valutazione popolazione finale ({len(candidates)}) sul test split …")
    for c in candidates:
        c["test_score"] = evaluate_code(c["code"], test_io)

    best = max(candidates, key=lambda c: c["dev_score"])
    print(f"[APE] Best: dev={best['dev_score']:.3f}  test={best['test_score']:.3f}")

    return {
        "best_prompt":      best["prompt"],
        "best_code":        best["code"],
        "best_dev":         best["dev_score"],
        "best_test":        best["test_score"],
        "n_dev":            n_dev,
        "n_test":           n_test,
        "final_population": [
            {
                "prompt":     c["prompt"],
                "code":       c["code"],
                "dev_score":  c["dev_score"],
                "test_score": c["test_score"],
            }
            for c in candidates
        ],
        "iterations":       iterations,
    }
