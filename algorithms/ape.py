"""
APE — Automatic Prompt Engineering (Zhou et al., 2022).
Ricampiona iterativamente il miglior prompt trovato; si ferma dopo
max_no_improve iterazioni consecutive senza miglioramento (beam width = 1).
"""

from utils import resampling, evaluate_code

MAX_NO_IMPROVE = 2


def run_ape(
    question: str,
    io_data: dict,
    solver_chain,
    model_optimizer: str,
    n_variants: int = 5,
    max_no_improve: int = MAX_NO_IMPROVE,
) -> dict:
    print("\n[APE] Avvio")

    best_prompt   = question
    best_accuracy = 0.0
    no_improve    = 0
    iterations    = []

    while no_improve < max_no_improve:
        iter_num = len(iterations) + 1
        print(f"  [APE] Iter {iter_num} — generazione varianti …")

        variants   = resampling(best_prompt, solver_chain, model_optimizer, n_variants)
        accuracies = [evaluate_code(v["code"], io_data) for v in variants]
        new_best   = max(accuracies) if accuracies else 0.0

        print(f"  [APE] Iter {iter_num}: new_best={new_best:.3f}  prev_best={best_accuracy:.3f}")

        iterations.append({
            "iter": iter_num,
            "variants": [
                {"prompt": v["prompt"], "code": v["code"], "accuracy": a}
                for v, a in zip(variants, accuracies)
            ],
            "best_accuracy": new_best,
        })

        if new_best > best_accuracy:
            best_prompt   = variants[accuracies.index(new_best)]["prompt"]
            best_accuracy = new_best
            no_improve    = 0
            print(f"  [APE] Miglioramento — reset contatore.")
        else:
            no_improve += 1
            print(f"  [APE] Nessun miglioramento ({no_improve}/{max_no_improve}).")

    print(f"[APE] Convergenza. Best: {best_accuracy:.3f}")
    return {"best_accuracy": best_accuracy, "iterations": iterations}
