from dotenv import load_dotenv
import os
import json
from datetime import datetime
from tqdm import tqdm

import utils
from algorithms import run_baseline, run_ape, run_apo

load_dotenv()

MODEL_TARGET      = os.getenv("MODEL_TARGET")
MODEL_OPTIMIZER   = os.getenv("MODEL_OPTIMIZER")
N_VARIANTS        = int(os.getenv("N_VARIANTS", 5))
N_INTRODUCTORY    = int(os.getenv("N_INTRODUCTORY", 10))
N_INTERVIEW       = int(os.getenv("N_INTERVIEW", 20))
N_COMPETITION     = int(os.getenv("N_COMPETITION", 30))
MAX_TEST_CASES    = int(os.getenv("MAX_TEST_CASES", 10))
APO_NUM_GRADIENTS = int(os.getenv("APO_NUM_GRADIENTS", 4))
APO_BEAM_WIDTH    = int(os.getenv("APO_BEAM_WIDTH", 4))
APO_MAX_ITERS     = int(os.getenv("APO_MAX_ITERS", 6))

OUTPUT_FILE   = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PASS_K_VALUES = [1, 3, 5, 10]


def _pass_at_k(accuracies: list[float]) -> dict:
    n = len(accuracies)
    return {f"pass@{k}": round(utils.compute_pass_at_k(accuracies, k), 4) for k in PASS_K_VALUES if k <= n}


def main():
    train_data, _ = utils.load_apps_dataset()
    samples      = utils.get_minibatch(train_data, n_per_difficulty={
        "introductory": N_INTRODUCTORY,
        "interview":    N_INTERVIEW,
        "competition":  N_COMPETITION,
    }, min_test_cases=MAX_TEST_CASES)
    solver_chain = utils.build_solver_chain(model=MODEL_TARGET)
    results      = []

    print(f"Campioni: {len(samples)} (intro={N_INTRODUCTORY}, interview={N_INTERVIEW}, competition={N_COMPETITION})")
    print(f"Output:   {OUTPUT_FILE}\n")

    for idx, problem in enumerate(tqdm(samples, desc="Problemi")):
        try:
            io_data = json.loads(problem["input_output"])
        except (json.JSONDecodeError, TypeError):
            print(f"[{idx}] input_output non valido, skip.")
            continue

        if not io_data.get("inputs"):
            print(f"[{idx}] Nessun test case, skip.")
            continue

        # Tronca ai primi MAX_TEST_CASES test case per rendere la metrica
        # comparabile tra difficoltà (evita il bias partial-credit).
        io_data["inputs"]  = io_data["inputs"][:MAX_TEST_CASES]
        io_data["outputs"] = io_data["outputs"][:MAX_TEST_CASES]

        print(f"\n=== Problema {idx} [{problem['difficulty']}] ===")

        n_test_cases  = len(io_data["inputs"])

        # Zero-shot: target model sul prompt originale, nessuna ottimizzazione
        original_code = utils.extract_code(solver_chain.invoke({"user_prompt": problem["question"]}))
        zero_shot_acc = utils.evaluate_code(original_code, io_data)

        baseline_result = run_baseline(
            question=problem["question"], io_data=io_data,
            solver_chain=solver_chain, model_optimizer=MODEL_OPTIMIZER,
        )

        ape_result = run_ape(
            question=problem["question"], io_data=io_data,
            solver_chain=solver_chain, model_optimizer=MODEL_OPTIMIZER, n_variants=N_VARIANTS,
        )
        ape_result["pass_at_k"] = _pass_at_k([v["accuracy"] for it in ape_result["iterations"] for v in it["variants"]])

        apo_result = run_apo(
            question=problem["question"], baseline_code=original_code, io_data=io_data,
            solver_chain=solver_chain, model_target=MODEL_TARGET, model_optimizer=MODEL_OPTIMIZER,
            num_gradients=APO_NUM_GRADIENTS, beam_width=APO_BEAM_WIDTH, max_iters=APO_MAX_ITERS,
        )
        apo_result["pass_at_k"] = _pass_at_k([c["accuracy"] for it in apo_result["iterations"] for c in it["all_candidates"]])

        results.append({
            "problem_idx":  idx,
            "difficulty":   problem["difficulty"],
            "n_test_cases": n_test_cases,
            "question":     problem["question"],
            "zero_shot":    {"accuracy": zero_shot_acc, "code": original_code},
            "baseline":     baseline_result,
            "APE":          ape_result,
            "APO":          apo_result,
        })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(
            f"  zero_shot={zero_shot_acc:.3f} | "
            f"baseline acc={baseline_result['accuracy']:.3f} | "
            f"APE pass@1={ape_result['pass_at_k'].get('pass@1', '-')} | "
            f"APO pass@1={apo_result['pass_at_k'].get('pass@1', '-')} "
            f"[{n_test_cases} test cases]"
        )

    print(f"\nEsperimenti completati. Risultati in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
