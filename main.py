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
DATASET           = os.getenv("DATASET", "apps")          # "apps" o "humaneval"
N_VARIANTS        = int(os.getenv("N_VARIANTS", 5))
N_INTRODUCTORY    = int(os.getenv("N_INTRODUCTORY", 5))
N_INTERVIEW       = int(os.getenv("N_INTERVIEW", 5))
N_COMPETITION     = int(os.getenv("N_COMPETITION", 5))
N_HUMANEVAL       = int(os.getenv("N_HUMANEVAL", 15))
MAX_TEST_CASES    = int(os.getenv("MAX_TEST_CASES", 10))
APO_NUM_GRADIENTS = int(os.getenv("APO_NUM_GRADIENTS", 4))
APO_BEAM_WIDTH    = int(os.getenv("APO_BEAM_WIDTH", 4))
APO_MAX_ITERS     = int(os.getenv("APO_MAX_ITERS", 6))

OUTPUT_FILE   = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PASS_K_VALUES = [1, 3, 5, 10]


def _pass_at_k(accuracies: list[float]) -> dict:
    n = len(accuracies)

    return {f"pass@{k}": round(utils.compute_pass_at_k(accuracies, k), 4) for k in PASS_K_VALUES if k <= n}


def main_apps():
    train_data, _ = utils.load_apps_dataset()
    samples      = utils.get_minibatch(train_data, n_per_difficulty={
        "introductory": N_INTRODUCTORY,
        "interview":    N_INTERVIEW,
        "competition":  N_COMPETITION,
    },
    min_test_cases=MAX_TEST_CASES
    )

    results      = []

    print(f"[APPS] numero campioni: {len(samples)} (intro={N_INTRODUCTORY}, interview={N_INTERVIEW}, competition={N_COMPETITION})")
    print(f"output:   {OUTPUT_FILE}\n")

    for idx, problem in enumerate(tqdm(samples, desc="Problemi")):
        try:
            io_data = json.loads(problem["input_output"])
        except (json.JSONDecodeError, TypeError):
            print(f"[{idx}] input_output non valido, skip.")
            continue

        if not io_data.get("inputs"):
            print(f"[{idx}] Nessun test case, skip.")
            continue


        io_data["inputs"]  = io_data["inputs"][:MAX_TEST_CASES]
        io_data["outputs"] = io_data["outputs"][:MAX_TEST_CASES]

        fn_name    = io_data.get("fn_name", "")
        call_based = bool(fn_name)

        print(f"\n Problema {idx} [{problem['difficulty']}] {'(call-based: ' + fn_name + ')' if call_based else '(stdin-based)'}")

        n_test_cases  = len(io_data["inputs"])

        direct_chain = utils.build_direct_chain(model=MODEL_TARGET, call_based=call_based, fn_name=fn_name)
        solver_chain = utils.build_solver_chain(model_optimizer=MODEL_OPTIMIZER, model_target=MODEL_TARGET, call_based=call_based, fn_name=fn_name)

        # Zero-shot (diretto al target, nessun prompt engineering)
        original_code = utils.extract_code(direct_chain.invoke({"user_prompt": problem["question"]}))
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

    print(f"\nEsperimenti APPS completati.")


def main_humaneval():
    data    = utils.load_humaneval_dataset()
    samples = utils.get_humaneval_sample(data, n=N_HUMANEVAL)
    results = []

    print(f"[HumanEval] numero campioni: {len(samples)}")
    print(f"output:   {OUTPUT_FILE}\n")

    for idx, problem in enumerate(tqdm(samples, desc="Problemi")):
        entry_point = problem["entry_point"]
        fn_name     = entry_point

        io_data = {
            "humaneval":   True,
            "test_code":   problem["test"],
            "entry_point": entry_point,
        }

        n_test_cases = len(utils._extract_assertions(problem["test"]))
        if n_test_cases == 0:
            print(f"[{idx}] {problem['task_id']} — nessuna assertion, skip.")
            continue

        print(f"\n Problema {idx} [{problem['task_id']}] (call-based: {fn_name}, {n_test_cases} assertions)")

        direct_chain = utils.build_direct_chain(model=MODEL_TARGET, call_based=True, fn_name=fn_name)
        solver_chain = utils.build_solver_chain(model_optimizer=MODEL_OPTIMIZER, model_target=MODEL_TARGET, call_based=True, fn_name=fn_name)

        # Zero-shot
        original_code = utils.extract_code(direct_chain.invoke({"user_prompt": problem["prompt"]}))
        zero_shot_acc = utils.evaluate_code(original_code, io_data)

        baseline_result = run_baseline(
            question=problem["prompt"], io_data=io_data,
            solver_chain=solver_chain, model_optimizer=MODEL_OPTIMIZER,
        )

        ape_result = run_ape(
            question=problem["prompt"], io_data=io_data,
            solver_chain=solver_chain, model_optimizer=MODEL_OPTIMIZER, n_variants=N_VARIANTS,
        )
        ape_result["pass_at_k"] = _pass_at_k([v["accuracy"] for it in ape_result["iterations"] for v in it["variants"]])

        apo_result = run_apo(
            question=problem["prompt"], baseline_code=original_code, io_data=io_data,
            solver_chain=solver_chain, model_target=MODEL_TARGET, model_optimizer=MODEL_OPTIMIZER,
            num_gradients=APO_NUM_GRADIENTS, beam_width=APO_BEAM_WIDTH, max_iters=APO_MAX_ITERS,
        )
        apo_result["pass_at_k"] = _pass_at_k([c["accuracy"] for it in apo_result["iterations"] for c in it["all_candidates"]])

        results.append({
            "problem_idx":  idx,
            "task_id":      problem["task_id"],
            "n_test_cases": n_test_cases,
            "question":     problem["prompt"],
            "zero_shot":    {"accuracy": zero_shot_acc, "code": original_code},
            "baseline":     baseline_result,
            "APE":          ape_result,
            "APO":          apo_result,
        })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nEsperimenti HumanEval completati.")


def main():
    if DATASET == "humaneval":
        main_humaneval()
    else:
        main_apps()


if __name__ == "__main__":
    main()
