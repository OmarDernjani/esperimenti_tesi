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
N_PER_DIFFICULTY  = int(os.getenv("N_PER_DIFFICULTY", 5))
N_HUMANEVAL       = int(os.getenv("N_HUMANEVAL", 15))
MAX_TEST_CASES    = int(os.getenv("MAX_TEST_CASES", 10))
APO_NUM_GRADIENTS   = int(os.getenv("APO_NUM_GRADIENTS",   2))
APO_NUM_EDITS       = int(os.getenv("APO_NUM_EDITS",       1))
APO_NUM_PARAPHRASES = int(os.getenv("APO_NUM_PARAPHRASES", 1))
APO_BEAM_WIDTH      = int(os.getenv("APO_BEAM_WIDTH",      2))
APO_MAX_ITERS       = int(os.getenv("APO_MAX_ITERS",       4))

AUGMENTED_DEV_FILE = os.getenv("AUG_FILE", "augmented_dev.json")
OUTPUT_FILE   = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
PASS_K_VALUES = [1, 3, 5, 10]


def _pass_at_k(accuracies: list[float]) -> dict:
    n = len(accuracies)

    return {f"pass@{k}": round(utils.compute_pass_at_k(accuracies, k), 4) for k in PASS_K_VALUES if k <= n}


def main_apps():
    train_data, _ = utils.load_apps_dataset()
    samples      = utils.get_minibatch(train_data,
                                       n_per_difficulty=N_PER_DIFFICULTY,
                                       min_test_cases=MAX_TEST_CASES)

    augmented = utils.load_augmented_dev(AUGMENTED_DEV_FILE, samples=samples)
    if augmented:
        print(f"Dev augmentato caricato da {AUGMENTED_DEV_FILE}: {len(augmented)} problemi coperti")
    else:
        print(f"Nessun dev augmentato ({AUGMENTED_DEV_FILE}): uso split classico sui test ufficiali")

    results      = []

    print(f"[APPS] numero campioni: {len(samples)} (N_PER_DIFFICULTY={N_PER_DIFFICULTY})")
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

        n_test_cases = len(io_data["inputs"])
        if n_test_cases < 2:
            print(f"[{idx}] Solo {n_test_cases} test case — split dev/test impossibile, skip.")
            continue

        fn_name    = io_data.get("fn_name", "")
        call_based = bool(fn_name)

        print(f"\n Problema {idx} [{problem['difficulty']}] {'(call-based: ' + fn_name + ')' if call_based else '(stdin-based)'}")

        aug_entry = augmented.get(idx)
        io_data_split = utils.inject_augmented(io_data, aug_entry)
        dev_io, test_io = utils.split_io_data(io_data_split)
        n_dev_cases  = utils.population_size(dev_io)
        n_test_split = utils.population_size(test_io)
        mode = "augmented" if aug_entry else "classic_split"
        print(f"   split [{mode}]: dev={n_dev_cases}, test={n_test_split} (official_total={n_test_cases})")

        target_chain    = utils.build_target_chain(model=MODEL_TARGET, call_based=call_based, fn_name=fn_name)
        optimizer_chain = utils.build_optimizer_chain(model=MODEL_OPTIMIZER)

        # Zero-shot: nessun prompt engineering, valutato sul test split
        original_code  = utils.extract_code(target_chain.invoke({"user_prompt": problem["question"]}))
        zero_shot_test = utils.evaluate_code(original_code, test_io)

        baseline_result = run_baseline(
            question=problem["question"], test_io=test_io,
            target_chain=target_chain, optimizer_chain=optimizer_chain,
        )

        ape_result = run_ape(
            question=problem["question"], dev_io=dev_io, test_io=test_io,
            target_chain=target_chain, optimizer_chain=optimizer_chain,
            model_optimizer=MODEL_OPTIMIZER, n_proposals=N_VARIANTS,
        )
        ape_result["pass_at_k"] = _pass_at_k([c["test_score"] for c in ape_result["final_population"]])

        apo_result = run_apo(
            question=problem["question"], baseline_code=original_code,
            dev_io=dev_io, test_io=test_io,
            target_chain=target_chain, model_optimizer=MODEL_OPTIMIZER,
            optimizer_chain=optimizer_chain,
            num_gradients=APO_NUM_GRADIENTS, num_edits=APO_NUM_EDITS,
            num_paraphrases=APO_NUM_PARAPHRASES,
            beam_width=APO_BEAM_WIDTH, max_iters=APO_MAX_ITERS,
        )
        apo_result["pass_at_k"] = _pass_at_k([c["test_score"] for c in apo_result["final_pool"]])

        results.append({
            "problem_idx":  idx,
            "difficulty":   problem["difficulty"],
            "n_test_cases": n_test_cases,
            "n_dev":        n_dev_cases,
            "n_test":       n_test_split,
            "question":     problem["question"],
            "zero_shot":    {"test_score": zero_shot_test, "code": original_code},
            "baseline":     baseline_result,
            "APE":          ape_result,
            "APO":          apo_result,
        })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nEsperimenti APPS completati.")


def main_humaneval():
    data    = utils.load_humaneval_plus_dataset()
    samples = utils.get_humaneval_plus_sample(data, n=N_HUMANEVAL)
    results = []

    print(f"[HumanEval+] numero campioni: {len(samples)}")
    print(f"output:   {OUTPUT_FILE}\n")

    for idx, problem in enumerate(tqdm(samples, desc="Problemi")):
        entry_point = problem["entry_point"]
        fn_name     = entry_point

        io_data = {
            "fn_name": fn_name,
            "inputs":  problem["inputs"],
            "outputs": [[r] for r in problem["results"]],
        }

        n_test_cases = len(io_data["inputs"])
        if n_test_cases < 2:
            print(f"[{idx}] {problem['task_id']} — solo {n_test_cases} test, split dev/test impossibile, skip.")
            continue

        print(f"\n Problema {idx} [{problem['task_id']}] (call-based: {fn_name}, {n_test_cases} tests)")

        # Split unico per il problema
        dev_io, test_io = utils.split_io_data(io_data)
        n_dev_cases  = utils.population_size(dev_io)
        n_test_split = utils.population_size(test_io)
        print(f"   split: dev={n_dev_cases}, test={n_test_split} (totale={n_test_cases})")

        target_chain    = utils.build_target_chain(model=MODEL_TARGET, call_based=True, fn_name=fn_name)
        optimizer_chain = utils.build_optimizer_chain(model=MODEL_OPTIMIZER)

        # Zero-shot
        original_code  = utils.extract_code(target_chain.invoke({"user_prompt": problem["prompt"]}))
        zero_shot_test = utils.evaluate_code(original_code, test_io)

        baseline_result = run_baseline(
            question=problem["prompt"], test_io=test_io,
            target_chain=target_chain, optimizer_chain=optimizer_chain,
        )

        ape_result = run_ape(
            question=problem["prompt"], dev_io=dev_io, test_io=test_io,
            target_chain=target_chain, optimizer_chain=optimizer_chain,
            model_optimizer=MODEL_OPTIMIZER, n_proposals=N_VARIANTS,
        )
        ape_result["pass_at_k"] = _pass_at_k([c["test_score"] for c in ape_result["final_population"]])

        apo_result = run_apo(
            question=problem["prompt"], baseline_code=original_code,
            dev_io=dev_io, test_io=test_io,
            target_chain=target_chain, model_optimizer=MODEL_OPTIMIZER,
            optimizer_chain=optimizer_chain,
            num_gradients=APO_NUM_GRADIENTS, num_edits=APO_NUM_EDITS,
            num_paraphrases=APO_NUM_PARAPHRASES,
            beam_width=APO_BEAM_WIDTH, max_iters=APO_MAX_ITERS,
        )
        apo_result["pass_at_k"] = _pass_at_k([c["test_score"] for c in apo_result["final_pool"]])

        results.append({
            "problem_idx":  idx,
            "task_id":      problem["task_id"],
            "n_test_cases": n_test_cases,
            "n_dev":        n_dev_cases,
            "n_test":       n_test_split,
            "question":     problem["prompt"],
            "zero_shot":    {"test_score": zero_shot_test, "code": original_code},
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
