"""
Completa solo i problemi mancanti di un file results_*.json.

Uso:
    python resume_missing.py <results_file.json>
    python resume_missing.py                      # usa il results_*.json più recente

Auto-detect del dataset (APPS vs HumanEval+) dal contenuto del file.
"""
from dotenv import load_dotenv
import os
import sys
import glob
import json
from datetime import datetime
from tqdm import tqdm

import utils
from algorithms import run_baseline, run_ape, run_apo

load_dotenv()

MODEL_TARGET      = os.getenv("MODEL_TARGET")
MODEL_OPTIMIZER   = os.getenv("MODEL_OPTIMIZER")
N_VARIANTS        = int(os.getenv("N_VARIANTS", 5))
N_HUMANEVAL       = int(os.getenv("N_HUMANEVAL", 15))
N_PER_DIFFICULTY  = int(os.getenv("N_PER_DIFFICULTY", 5))
MAX_TEST_CASES    = int(os.getenv("MAX_TEST_CASES", 10))
APO_NUM_GRADIENTS   = int(os.getenv("APO_NUM_GRADIENTS",   2))
APO_NUM_EDITS       = int(os.getenv("APO_NUM_EDITS",       1))
APO_NUM_PARAPHRASES = int(os.getenv("APO_NUM_PARAPHRASES", 1))
APO_BEAM_WIDTH      = int(os.getenv("APO_BEAM_WIDTH",      2))
APO_MAX_ITERS       = int(os.getenv("APO_MAX_ITERS",       4))

AUGMENTED_DEV_FILE = os.getenv("AUG_FILE", "augmented_dev.json")
PASS_K_VALUES      = [1, 3, 5, 10]


def _pass_at_k(accuracies: list[float]) -> dict:
    n = len(accuracies)
    return {f"pass@{k}": round(utils.compute_pass_at_k(accuracies, k), 4) for k in PASS_K_VALUES if k <= n}


def _pick_input_file() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = sorted(glob.glob("results_*.json"), key=os.path.getmtime, reverse=True)
    candidates = [c for c in candidates if "_resume_" not in c]
    if not candidates:
        sys.exit("Nessun file results_*.json trovato. Passa il path come argomento.")
    return candidates[0]


def _detect_dataset(existing: list[dict]) -> str:
    if not existing:
        return os.getenv("DATASET", "apps")
    first = existing[0]
    if "difficulty" in first:
        return "apps"
    tid = str(first.get("task_id", ""))
    if tid.startswith("HumanEval/"):
        return "humaneval"
    return os.getenv("DATASET", "apps")


def _resume_humaneval(existing_results: list[dict], output_file: str) -> None:
    done_indices = {r["problem_idx"] for r in existing_results}
    print(f"Già completati: {sorted(done_indices)} ({len(done_indices)} problemi)")

    data    = utils.load_humaneval_plus_dataset()
    samples = utils.get_humaneval_plus_sample(data, n=N_HUMANEVAL)

    missing = [(idx, prob) for idx, prob in enumerate(samples) if idx not in done_indices]
    print(f"Mancanti: {[idx for idx, _ in missing]} ({len(missing)} problemi)")
    print(f"Output: {output_file}\n")

    results = list(existing_results)

    for idx, problem in tqdm(missing, desc="Problemi mancanti"):
        fn_name = problem["entry_point"]

        io_data = {
            "fn_name": fn_name,
            "inputs":  problem["inputs"],
            "outputs": [[r] for r in problem["results"]],
        }

        n_test_cases = len(io_data["inputs"])
        if n_test_cases < 2:
            print(f"[{idx}] {problem['task_id']} — solo {n_test_cases} test, skip.")
            continue

        print(f"\n=== Problema {idx} [{problem['task_id']}] (call-based: {fn_name}, {n_test_cases} tests) ===")

        dev_io, test_io = utils.split_io_data(io_data)
        n_dev_cases  = utils.population_size(dev_io)
        n_test_split = utils.population_size(test_io)
        print(f"   split: dev={n_dev_cases}, test={n_test_split} (totale={n_test_cases})")

        target_chain    = utils.build_target_chain(model=MODEL_TARGET, call_based=True, fn_name=fn_name)
        optimizer_chain = utils.build_optimizer_chain(model=MODEL_OPTIMIZER)

        original_code  = utils.extract_code(target_chain.invoke({"user_prompt": problem["prompt"]}))
        zero_shot_test = utils.evaluate_code(original_code, test_io)

        human_prompt      = utils.HUMAN_PROMPT_TEMPLATE.format(problem_description=problem["prompt"])
        human_code        = utils.extract_code(target_chain.invoke({"user_prompt": human_prompt}))
        human_prompt_test = utils.evaluate_code(human_code, test_io)

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
            "human_prompt": {"test_score": human_prompt_test, "code": human_code, "prompt": human_prompt},
            "baseline":     baseline_result,
            "APE":          ape_result,
            "APO":          apo_result,
        })

        results_sorted = sorted(results, key=lambda r: r["problem_idx"])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)

        print(
            f"  zero_shot={zero_shot_test:.3f} | "
            f"baseline test={baseline_result['test_score']:.3f} | "
            f"APE test={ape_result['best_test']:.3f} (pass@1={ape_result['pass_at_k'].get('pass@1', '-')}) | "
            f"APO test={apo_result['best_test']:.3f} (pass@1={apo_result['pass_at_k'].get('pass@1', '-')}) "
            f"[dev={n_dev_cases}, test={n_test_split}]"
        )

    print(f"\nCompletato. Risultati in {output_file}")


def _resume_apps(existing_results: list[dict], output_file: str) -> None:
    done_indices = {r["problem_idx"] for r in existing_results}
    print(f"Già completati: {sorted(done_indices)} ({len(done_indices)} problemi)")

    train_data, _ = utils.load_apps_dataset()
    samples      = utils.get_minibatch(train_data,
                                       n_per_difficulty=N_PER_DIFFICULTY,
                                       min_test_cases=MAX_TEST_CASES)

    augmented = utils.load_augmented_dev(AUGMENTED_DEV_FILE, samples=samples)
    if augmented:
        print(f"Dev augmentato caricato da {AUGMENTED_DEV_FILE}: {len(augmented)} problemi coperti")
    else:
        print(f"Nessun dev augmentato ({AUGMENTED_DEV_FILE}): split classico")

    missing = [(idx, prob) for idx, prob in enumerate(samples) if idx not in done_indices]
    print(f"Mancanti: {[idx for idx, _ in missing]} ({len(missing)} problemi)")
    print(f"Output: {output_file}\n")

    results = list(existing_results)

    for idx, problem in tqdm(missing, desc="Problemi mancanti"):
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
            print(f"[{idx}] Solo {n_test_cases} test case, skip.")
            continue

        fn_name    = io_data.get("fn_name", "")
        call_based = bool(fn_name)

        print(f"\n=== Problema {idx} [{problem['difficulty']}] {'(call-based: ' + fn_name + ')' if call_based else '(stdin-based)'} ===")

        aug_entry = augmented.get(idx)
        io_data_split = utils.inject_augmented(io_data, aug_entry)
        dev_io, test_io = utils.split_io_data(io_data_split)
        n_dev_cases  = utils.population_size(dev_io)
        n_test_split = utils.population_size(test_io)
        mode = "augmented" if aug_entry else "classic_split"
        print(f"   split [{mode}]: dev={n_dev_cases}, test={n_test_split} (official_total={n_test_cases})")

        target_chain    = utils.build_target_chain(model=MODEL_TARGET, call_based=call_based, fn_name=fn_name)
        optimizer_chain = utils.build_optimizer_chain(model=MODEL_OPTIMIZER)

        original_code  = utils.extract_code(target_chain.invoke({"user_prompt": problem["question"]}))
        zero_shot_test = utils.evaluate_code(original_code, test_io)

        human_prompt      = utils.HUMAN_PROMPT_TEMPLATE.format(problem_description=problem["question"])
        human_code        = utils.extract_code(target_chain.invoke({"user_prompt": human_prompt}))
        human_prompt_test = utils.evaluate_code(human_code, test_io)

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
            "human_prompt": {"test_score": human_prompt_test, "code": human_code, "prompt": human_prompt},
            "baseline":     baseline_result,
            "APE":          ape_result,
            "APO":          apo_result,
        })

        results_sorted = sorted(results, key=lambda r: r["problem_idx"])
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)

        print(
            f"  zero_shot={zero_shot_test:.3f} | "
            f"baseline test={baseline_result['test_score']:.3f} | "
            f"APE test={ape_result['best_test']:.3f} (pass@1={ape_result['pass_at_k'].get('pass@1', '-')}) | "
            f"APO test={apo_result['best_test']:.3f} (pass@1={apo_result['pass_at_k'].get('pass@1', '-')}) "
            f"[dev={n_dev_cases}, test={n_test_split}]"
        )

    print(f"\nCompletato. Risultati in {output_file}")


def main():
    input_file = _pick_input_file()
    if not os.path.exists(input_file):
        sys.exit(f"File non trovato: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)

    dataset = _detect_dataset(existing_results)
    stem = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{stem}_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Input:   {input_file}")
    print(f"Dataset: {dataset} (auto-detect)")

    if dataset == "humaneval":
        _resume_humaneval(existing_results, output_file)
    else:
        _resume_apps(existing_results, output_file)


if __name__ == "__main__":
    main()
