# CodePromptOpt data audit

Source files below are the only experimental data inputs. Original experiments and algorithms are unchanged.

| Source | Documents | Rows | SHA-256 |
|---|---:|---:|---|
| results_esperimenti/best_prompts.json | 1 | 90 | 8bedb01708004dd0eefa59b10cdc3ab30564fc65afed8cd6112cc7841874b999 |
| results_esperimenti/results_APPS/qwen3.5+qwen3.5-coder.json | 1 | 31 | 6b7697568779e868940f7d0ae41fcc323bfab207a5356856007bfebdd80f76c6 |
| results_esperimenti/results_APPS/qwen3.5-coder+qwen3.5.json | 1 | 45 | 6f02cafa90b21132a23d279c52e1df565be95f880a7e09688867401d9eac15da |
| results_esperimenti/results_APPS/qwen3.5-coder_self.json | 1 | 45 | e164c5d4efb813fc8429bce71df4fc629fe77f09c014c92a63c1e0f7abd48c6e |
| results_esperimenti/results_APPS/qwen3.5_self.json | 1 | 45 | 093cc2a895d72501534cb845dfe1e406378b1e1724a8440e8270c8bc2488ac5e |
| results_esperimenti/results_HUMANEVAL+/qwen3.5+qwen3.5-coder.json | 1 | 45 | 0f02c11abf2070a57398c61e3a4534b6b0d8f47dde005ac3e657f8f6a37dbd30 |
| results_esperimenti/results_HUMANEVAL+/qwen3.5-coder+qwen3.5.json | 1 | 45 | 6ff5192e096e8cf52604270fa360b02be90fe7c89a064382c198299a6ed85fc4 |
| results_esperimenti/results_HUMANEVAL+/qwen3.5-coder_self.json | 1 | 45 | 5a92f202f3aaa3f7672f58ab7e685edf05cd264857fa740fb90693c1a01931f0 |
| results_esperimenti/results_HUMANEVAL+/qwen3.5_self.json | 2 | 46 | 6d56c0ee06d1dad034a1ee1c9d183694c71f648fc1f74ce5eb966e0d77745e72 |

## Counts

- Unique tasks: 90.
- Task/configuration records: 346.
- Experimental method runs: 1730.
- Canonical candidates: 7577; distinct exact prompt strings: 7040.
- Observed dev evaluations: 6548; test evaluations: 6299.
- Distinct (candidate, code) completions: 7329.

## Verified mapping and limitations

`question` is preserved byte-for-byte as decoded Unicode text. HumanEval source IDs are saved; APPS problem_idx is a local sampling index, not an APPS dataset ID. Task IDs hash benchmark and original problem.

APE iterations contain new proposals with code and dev_score. Final populations repeat surviving proposals and add test_score; best summaries repeat the selected observation. APO all_candidates preserves prompts/dev scores/source kind, beam preserves code for selected members, and final_pool preserves code/test scores for the final pool. Intermediate discarded APO completions and initial APO prompts from searches that continued are not fully logged.

Candidates are exact-text deduplicated within a method run. Primary iteration occurrences are separate observed evaluations; beam, final, and best copies merge into compatible existing observations. Missing code is enriched only by a matching copied observation. Counts do not treat saved aliases as independent evaluations.

No parent IDs or persisted critiques exist in observed candidate schemas; source=gradient/paraphrase does not identify parents. Lineage cannot be reconstructed reliably. All parent_candidate_id and unavailable feedback remain null.

Comparable dev pools combine APE and APO only within the same stored task/configuration record. main.py passes the same dev_io to both methods, and both call evaluate_code. Exact test_suite_id is unavailable. Equal test counts across different files do not establish identical suites; cross-file rankings and pairs are prohibited.

SFT emits one best positive-scoring observed prompt per task/target/comparison group. Multiple groups for the same target remain separate; duplicate identical outputs are retained once using sample_id order, never a cross-suite score ranking.

Filename model labels identify the target first and optimizer second. qwen3.5-coder is the filename label, not a verified Ollama model identifier. Exact quantization, generation seeds, decoding settings, and historical executable versions were not persisted in these source files and are null. Current code defaults are not asserted as historical runtime settings.

Zero-shot, human prompt and baseline have held-out test scores only and cannot supply SFT labels or DPO preferences. best_prompts.json is an unscored archive: exact task/prompt matches link to existing candidates. HumanEval archive problems may differ only by outer whitespace; a matching benchmark task ID plus stripped-text equality establishes identity, while the canonical original problem remains unchanged. Unmatched prompts remain in archival runs with unknown target and no fabricated evaluations.

Concatenated JSON documents are explicitly parsed; identical repeated records in one source file are collapsed and reported in source_issues.jsonl. Nonidentical repeats remain separate conditions.

Quality filtering requires the exact full original problem to occur in the prompt. Paraphrases may be semantically correct but cannot be certified automatically and are excluded rather than repaired. Additional code, evaluation feedback and requirement overrides are screened heuristically; passing is not proof of semantic equivalence. Manual review remains necessary.

All unavailable fields use null. passed_tests is not reconstructed by multiplying rounded scores. Suite contents and reference solutions are never added to training inputs. Language is python as established by the experimental generator/evaluator. APPS interfaces cannot be recovered reliably and remain null.

## Actual source schemas

```json
{
  "APE": [
    "best_code",
    "best_dev",
    "best_prompt",
    "best_test",
    "final_population",
    "iterations",
    "n_dev",
    "n_test",
    "pass_at_k"
  ],
  "APE/candidates": [
    "code",
    "dev_score",
    "prompt"
  ],
  "APE/iteration": [
    "best_dev",
    "candidates",
    "round"
  ],
  "APO": [
    "best_code",
    "best_dev",
    "best_prompt",
    "best_test",
    "final_pool",
    "iterations",
    "n_dev",
    "n_test",
    "pass_at_k"
  ],
  "APO/all_candidates": [
    "dev_score",
    "prompt",
    "source"
  ],
  "APO/beam": [
    "code",
    "dev_score",
    "prompt",
    "source"
  ],
  "APO/iteration": [
    "all_candidates",
    "beam",
    "best_dev",
    "iter",
    "n_gradient",
    "n_paraphrase",
    "pool_size"
  ],
  "baseline": [
    "code",
    "enriched_prompt",
    "test_score"
  ],
  "human_prompt": [
    "code",
    "prompt",
    "test_score"
  ],
  "zero_shot": [
    "code",
    "test_score"
  ]
}
```

## Missing data

```json
{
  "candidate_feedback": 7577,
  "candidate_parent_ids": 7577,
  "candidates_without_dev": 1038,
  "evaluations_without_execution_status": 12847,
  "evaluations_without_generated_code": 256,
  "evaluations_without_passed_tests": 12847,
  "evaluations_without_suite_id": 12847,
  "runs_without_decoding_parameters": 1730,
  "runs_without_exact_model_quantization": 1730,
  "runs_without_seed": 1730,
  "tasks_without_interface": 45,
  "tasks_without_original_benchmark_source_id": 45
}
```
