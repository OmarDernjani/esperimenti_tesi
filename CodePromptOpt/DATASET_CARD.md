# CodePromptOpt dataset card

## Purpose and contents

Train a direct prompt optimizer conditioned on a previously unseen programming problem and a target code-generation model. Targets are optimized prompts actually observed in Omar Dernjani's per-problem APE/APO experiments. Preferences describe observed functional-correctness differences on development cases, not universal prompt quality.

The source collection contains 90 tasks: 45 APPS tasks (15 introductory, 15 interview, 15 competition) and 45 HumanEval+ tasks. Refer to the generated statistics for current SFT/DPO counts, missing fields and distributions. Experimental results under `dataset_code_prompt/` are the sole observation source; no new inference or synthetic data are used.

## Correlation and partitioning

Experiments optimize each programming problem separately. Multiple candidates for the same task are correlated search observations, not independent tasks. Prompt candidates, runs, observations and derived examples share a deterministic task split (default 70 train, 10 validation, 10 internal test). Benchmark/APPS difficulty stratification is used where feasible. The effective sample size is the number of tasks, not the number of prompts or pairs.

The internal split is retrospective: all original tasks already underwent optimization in the source experiments. Internal test tasks have canonical raw observations but no generated SFT/DPO label files. Keep canonical test-task scores and optimized prompts away from model training, checkpoint selection and prompt tuning. The build never selects a test-task label.

## Label construction

Development and held-out test observations are strictly separate. Only dev observations produce SFT labels or DPO preferences. Zero-shot/human/baseline results have test-only scores and are not eligible for labels. SFT labels are best **observed**, positive-scoring prompts within an evidenced comparable setting. Ties prefer shorter complete prompts, then stable IDs. Different experiment files are not ranked together without suite identities. DPO requires same task, target and stored configuration record, strict higher dev score, minimum margin 0.20, diverse endpoints, and at most five pairs per task/target.

Inputs preserve the exact original problem and condition explicitly on the filename target-model label. Completions, feedback and scores never enter model training inputs. Use the shared input template in README and both sample configurations.

## Quality and limitations

Exact original-problem containment is required to avoid unverifiable loss of requirements. This rejects many potentially valid paraphrases and can bias the retained sample. Additional solution code, feedback leakage and requirement overrides are heuristically screened. A prompt can retain the exact problem while introducing a subtle contradiction; automated passing does not guarantee semantic preservation. Manual review is recommended, and all decisions and raw candidates are retained for audit.

No parent identifiers or persisted critiques permit reliable lineage reconstruction. Discarded intermediate APO code and the initial APO prompt from continued searches are incompletely logged. Missing completions remain null. Dev suite contents/fingerprints, passed-test counts, statuses, generation seeds, decoding settings and historical execution versions are absent. Current source-code defaults must not be treated as evidence of historical settings.

Model labels come from filenames. `qwen3.5-coder` is an archive label and must not be mistaken for a verified serving model identifier. Quantization is not inferred. APPS original benchmark IDs/interfaces were not saved. Suite comparability is limited to the same stored task record, as supported by the shared dev_io call structure. Scores reflect limited dev samples and one or more stochastic completions rather than a controlled estimate of expected prompt utility. Conflicting repeated scores are excluded from training.

SFT can retain multiple settings for a task/target; they remain separate examples and should not be interpreted as a single global ranking. There are no dev baselines for a reliable improvement-over-zero-shot statistic. Within-search score spread is reported separately and is not evidence of causal improvement.

## Recommended evaluation

Use validation tasks only for checkpoint/hyperparameter selection. On reserved internal test tasks, have the trained optimizer produce prompts from the original problem and target-model label alone. In a separate future evaluation, execute fresh target-model completions with a fixed documented generation policy, multiple seeds where appropriate, and truly held-out suites. Compare against raw problems and fixed templates under identical conditions. This pipeline does not run that evaluation.

Report correctness aggregated by task, benchmark and APPS difficulty; use task-level uncertainty estimates. Evaluate transfer on separately sourced unseen tasks/benchmarks and different target models. Investigate exact/near-duplicate task overlap with any upstream pretraining corpus; a repository split cannot establish absence of pretraining contamination. Do not use stored optimized test-task prompts as inputs to claim generalization.

## Provenance and distribution

Canonical candidates/evaluations retain source files, document indices and JSON pointers. Source SHA-256 digests and output artifact hashes permit traceability and deterministic rebuilds. Original experiments and algorithms are unchanged. No new license is assigned to upstream benchmark text or experiment outputs; verify applicable benchmark and repository rights before redistribution. This dataset construction does not supply hidden tests or reference solutions in training descriptions.
