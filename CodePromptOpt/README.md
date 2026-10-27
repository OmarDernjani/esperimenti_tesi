# CodePromptOpt

Per la nuova direzione **strategie riutilizzabili e Strategy-Guided Dynamic Prompting**, vedere [strategies/README.md](strategies/README.md): estrazione di tutti i candidati, annotazione con evidenze, analisi esplorativa e prompt M0–M4 con ablazioni.

Per addestrare Qwen3 e Phi-4-mini come prompt optimizer con QLoRA e confrontare la correttezza del codice generato, vedere [finetuning/README.md](finetuning/README.md). La cartella include training, inferenza, valutazione e Containerfile Podman.

An offline, reproducible dataset for **programming problem → observed optimized code-generation prompt**, supporting SFT and DPO. Only result files under `dataset_code_prompt/` supply experimental observations. No LLM calls, code execution, benchmark downloads, or synthetic observations are involved. Python 3.10+ and its standard library are sufficient.

From the repository root:

```powershell
python -m CodePromptOpt.build --input-dir dataset_code_prompt --output-dir CodePromptOpt/data
python -m CodePromptOpt.validate --data-dir CodePromptOpt/data --input-dir dataset_code_prompt
python -m unittest discover -s CodePromptOpt/tests -v
```

On the current Windows workspace, a project-local portable interpreter is available (ignored by Git):

```powershell
& .\CodePromptOpt\.runtime\python\python.exe -m CodePromptOpt.build --input-dir dataset_code_prompt --output-dir CodePromptOpt/data
& .\CodePromptOpt\.runtime\python\python.exe -m CodePromptOpt.validate --data-dir CodePromptOpt/data --input-dir dataset_code_prompt
& .\CodePromptOpt\.runtime\python\python.exe -m unittest discover -s CodePromptOpt/tests -v
```

The import package uses the `CodePromptOpt` spelling on all platforms. Windows treats `codepromptopt` and `CodePromptOpt` as the same directory; using the exact package spelling also works on Linux.

## Outputs

- `data/tasks.jsonl`: one original task, exact original `question`, source IDs where logged, metadata and task split.
- `data/runs.jsonl`: one task/configuration/method run with source pointer and conservative comparison group.
- `data/candidates.jsonl`: distinct exact prompt within a run, first known iteration, all appearances, final selection flag and provenance.
- `data/evaluations.jsonl`: observed dev/test scores, available completions, documented aliases and unknown fields as null.
- `data/sft/{train,validation}.jsonl` and `data/dpo/{train,validation}.jsonl`: dev-labelled training records; no internal-test training exports.
- `data/splits.json`: the shared task partition; default 70/10/10 for 90 tasks.
- `data/filtering_report.jsonl`: every flagged/excluded candidate and reasons. Raw canonical observations remain available, including malformed prompts explicitly flagged in this report.
- `data/source_issues.jsonl`: parsing/schema issues and duplicate source records.
- `data/{statistics,validation,build_manifest,comparable_groups,artifact_hashes}.json`: machine-readable statistics, checks, parameters, source checksums and output checksums.
- [DATA_AUDIT.md](DATA_AUDIT.md) and [DATASET_STATISTICS.md](DATASET_STATISTICS.md): generated human-readable reports, also stored in `data/`.
- [DATASET_CARD.md](DATASET_CARD.md): intended use and limitations.

## Construction

Read all JSON/JSONL result files recursively, including concatenated JSON documents. Preserve all logged candidates; deduplicate exact prompt text within a method run. APE iteration proposals are primary observations; final populations and best summaries are copies. APO full candidate pools are primary observations; beams enrich available code, and final pools/summaries carry copied dev and held-out test observations. Primary repeated generations of the same prompt remain separate evaluations. Identical duplicated source task records are collapsed with an explicit report.

HumanEval+ task IDs and signatures are available. APPS sampling indices are local experiment indices, so `source_id`, `interface` and `entry_point` remain null rather than guessed from generated code. Model names are filename labels (`qwen3.5`, `qwen3.5-coder`), with target first in mixed filenames. The latter is **not** asserted to be an exact runtime model name. Quantization and historical decoding configurations are unavailable in these inputs.

`best_prompts.json` has no scores or originating model labels. Its 90 problem records are matched by exact original problem, or by a verified matching HumanEval source ID and equality after stripping outer whitespace. The original canonical problem is unchanged. Already observed exact prompts get unscored archive links; unmatched prompts are preserved without fabricated labels or model assignments.

Candidates with conflicting repeated dev scores are excluded from training. The same prompt receiving conflicting scores across methods in one comparable group is also excluded. Only APE/APO prompts with the full original problem appearing verbatim are eligible. The strict default also screens added complete solutions, test feedback and requirement overrides. Semantically equivalent rewrites cannot be verified by this pipeline and are conservatively excluded. Heuristic screening is not a proof of correctness; review flagged and retained prompts before training.

SFT selects the highest positive dev score, then shortest prompt in characters, then candidate ID. Within one stored task/configuration record, APE and APO share the same dev tests in `main.py`, so their candidates can be combined. Different source files lack suite fingerprints and are never merged into a dev ranking merely because counts match. Thus a task/target may have multiple SFT records, one per independent setting; identical SFT outputs are deduplicated without ranking across settings.

DPO compares only candidates in the same task/target/comparison group, requires a strictly positive margin of at least 0.20, excludes identical prompts and conflicting observations, and caps the total at five pairs per task/target. Greedy selection prefers unused endpoints, then larger margins and deterministic IDs. No held-out test field participates in SFT or DPO labels.

## Common input template

Use **exactly the same conditioning template** for SFT and DPO:

```text
{instruction}

Target code generation model: {target_model}

Programming problem:
{input}
```

SFT assistant text is `output`. DPO chosen/rejected assistant texts are `chosen`/`rejected`. Treat model labels consistently; do not omit the target-model line. Do not add code completions, scores, critiques or hidden tests to the training input. Example trainer-neutral configurations are in `configs/`; adapt their field mappings to your trainer without changing the conditioning template. They are examples, not a training implementation, and this task performs no training.

## Configuration

```powershell
python -m CodePromptOpt.build --input-dir dataset_code_prompt --output-dir CodePromptOpt/data --seed 42 --split-ratios 70 10 10 --min-margin 0.20 --max-pairs 5 --deduplication exact --quality-filtering strict
```

Split ratios are nonnegative weights, not hard counts. Largest-remainder allocation produces the requested global task counts and stratifies by benchmark/APPS difficulty where feasible. Custom small datasets retain disjoint task partitions even when a stratum cannot populate every split. Defaults yield 70/10/10 for this source collection.

`--deduplication whitespace` optionally collapses whitespace-normalized prompts in training pools; canonical text always remains exact. `--quality-filtering flag` keeps heuristic warnings on samples, but still requires a complete original problem, reliable dev observations and comparable conditions. Use strict filtering for the released dataset. Neither option manufactures missing data.

Build into a separate output directory outside the input tree. Repeating the build with unchanged inputs/options gives identical artifact hashes. `validate` checks every exported source pointer and score/code value against the input files, output hashes, task splits, label evidence, preferences and SFT ranking. Unit tests include held-out-score invariance, repeated candidates, missing/conflicting dev observations, split integrity, reproducibility and deliberately corrupted labels.

Current experimental scripts are inspected as evidence of schema/evaluation semantics and are never imported or modified. The build does not read environment files or copy credentials.
