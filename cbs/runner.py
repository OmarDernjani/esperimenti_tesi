"""End-to-end CBS experiment driver.

Two-pass design:
  PASS 1: generate K completions per (entry x flag) via Ollama.
  PASS 2: build the global attribute pool from all completions, then run
          metamorphic bias@k on each completion.

After every entry the full results list is dumped to disk so a crashed run
can be inspected/resumed without losing progress."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from .metamorphic import build_global_pool, compute_biask
from .prompts import system_prompt

load_dotenv()

PKG_DIR = Path(__file__).parent

MODEL_TARGET     = os.getenv("MODEL_TARGET", "llama3.1:8b")
NUM_CTX          = int(os.getenv("NUM_CTX", 8192))
K_TIMES          = int(os.getenv("CBS_K_TIMES", 3))
N_PER_CATEGORY   = int(os.getenv("CBS_N_PER_CATEGORY", 5))
FLAGS            = os.getenv(
    "CBS_FLAGS", "Raw,Zero-shot,One-shot,Few-shot,CoT1,CoT2"
).split(",")
DATASET_PATH     = Path(os.getenv("CBS_DATASET", PKG_DIR / "dataset.json"))
OUTPUT_DIR       = Path(os.getenv("CBS_OUTPUT_DIR", "result"))
MAX_CONST_COMBOS = int(os.getenv("CBS_MAX_CONST_COMBINATIONS", 0)) or None
SEED             = int(os.getenv("CBS_SEED", 42))


_FENCED_PATTERNS = (
    re.compile(r"```python\s*\n?(.*?)```", re.DOTALL),
    re.compile(r"```py\s*\n?(.*?)```",     re.DOTALL),
    re.compile(r"```\s*\n?(.*?)```",       re.DOTALL),
)
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*)$", re.DOTALL)


def _extract_code(response: str) -> str:
    """Pull a Python block out of an LLM response.
    Fenced (longest match) -> unclosed-fence -> raw text."""
    for pat in _FENCED_PATTERNS:
        if (m := pat.findall(response)):
            return max(m, key=len).strip()
    if (m := _OPEN_FENCE.search(response)):
        return m.group(1).strip()
    return response.strip()


def _sample_balanced(data: list[dict], n_per_cat: int, rng: random.Random) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_cat[entry.get("category", "unknown")].append(entry)
    out: list[dict] = []
    for entries in by_cat.values():
        out.extend(rng.sample(entries, min(n_per_cat, len(entries))))
    return out


def _build_chain(flag: str):
    template = ChatPromptTemplate([
        ("system", system_prompt(flag)),
        ("human", "{task}"),
    ])
    llm = ChatOllama(model=MODEL_TARGET, num_ctx=NUM_CTX, num_keep=0)
    return template | llm | StrOutputParser()


def _generate(flag: str, task: str) -> str:
    return _extract_code(_build_chain(flag).invoke({"task": task}))


def _dump(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _pass1_generate(sampled: list[dict], out_path: Path) -> list[dict]:
    results: list[dict] = []
    for entry in tqdm(sampled, desc="pass1 generate"):
        for flag in FLAGS:
            for k in range(K_TIMES):
                try:
                    code = _generate(flag, entry["prompt"])
                except Exception as e:
                    print(f"[gen-err] {flag} k={k}: {e}", file=sys.stderr)
                    code = ""
                results.append({
                    "prompt":     entry["prompt"],
                    "category":   entry["category"],
                    "flag":       flag,
                    "k":          k,
                    "completion": code,
                    "model":      MODEL_TARGET,
                })
                _dump(results, out_path)
    return results


def _pass2_biask(results: list[dict], out_path: Path, rng: random.Random) -> list[dict]:
    codes = [r["completion"] for r in results if r.get("completion")]
    pool = build_global_pool(codes)
    print(f"[pass2] global pool keys: {list(pool)[:20]}{' ...' if len(pool) > 20 else ''}")
    for r in tqdm(results, desc="pass2 bias@k"):
        r.update(compute_biask(r["completion"], pool, MAX_CONST_COMBOS, rng))
        _dump(results, out_path)
    return results


def run_experiment() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"cbs_results_{datetime.now():%Y%m%d_%H%M%S}.json"

    rng = random.Random(SEED)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    sampled = _sample_balanced(dataset, N_PER_CATEGORY, rng)

    print(f"[cbs] model={MODEL_TARGET} | flags={FLAGS} | k={K_TIMES} | "
          f"n_per_cat={N_PER_CATEGORY} | entries={len(sampled)}")
    print(f"[cbs] dataset={DATASET_PATH}  output={out_path}")

    results = _pass1_generate(sampled, out_path)
    results = _pass2_biask(results, out_path, rng)
    print(f"[cbs] done: {len(results)} items -> {out_path}")
    return out_path


if __name__ == "__main__":
    run_experiment()
