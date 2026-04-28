"""CBS bias-testing experiment package.

Implements the framework from Huang et al. 2024
("Bias Testing and Mitigation in LLM-based Code Generation"):
 - generate code from prompts via 6 prompt-engineering strategies
 - test each generated function for bias via metamorphic execution
   (Definition 1, §3.3)
 - compute the 3 metrics from §3.4: CBS, CBS_U@K, CBS_I@K

Public API:
    run_experiment()        end-to-end driver
    evaluate(path)          compute metrics on a results JSON
    SYSTEM_PROMPTS          dict[flag] -> system prompt
    PROTECTED               protected-attribute keywords (paper Tab. 1 union)
"""

from .metrics import evaluate
from .prompts import SYSTEM_PROMPTS, system_prompt
from .runner import run_experiment
from .metamorphic import PROTECTED

__all__ = [
    "run_experiment",
    "evaluate",
    "SYSTEM_PROMPTS",
    "system_prompt",
    "PROTECTED",
]
