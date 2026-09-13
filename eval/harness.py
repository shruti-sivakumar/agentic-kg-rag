"""
Evaluation harness: runs a system over a set of questions and logs
per-question accuracy and cost.

The harness is deliberately decoupled from any particular system's
implementation. It only requires that a system be a callable taking one
`Example` and returning an object exposing `predicted_answer`,
`llm_calls`, `total_tokens`, and `latency_seconds` — the structural
interface below, which `agents.graphs.system_a.SystemResult` already
satisfies and which Systems B and C will satisfy in turn without the
harness needing to change.

What is logged, per (question, system) row, follows directly from the
evaluation design: the raw and normalized prediction, exact match and
F1 against the gold answer, the cost figures needed to characterize the
accuracy-cost tradeoff (LLM calls, tokens, wall-clock latency), and the
benchmark and question-type labels needed to break results down by
question type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from data.load_2wiki import Example
from eval.metrics import exact_match, f1, normalize_answer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class SystemResult(Protocol):
    predicted_answer: str
    llm_calls: int
    total_tokens: int
    latency_seconds: float
    truncated: bool


SystemFn = Callable[[Example], SystemResult]


def run_harness(
    examples: list[Example],
    system_fn: SystemFn,
    system_name: str,
    benchmark: str = "2wikimultihopqa",
) -> pd.DataFrame:
    """Run `system_fn` over `examples` and return one scored row per question."""
    rows = []
    for example in examples:
        result = system_fn(example)
        rows.append(
            {
                "qid": example.qid,
                "system": system_name,
                "benchmark": benchmark,
                "question_type": example.question_type,
                "gold_answer": example.answer,
                "predicted_answer": result.predicted_answer,
                "predicted_answer_normalized": normalize_answer(result.predicted_answer),
                "em": exact_match(result.predicted_answer, example.answer),
                "f1": f1(result.predicted_answer, example.answer),
                "llm_calls": result.llm_calls,
                "total_tokens": result.total_tokens,
                "latency_seconds": result.latency_seconds,
                "truncated": result.truncated,
            }
        )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, filename: str) -> Path:
    """Write a results DataFrame to results/<filename>, creating the directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean EM/F1 and cost figures per system and question type."""
    return df.groupby(["system", "question_type"]).agg(
        n=("qid", "count"),
        em=("em", "mean"),
        f1=("f1", "mean"),
        llm_calls=("llm_calls", "mean"),
        total_tokens=("total_tokens", "mean"),
        latency_seconds=("latency_seconds", "mean"),
    )


if __name__ == "__main__":
    from data.load_2wiki import load_2wiki
    from agents.graphs.system_a import run_system_a

    # A modest sample: enough to validate the harness against real model
    # output without drawing meaningfully on the account's daily call
    # budget, which the full evaluation run will need in full.
    examples = load_2wiki(n=30, seed=0)
    results = run_harness(examples, run_system_a, system_name="A")

    path = save_results(results, "system_a_smoke.csv")
    print(f"Wrote {len(results)} rows to {path}")
    print()
    print(summarize(results))
    print()
    print(f"Overall EM: {results['em'].mean():.3f}  F1: {results['f1'].mean():.3f}")
    print(f"Truncated completions: {results['truncated'].sum()}/{len(results)}")
