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

A full run generates one LLM call per question against a rate-limited
external API and can run for the better part of an hour, so `run_harness`
supports checkpointing: given a `checkpoint_path`, it writes each row to
disk as soon as it is computed rather than only at the end, and skips any
question already present in that file on the next invocation. A run
interrupted by a rate limit, a network fault, or anything else can
therefore be resumed by rerunning with the same checkpoint path, without
losing completed work or re-spending quota on questions already answered.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from data.load_2wiki import Example
from eval.metrics import exact_match, f1, normalize_answer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_FIELDNAMES = [
    "qid",
    "system",
    "benchmark",
    "question_type",
    "gold_answer",
    "predicted_answer",
    "predicted_answer_normalized",
    "em",
    "f1",
    "llm_calls",
    "total_tokens",
    "latency_seconds",
    "truncated",
]


class SystemResult(Protocol):
    predicted_answer: str
    llm_calls: int
    total_tokens: int
    latency_seconds: float
    truncated: bool


SystemFn = Callable[[Example], SystemResult]


def _row_for(example: Example, result: SystemResult, system_name: str, benchmark: str) -> dict:
    return {
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


def run_harness(
    examples: list[Example],
    system_fn: SystemFn,
    system_name: str,
    benchmark: str = "2wikimultihopqa",
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run `system_fn` over `examples` and return one scored row per question.

    If `checkpoint_path` is given, questions already recorded there (by
    `qid`) are skipped rather than re-run, and each newly computed row is
    appended to the file immediately. This makes a run resumable: calling
    this function again with the same path and question set picks up
    where an interrupted run left off.
    """
    rows: list[dict] = []
    done_qids: set[str] = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        rows.extend(existing.to_dict(orient="records"))
        done_qids = set(existing["qid"])

    file_handle = None
    writer = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not checkpoint_path.exists()
        file_handle = open(checkpoint_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(file_handle, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
            file_handle.flush()

    n_skipped = len(done_qids)
    if verbose and n_skipped:
        print(f"Resuming: {n_skipped} questions already recorded in {checkpoint_path}")

    try:
        for i, example in enumerate(examples, start=1):
            if example.qid in done_qids:
                continue
            result = system_fn(example)
            row = _row_for(example, result, system_name, benchmark)
            rows.append(row)
            if writer is not None:
                writer.writerow(row)
                file_handle.flush()
            if verbose:
                print(f"[{i}/{len(examples)}] em={row['em']} f1={row['f1']:.2f} {example.qid}")
    finally:
        if file_handle is not None:
            file_handle.close()

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

    examples = load_2wiki(n=1000, seed=42)
    results = run_harness(
        examples,
        run_system_a,
        system_name="A",
        checkpoint_path=RESULTS_DIR / "system_a_full.csv",
        verbose=True,
    )

    print()
    print(f"Wrote {len(results)} rows to {RESULTS_DIR / 'system_a_full.csv'}")
    print()
    print(summarize(results))
    print()
    print(f"Overall EM: {results['em'].mean():.3f}  F1: {results['f1'].mean():.3f}")
    print(f"Truncated completions: {results['truncated'].sum()}/{len(results)}")
