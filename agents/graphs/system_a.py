"""
System A: vanilla single-shot vector RAG.

The baseline against which Systems B and C are measured. A single dense
retrieval over the question's passage corpus supplies the context for a
single generation call; there is no iteration, no query reformulation,
and no state machine, since none is needed for a fixed one-retrieve,
one-generate pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.nodes.generate import DEFAULT_MODEL, generate_answer
from data.load_2wiki import Example
from retrieval.faiss_index import PassageIndex

DEFAULT_TOP_K = 3


@dataclass
class SystemResult:
    qid: str
    predicted_answer: str
    llm_calls: int
    total_tokens: int
    latency_seconds: float
    truncated: bool


def run_system_a(
    example: Example,
    k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_MODEL,
) -> SystemResult:
    """Answer one question by retrieving top-k passages and generating once."""
    index = PassageIndex(example.passages)
    retrieved = index.search(example.question, k=k)
    context = [result.passage.text for result in retrieved]

    generation = generate_answer(example.question, context, model=model)

    return SystemResult(
        qid=example.qid,
        predicted_answer=generation.answer,
        llm_calls=1,  # retrieval here is embedding-only; Generate is the only LLM call
        total_tokens=generation.total_tokens,
        latency_seconds=generation.latency_seconds,
        truncated=generation.truncated,
    )


if __name__ == "__main__":
    from data.load_2wiki import load_2wiki

    examples = load_2wiki(n=5, seed=0)
    for example in examples:
        result = run_system_a(example)
        print(f"question: {example.question}")
        print(f"gold: {example.answer!r}  predicted: {result.predicted_answer!r}")
        print(
            f"llm_calls={result.llm_calls} total_tokens={result.total_tokens} "
            f"latency={result.latency_seconds:.2f}s"
        )
        print()
