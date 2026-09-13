"""
Generation step shared by Systems A, B, and C.

Every system answers through this same function, with the same prompt
template and the same frozen model, so that differences in the final
metrics are attributable to retrieval strategy rather than to generation.
This module makes no assumption about how its caller assembled the
context passed in: a single-shot retrieval (System A), an iteratively
reformulated one (System B), or a mix of retrieved passages and
verbalized graph edges (System C) are all just a list of strings here.

Model
-----
Generation runs against Groq's hosted `openai/gpt-oss-20b`, a
production-tier (non-preview) open-weight model. `reasoning_effort` is
set to its lowest level: the task is short-form factoid extraction from
supplied context, not open-ended reasoning, and a lower reasoning budget
reduces both latency and the token cost charged against the account's
per-minute quota without a loss of accuracy observed in initial testing.
Reasoning tokens are billed and reported separately from the visible
completion (`usage.completion_tokens_details.reasoning_tokens`) and are
never mixed into the returned answer text.

`max_tokens` is set well above what a short answer alone would need,
because reasoning tokens are drawn from the same completion budget: too
tight a cap truncates the visible answer once reasoning has consumed
most of it, which was observed directly when testing the larger sibling
model at a low cap.

Temperature is fixed at 0 for reproducibility, consistent with using an
unmodified, frozen model throughout.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
_REASONING_EFFORT = "low"
_MAX_TOKENS = 300
_TEMPERATURE = 0

_PROMPT_TEMPLATE = """Answer the question using only the information given in the context below.

Context:
{context}

Question: {question}

Give the shortest possible answer: a single name, date, or short phrase, and nothing else. Do not explain your reasoning or add any surrounding text."""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


def generate_answer(
    question: str,
    context: list[str],
    model: str = DEFAULT_MODEL,
) -> GenerationResult:
    """Produce a short-form answer from a question and its assembled context.

    `context` is a list of already-verbalized facts (retrieved passage
    text, or verbalized graph triples) in the order they should appear
    in the prompt; this function performs no retrieval or reformulation
    of its own.
    """
    prompt = _PROMPT_TEMPLATE.format(context="\n".join(context), question=question)

    start = time.monotonic()
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        reasoning_effort=_REASONING_EFFORT,
    )
    latency = time.monotonic() - start

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return GenerationResult(
        answer=answer,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_seconds=latency,
    )
