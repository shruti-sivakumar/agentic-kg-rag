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
Generation runs against `gpt-4o-mini`. Unlike a reasoning-tuned model,
it returns its answer directly with no hidden reasoning tokens competing
for the completion budget, which keeps the prompt-to-completion token
accounting straightforward and removes any risk of the visible answer
being truncated before it is written.

Temperature is fixed at 0 for reproducibility, consistent with using an
unmodified, frozen model throughout.

Rate limiting
-------------
`generate_answer` retries on a 429 response, sleeping for the exact
duration the API reports until the relevant quota window resets (via the
`x-ratelimit-reset-*` response headers) rather than a guessed backoff
interval. Time spent waiting on a rate limit is excluded from
`latency_seconds`: it is an artifact of the account's quota, not a
property of the retrieval or generation strategy being measured, and
including it would distort the cost comparison between systems.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import openai
from dotenv import load_dotenv
from openai import OpenAI

from agents.state import AgentState

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
_MAX_COMPLETION_TOKENS = 300
_TEMPERATURE = 0

_MAX_RATE_LIMIT_RETRIES = 5
_RETRY_SAFETY_MARGIN_SECONDS = 0.5
_FALLBACK_RETRY_SECONDS = 5.0

# Matches the duration format used in the API's x-ratelimit-reset-* headers,
# e.g. "1ms", "8.64s", "1m26.4s".
_DURATION_PATTERN = re.compile(r"^(?:(?P<minutes>\d+)m)?(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)$")

_PROMPT_TEMPLATE = """Answer the question using only the information given in the context below.

Context:
{context}

Question: {question}

Give the shortest possible answer: a single name, date, or short phrase, and nothing else. Do not explain your reasoning or add any surrounding text."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _parse_duration_seconds(text: str) -> float:
    match = _DURATION_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"unrecognized duration format: {text!r}")
    minutes = int(match.group("minutes") or 0)
    value = float(match.group("value"))
    if match.group("unit") == "ms":
        value /= 1000
    return minutes * 60 + value


def _seconds_until_retry(error: openai.RateLimitError) -> float:
    """Read how long to wait before retrying from the response's reset headers.

    Falls back to a fixed interval if the headers are absent or in an
    unrecognized format, rather than failing the call outright.
    """
    headers = getattr(error.response, "headers", {})
    for header in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = headers.get(header)
        if value:
            try:
                return _parse_duration_seconds(value) + _RETRY_SAFETY_MARGIN_SECONDS
            except ValueError:
                continue
    return _FALLBACK_RETRY_SECONDS


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    truncated: bool  # completion hit the token cap before finishing


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

    attempt = 0
    while True:
        try:
            start = time.monotonic()
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                temperature=_TEMPERATURE,
            )
            latency = time.monotonic() - start
            break
        except openai.RateLimitError as error:
            attempt += 1
            if attempt > _MAX_RATE_LIMIT_RETRIES:
                raise
            time.sleep(_seconds_until_retry(error))

    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    usage = response.usage
    return GenerationResult(
        answer=answer,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_seconds=latency,
        truncated=choice.finish_reason == "length",
    )


def generate_node(state: AgentState) -> dict:
    """LangGraph node wrapping `generate_answer` for Systems B and C.

    `answer` overwrites (Generate runs at most once per walk in the
    current design); `call_count`, `total_tokens`, and
    `total_latency_seconds` are increments merged by the state schema's
    reducers, so they accumulate correctly alongside whatever Vector-
    Retrieve or Graph-Traverse already contributed. `truncated` is
    merged by logical OR, so one truncated call marks the whole walk.
    """
    result = generate_answer(state["question"], state["accumulated_context"])
    return {
        "answer": result.answer,
        "call_count": 1,
        "total_tokens": result.total_tokens,
        "total_latency_seconds": result.latency_seconds,
        "truncated": result.truncated,
    }
