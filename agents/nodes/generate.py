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
most of it. This was observed directly during testing — the number of
reasoning tokens the model spends before committing to an answer varies
widely across questions (a few dozen to several hundred, even at
`reasoning_effort="low"`), so a cap sized for the typical case still
fails on harder questions, producing an empty completion that is
indistinguishable, downstream, from a genuine failure to answer.

Temperature is fixed at 0 for reproducibility, consistent with using an
unmodified, frozen model throughout.

Rate limiting
-------------
The free tier this project runs under enforces a per-minute token quota
well below what a full evaluation run needs in aggregate, so a sequence
of calls issued without pausing will periodically exceed it. `generate_answer`
retries on a 429 response, sleeping for the exact duration the API
reports until the quota window resets (via the `x-ratelimit-reset-*`
response headers) rather than a guessed backoff interval. Time spent
waiting on a rate limit is excluded from `latency_seconds`: it is an
artifact of the account's quota, not a property of the retrieval or
generation strategy being measured, and including it would distort the
cost comparison between systems.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
_REASONING_EFFORT = "low"
_MAX_TOKENS = 1024
_TEMPERATURE = 0

_MAX_RATE_LIMIT_RETRIES = 5
_RETRY_SAFETY_MARGIN_SECONDS = 0.5
_FALLBACK_RETRY_SECONDS = 5.0

# Matches the duration format used in Groq's x-ratelimit-reset-* headers,
# e.g. "592ms", "2.88s", "1m26.4s".
_DURATION_PATTERN = re.compile(r"^(?:(?P<minutes>\d+)m)?(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s)$")

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


def _parse_duration_seconds(text: str) -> float:
    match = _DURATION_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"unrecognized duration format: {text!r}")
    minutes = int(match.group("minutes") or 0)
    value = float(match.group("value"))
    if match.group("unit") == "ms":
        value /= 1000
    return minutes * 60 + value


def _seconds_until_retry(error: groq.RateLimitError) -> float:
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
    truncated: bool  # completion hit max_tokens before finishing — answer may be incomplete or empty


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
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                reasoning_effort=_REASONING_EFFORT,
            )
            latency = time.monotonic() - start
            break
        except groq.RateLimitError as error:
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
