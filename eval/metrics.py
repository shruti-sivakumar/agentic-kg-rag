"""
Exact-match and F1 scoring for short-answer question answering.

This is the standard SQuAD-style scoring procedure: both the prediction
and the gold answer are normalized before comparison (lowercased,
stripped of punctuation and articles, whitespace collapsed), exact match
is then a string equality test, and F1 is computed over the bag of
normalized tokens. Applying this normalization is worthwhile even when
the generation prompt already constrains the model to a short answer: a
correct answer that differs from gold only in casing, a trailing
article, or an incidental punctuation mark should not be scored as
wrong, and a run against real model output showed a case that
normalization alone accounts for — an answer differing from gold only
by a non-breaking space character, which the whitespace-collapsing step
below resolves without any special-casing, since Python's `str.split()`
treats any Unicode whitespace character as a separator.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLE_PATTERN.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    """1 if the normalized prediction equals the normalized gold answer, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    """Token-level bag-of-words F1 between the normalized prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
