"""
Loading and sampling for 2WikiMultiHopQA (Ho et al., 2020).

This module handles corpus acquisition only: downloading the dataset,
parsing it into typed records, sampling a fixed-size evaluation set, and
constructing the per-question retrieval corpus. Embedding, indexing, and
knowledge-graph construction are handled by separate modules.

Source and data variant
------------------------
The dataset is distributed by the authors (github.com/Alab-NII/2wikimultihop)
in two archive variants. This module uses the "data_ids" variant, which
augments the base release with Wikidata QIDs (`entity_ids`, `evidences_id`,
`answer_id`) alongside the surface-form question, answer, and evidence
fields. The QIDs are required here because entity linking (Section 7.1 of
the project specification) resolves question mentions against gold
Wikidata entities rather than performing open-world disambiguation, and
because knowledge-graph construction keys nodes by QID rather than by
surface string, which is the correct choice given that distinct entities
can share a surface form.

Only the validation split (`dev.json`) and the entity-alias table
(`id_aliases.json`) are extracted from the archive. The training split is
not required by the current evaluation design, and the test split ships
without gold answers or evidence, making it unusable for offline scoring.

Record schema
-------------
Each question in `dev.json` carries:

    _id              unique question identifier
    type             one of: comparison, inference, compositional,
                     bridge_comparison
    question         question text
    answer           gold answer, surface form
    answer_id        gold answer's Wikidata QID, where the answer denotes
                     a Wikidata entity. Absent when the answer is a literal
                     (a date, a quantity, etc.) rather than an entity.
    context          ten paragraphs per question, each a (title, sentences)
                     pair, comprising the gold supporting passages together
                     with distractor passages; this is the retrieval corpus
                     for that question.
    supporting_facts (title, sentence index) pairs identifying the gold
                     evidence sentences within `context`.
    evidences        the gold reasoning path, as (subject, relation,
                     object) triples in surface form. Path length varies
                     with question type and is not fixed at two hops.
    evidences_id     the same reasoning path with subject and object given
                     as Wikidata QIDs where the triple resolves to entities.
                     For comparison-type questions whose compared attribute
                     is a literal (e.g. a publication date), no triple in
                     the path resolves to an entity pair, and the field is
                     reported as an empty list rather than a partial one;
                     code that consumes `evidences_id` must therefore not
                     assume it is the same length as `evidences`.
    entity_ids       the question's seed Wikidata entities, given as a
                     single underscore-joined string (e.g.
                     "Q3569599_Q3570840") rather than a list.

`id_aliases.json` is newline-delimited JSON, distinct from the array
format of `dev.json`; each line maps a Wikidata QID to its known aliases
and demonyms. It is not consumed by the current evaluation pipeline but is
retained here for alias-aware answer scoring in later work.

Note on graph construction
---------------------------
`evidences_id` gives the gold reasoning path for a single question and
must not be treated as the traversal target: a knowledge-graph traversal
node that looked up this path directly would be answering from the
reference solution rather than searching for it. The dense graph used for
traversal is built separately, from `evidences_id` aggregated across the
full sampled set, so that any one question's frontier entity generally has
several candidate outgoing edges and the correct one must be selected by
scoring, not retrieved by identity.
"""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

_DATA_IDS_ZIP_URL = "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1"
_ZIP_MEMBER_DEV = "data_ids/dev.json"
_ZIP_MEMBER_ID_ALIASES = "data_ids/id_aliases.json"

RAW_DIR = Path(__file__).resolve().parent / "raw"
_CACHE_ZIP = RAW_DIR / "data_ids.zip"
_DEV_JSON = RAW_DIR / "dev.json"
_ID_ALIASES_JSON = RAW_DIR / "id_aliases.json"

VALID_QUESTION_TYPES = {"comparison", "inference", "compositional", "bridge_comparison"}


@dataclass
class Passage:
    """A single context paragraph belonging to one question's retrieval corpus."""

    title: str
    text: str
    is_supporting: bool  # whether this paragraph is part of the gold evidence


@dataclass
class Example:
    """A single 2WikiMultiHopQA question, prepared for retrieval and generation."""

    qid: str
    question: str
    answer: str
    answer_id: str | None
    question_type: str
    entity_ids: list[str]
    evidences: list[list[str]]
    evidences_id: list[list[str]]
    supporting_facts: list[list]
    passages: list[Passage]


def _download_raw(force: bool = False) -> None:
    """Fetch and cache dev.json and id_aliases.json from the source archive.

    Downloaded files are written to data/raw/, which is not version
    controlled: the files are large and are deterministically re-derivable
    from the source URL.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _DEV_JSON.exists() and _ID_ALIASES_JSON.exists() and not force:
        return

    if not _CACHE_ZIP.exists() or force:
        urllib.request.urlretrieve(_DATA_IDS_ZIP_URL, _CACHE_ZIP)

    with zipfile.ZipFile(_CACHE_ZIP) as zf:
        for member, target in (
            (_ZIP_MEMBER_DEV, _DEV_JSON),
            (_ZIP_MEMBER_ID_ALIASES, _ID_ALIASES_JSON),
        ):
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    # The archive itself is redundant once its members are extracted, and
    # is large enough to be worth discarding on memory-constrained hosts.
    _CACHE_ZIP.unlink(missing_ok=True)


def _build_passages(record: dict) -> list[Passage]:
    supporting_titles = {title for title, _sent_id in record["supporting_facts"]}
    return [
        Passage(
            title=title,
            text=" ".join(sentence.strip() for sentence in sentences),
            is_supporting=title in supporting_titles,
        )
        for title, sentences in record["context"]
    ]


def _parse_record(record: dict) -> Example:
    if record["type"] not in VALID_QUESTION_TYPES:
        raise ValueError(f"unrecognized question type: {record['type']!r}")
    entity_ids = record["entity_ids"].split("_") if record["entity_ids"] else []
    return Example(
        qid=record["_id"],
        question=record["question"],
        answer=record["answer"],
        answer_id=record.get("answer_id") or None,
        question_type=record["type"],
        entity_ids=entity_ids,
        evidences=record["evidences"],
        evidences_id=record["evidences_id"],
        supporting_facts=record["supporting_facts"],
        passages=_build_passages(record),
    )


def load_2wiki(
    split: str = "dev",
    n: int | None = 1000,
    seed: int = 42,
    force_redownload: bool = False,
) -> list[Example]:
    """Load a sample of 2WikiMultiHopQA, downloading the source data if needed.

    Args:
        split: dataset split to load. Only "dev" (the validation split) is
            supported, matching the evaluation design for this phase.
        n: number of questions to sample. If None, the full split is
            returned. Sampling is a plain reproducible random draw
            (`random.Random(seed).sample`) rather than one stratified by
            question type; if the resulting type distribution proves too
            skewed for a meaningful per-type breakdown, stratified sampling
            is the natural refinement.
        seed: seed for the sampling RNG.
        force_redownload: re-download the source archive even if a local
            cache already exists.

    Returns:
        A list of Example, one per sampled question.
    """
    if split != "dev":
        raise ValueError(f"unsupported split: {split!r}; only 'dev' is available")

    _download_raw(force=force_redownload)

    with open(_DEV_JSON, encoding="utf-8") as f:
        raw_records = json.load(f)

    if n is not None and n < len(raw_records):
        raw_records = random.Random(seed).sample(raw_records, n)

    return [_parse_record(r) for r in raw_records]


def load_id_aliases() -> dict[str, dict]:
    """Load the Wikidata alias table, keyed by QID.

    Each entry maps a QID to its known aliases and demonyms, as shipped
    alongside the dataset. Not consumed by the current evaluation
    pipeline; retained for alias-aware answer scoring in later work.
    """
    _download_raw()
    aliases: dict[str, dict] = {}
    with open(_ID_ALIASES_JSON, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            aliases[record["Q_id"]] = {
                "aliases": record["aliases"],
                "demonyms": record["demonyms"],
            }
    return aliases


if __name__ == "__main__":
    from collections import Counter

    examples = load_2wiki(n=1000, seed=42)
    print(f"Loaded {len(examples)} examples")
    print("Question type distribution:", Counter(e.question_type for e in examples))

    missing_answer_id = sum(1 for e in examples if e.answer_id is None)
    print(f"Examples with no answer_id (literal answers): {missing_answer_id}/{len(examples)}")

    empty_evidences_id = sum(1 for e in examples if e.evidences and not e.evidences_id)
    print(f"Examples with a reasoning path but no resolvable entity IDs: {empty_evidences_id}/{len(examples)}")

    passage_counts = {len(e.passages) for e in examples}
    print(f"Distinct passage-count values across the sample: {passage_counts}")

    example = examples[0]
    print("\nExample record:")
    print(f"  qid: {example.qid}")
    print(f"  type: {example.question_type}")
    print(f"  question: {example.question}")
    print(f"  answer: {example.answer!r} (answer_id={example.answer_id!r})")
    print(f"  entity_ids: {example.entity_ids}")
    print(f"  evidences: {example.evidences}")
    print(f"  passages: {len(example.passages)}")
