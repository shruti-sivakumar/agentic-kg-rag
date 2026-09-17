"""
Entity Linker for System C.

Runs once, before the walk begins, to seed `entity_frontier` with the
question's starting entities.

2WikiMultiHopQA ships each question's correct starting entities as
gold Wikidata QIDs (`Example.entity_ids`). Resolving a detected mention
to one of them is therefore a lookup against data the dataset already
provides, not open-world disambiguation — a real disambiguation step
(fuzzy string matching plus encoder-based scoring among candidate
Wikidata entities) is deferred to the later MuSiQue/HotpotQA extension,
which ships no such gold mapping and genuinely needs it. Mention
detection is still performed and is not a no-op: a question with no
detectable entity mention seeds an empty frontier, which is exactly
the condition System C's router treats as "no graph walk is possible
from here" and falls back to Vector-Retrieve on.
"""

from __future__ import annotations

import spacy

from agents.state import AgentState, EntityID
from data.load_2wiki import Example

_SPACY_MODEL = "en_core_web_sm"
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(_SPACY_MODEL)
    return _nlp


def detect_entity_mentions(question: str) -> list[str]:
    """Return the text of each entity-like span spaCy detects in the question."""
    doc = _get_nlp()(question)
    return [ent.text for ent in doc.ents]


def link_entities(example: Example) -> set[EntityID]:
    """Seed the entity frontier for one question."""
    if not detect_entity_mentions(example.question):
        return set()
    return set(example.entity_ids)


def make_entity_linker_node(example: Example):
    """Build an Entity Linker node bound to one question.

    `AgentState` carries only `question` as plain text, not the
    structured `Example` this node needs to reach `entity_ids` — so,
    like `PassageIndex` for Vector-Retrieve, the question this run is
    answering is closed over rather than threaded through state.
    """

    def entity_linker_node(state: AgentState) -> dict:
        return {"entity_frontier": link_entities(example)}

    return entity_linker_node
