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

This is a documented simplification, not a claim that entity
resolution is a solved non-problem: it is set aside for this phase
because the starting entity is always a name already present in the
question text (a proper noun the question hands the system directly),
never a hidden intermediate entity the walk would need to have
discovered. Resolving it is therefore a different problem from the
multi-hop reasoning under test, and leaks no part of the reasoning
path itself. That the reasoning path is not being leaked elsewhere is
not asserted on this basis alone — it is the property that
`graph_traverse.py`'s edge scoring is measured against directly (see
its module docstring): every hop after the starting entity is selected
with no gold signal, correct 87.7% of the time on questions with a
genuine choice among candidates. That measurement, not this module's
shortcut, is the evidence against the traversal step secretly reading
the answer key.

Taking the starting entity as given is also the prevailing convention
in the KGQA literature, not a choice specific to this project: MetaQA
(Zhang et al., "Variational Reasoning for Question Answering with
Knowledge Graph," AAAI 2018) defines each question as a head entity
plus a reasoning path plus an answer, with the head entity given as
input rather than resolved by the system. DoG (Li et al., "Decoding
on Graphs: Faithful and Sound Reasoning on Knowledge Graphs," 2024)
evaluates on WebQSP, CWQ, and this project's own dataset,
2WikiMultiHopQA, initializing its subgraph from a given query entity
and using ground-truth reasoning paths for evaluation only — available
for 2WikiMultiHopQA in its setup but not for WebQSP or CWQ, the same
role gold paths play here. ProgRAG (Park et al., "ProgRAG: Hallucination-Resistant
Progressive Retrieval and Reasoning over Knowledge Graphs," 2025)
likewise begins each question's retrieval from a given source entity.
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
