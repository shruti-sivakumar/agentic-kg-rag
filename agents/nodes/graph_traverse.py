"""
Graph-Traverse node for System C.

This is the step that has to avoid the oracle trap in practice, not
just in how the graph was built (see data/kg_2wiki.py): given the
entities currently in the frontier, it must genuinely search among
their real outgoing edges and select one by scoring, never by looking
up the question's own gold triple.

Scoring works by verbalizing each candidate triple as a short sentence,
embedding it with the same encoder used for passage retrieval, and
comparing it against the question's embedding by cosine similarity —
selecting the highest-scoring candidate across all frontier entities'
edges combined, not per entity.

Threshold, chosen from measurement rather than guessed
--------------------------------------------------------
Checked against 138 questions in this project's evaluation sample
whose first gold hop has real branching (the seed entity has more than
one outgoing edge): the highest-scoring candidate is the correct one
87.7% of the time. Its score, however, does not reliably signal
whether that pick was right: the mean top score when the pick is
correct (0.731) is essentially the same as when it is wrong (0.744) —
a wrong pick is usually still a topically plausible one, not an
obviously bad one, so score magnitude cannot be used to second-guess
argmax's choice. What score magnitude does separate cleanly is the
presence of any plausible candidate at all: scoring a genuinely
unrelated question against unrelated triples gives a ceiling of 0.41,
well below the 0.51 minimum seen even among *correct* picks. The dead-
end threshold below is set at 0.45, in the gap between those two
figures, so it catches a frontier whose edges are all off-topic noise
without ever rejecting a plausible (right or wrong) candidate.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from agents.state import AgentState
from retrieval.embed import encode

DEAD_END_SIMILARITY_THRESHOLD = 0.45


def verbalize_triple(subject_name: str, relation: str, object_name: str) -> str:
    """Render a (subject, relation, object) triple as a short natural-language sentence.

    Template-based and cheap: no LLM call. The result only needs to be
    good enough for the shared sentence encoder to judge its relevance
    to the question, not to read fluently — 2Wiki's relation strings
    (`father`, `employer`, `country of citizenship`, ...) are already
    close enough to natural English for this generic template to work
    across all of them.
    """
    return f"{subject_name}'s {relation} is {object_name}."


def make_graph_traverse_node(graph: nx.MultiDiGraph):
    """Build a Graph-Traverse node bound to the dense knowledge graph.

    Like Vector-Retrieve's `PassageIndex`, the graph is built once
    (shared across every question in a run, unlike the per-question
    passage index) and closed over rather than threaded through state.
    """

    def graph_traverse_node(state: AgentState) -> dict:
        candidates = []  # (object_qid, verbalized_text)
        for entity in state["entity_frontier"]:
            if entity not in graph:
                continue
            for _subject, obj_qid, data in graph.out_edges(entity, data=True):
                text = verbalize_triple(graph.nodes[entity]["name"], data["relation"], graph.nodes[obj_qid]["name"])
                candidates.append((obj_qid, text))

        if not candidates:
            # No frontier entity has any outgoing edge at all.
            return {"hop_count": 1, "made_progress": False}

        texts = [text for _, text in candidates]
        candidate_embeddings = encode(texts)
        question_embedding = encode([state["question"]])[0]
        scores = candidate_embeddings @ question_embedding  # both L2-normalized: inner product = cosine similarity

        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score < DEAD_END_SIMILARITY_THRESHOLD:
            # Every candidate looks unrelated to the question — a real
            # search was performed, it just found nothing worth acting
            # on, so this is reported as a dead end rather than
            # committing to a low-confidence guess.
            return {"hop_count": 1, "made_progress": False}

        best_object_qid, best_text = candidates[best_index]
        return {
            "accumulated_context": [best_text],
            "entity_frontier": {best_object_qid},
            "hop_count": 1,
            "made_progress": True,
        }

    return graph_traverse_node
