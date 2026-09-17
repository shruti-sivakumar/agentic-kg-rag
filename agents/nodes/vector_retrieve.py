"""
Vector-Retrieve node for Systems B and C.

Unlike System A, which retrieves once against the raw question, this
node may run several times over the course of a walk, each time
against a reformulated query built from what has been gathered so far.
It is built as a factory (`make_vector_retrieve_node`) rather than a
bare function because it needs a `PassageIndex` bound to one question's
ten-passage corpus, and that index is constructed once per question
before the graph runs — the same index System A builds — rather than
carried through the shared state schema, which otherwise has no reason
to know about passages at all.
"""

from __future__ import annotations

from agents.state import AgentState
from retrieval.faiss_index import PassageIndex

DEFAULT_TOP_K = 3


def reformulate_query(state: AgentState) -> str:
    """Build the search query for the next retrieval round.

    The reformulation is a plain heuristic — the original question with
    everything retrieved so far appended — rather than an LLM call.
    This is a deliberate choice, not an oversight: if reformulation
    itself spent an LLM call, System B's cost would no longer be
    comparable to System A's on Generate calls alone, and the
    comparison the two systems exist to support (does iteration alone
    help, independent of generation cost) would be confounded by
    iteration itself becoming more expensive to run.
    """
    if not state["accumulated_context"]:
        return state["question"]
    return state["question"] + " " + " ".join(state["accumulated_context"])


def make_vector_retrieve_node(passage_index: PassageIndex, k: int = DEFAULT_TOP_K):
    """Build a Vector-Retrieve node bound to one question's passage index."""

    def vector_retrieve_node(state: AgentState) -> dict:
        query = reformulate_query(state)
        already_seen = set(state["accumulated_context"])

        # Search the whole corpus rather than just the top k, then drop
        # passages already gathered on an earlier hop. Reformulating the
        # query by appending prior context tends to keep the same
        # top-ranked passages near the top again on a corpus this small
        # (ten passages), so without this filtering step, a second hop
        # would often just re-retrieve the first hop's results and
        # contribute nothing new — defeating the purpose of iterating.
        candidates = passage_index.search(query, k=len(passage_index))
        new_results = [r for r in candidates if r.passage.text not in already_seen][:k]

        return {
            "accumulated_context": [r.passage.text for r in new_results],
            "hop_count": 1,
        }

    return vector_retrieve_node
