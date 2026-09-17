"""
Shared state schema for Systems B and C.

System A needs no state machine (Section 8: one retrieve, one generate,
done), but B and C both iterate under a shared hop/call budget, choosing
at each step among retrieval, graph traversal, and answering. Defining
one state schema for both keeps the two systems' cost accounting
directly comparable: the same fields are incremented the same way
regardless of which actions a given system's router can reach.

`accumulated_context` and the running counters are declared with an
`operator.add` reducer rather than left to default (overwrite)
semantics, since every node that touches them is contributing to a
running total across the walk, not replacing it — a node returns only
the increment or the newly gathered facts, and LangGraph merges them
into the running state. `entity_frontier`, by contrast, is meant to be
replaced outright on each graph-traversal hop (the frontier moves to
the object entity of the selected triple; it does not accumulate), so
it is left with default overwrite semantics.

Cost accounting (`total_tokens`, `total_latency_seconds`, `truncated`)
is part of this state for the same reason `hop_count` and `call_count`
are: the accuracy-cost comparison between Systems A, B, and C requires
these figures to be tracked identically regardless of which actions a
given system's router can reach, and a value computed inside a node
(e.g. the token count of one generation call) is only usable by the
evaluation harness if it survives to the end of the walk. Threading it
through the state's own reducers keeps it consistent with how every
other running total in the walk is accumulated, rather than requiring
a separate mechanism outside the graph.

`made_progress` exists for System C's Graph-Traverse step: it reports
whether the most recent hop actually found something (a real outgoing
edge scoring above the dead-end threshold), so that System C's router
can tell a frontier worth continuing to traverse apart from one that
has gone stale, without needing to re-inspect the graph itself. It
reflects only the outcome of the most recent hop, not a running total,
so it uses default overwrite semantics rather than a reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

EntityID = str  # a Wikidata QID, e.g. "Q42"


class AgentState(TypedDict):
    question: str
    accumulated_context: Annotated[list[str], operator.add]
    entity_frontier: set[EntityID]
    hop_count: Annotated[int, operator.add]
    call_count: Annotated[int, operator.add]
    max_hops: int
    max_calls: int
    answer: str  # set by Generate; empty until the run reaches it
    total_tokens: Annotated[int, operator.add]
    total_latency_seconds: Annotated[float, operator.add]
    truncated: Annotated[bool, operator.or_]  # any generation call hit its token cap
    made_progress: bool  # did the most recent hop find a real, above-threshold candidate?


def initial_state(
    question: str,
    entity_frontier: set[EntityID] | None = None,
    max_hops: int = 5,
    max_calls: int = 5,
) -> AgentState:
    """Construct the starting state for one question.

    `entity_frontier` defaults to empty rather than requiring every
    caller to pass one explicitly: System B never populates it (it has
    no graph-traversal action to seed), and System C's entity linker
    fills it in as a separate step rather than at state construction.
    """
    return AgentState(
        question=question,
        accumulated_context=[],
        entity_frontier=entity_frontier or set(),
        hop_count=0,
        call_count=0,
        max_hops=max_hops,
        max_calls=max_calls,
        answer="",
        total_tokens=0,
        total_latency_seconds=0.0,
        truncated=False,
        made_progress=True,
    )
