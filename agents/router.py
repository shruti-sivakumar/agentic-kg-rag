"""
Non-LLM routing heuristics for Systems B and C.

Built as a heuristic rather than an LLM call, per the project's cost-
accounting design: if the router itself required a model call, every
hop would cost two calls instead of one, and the accuracy-cost
comparison between systems would be measuring routing overhead as much
as retrieval strategy.

Each router function folds in the Budget Check itself rather than
treating it as a separate graph node. A LangGraph conditional edge is
already "the check that runs before every routing decision" the
project's design calls for — composing `budget_exhausted` with the
routing heuristic achieves that directly, without an extra node that
would have no state of its own to transform.
"""

from __future__ import annotations

from typing import Literal

from agents.budget import budget_exhausted
from agents.state import AgentState

# Across the full dataset, roughly three-quarters of questions resolve
# via a gold reasoning path of exactly two triples (the remainder,
# mostly comparison-type questions, chain through more). Two retrieval
# rounds is therefore a reasonable "context probably looks sufficient"
# default for System B's router rather than an arbitrary constant — it
# is a fixed heuristic, not one adapted to a given question's apparent
# difficulty, and is exactly the kind of threshold Section 12 flags to
# revisit once real accuracy numbers are in.
TARGET_RETRIEVALS_B = 2


def route_system_b(state: AgentState) -> Literal["retrieve", "generate"]:
    """System B's two-way router: retrieve up to TARGET_RETRIEVALS_B times, then generate.

    Graph-Traverse is not a reachable action here — this is what makes
    System B the scientific control for isolating whether iteration
    alone (as opposed to the knowledge graph specifically) improves on
    System A.
    """
    if budget_exhausted(state):
        return "generate"
    if state["hop_count"] < TARGET_RETRIEVALS_B:
        return "retrieve"
    return "generate"


# Same rationale as TARGET_RETRIEVALS_B: two hops covers most gold
# reasoning paths in this dataset. System C counts any hop toward this
# total regardless of action (a Vector-Retrieve hop and a Graph-Traverse
# hop count the same), so this is a total-hops budget, not a per-action one.
TARGET_HOPS_C = 2


def route_system_c(state: AgentState) -> Literal["retrieve", "traverse", "generate"]:
    """System C's three-way router.

    Graph-Traverse is preferred over Vector-Retrieve whenever there is
    a frontier to walk from and the previous hop (if the previous hop
    was a traversal) made progress. `made_progress` is what prevents
    the router from retrying a frontier Graph-Traverse has already
    reported as a dead end: since neither the frontier nor
    `made_progress` changes on a dead-end hop, retrying it would only
    reproduce the same dead end, so once it goes false the router
    falls back to Vector-Retrieve for the remainder of the walk (or
    until Vector-Retrieve's own gathered context is judged sufficient).
    """
    if budget_exhausted(state):
        return "generate"
    if state["hop_count"] >= TARGET_HOPS_C:
        return "generate"
    if state["entity_frontier"] and state["made_progress"]:
        return "traverse"
    return "retrieve"
