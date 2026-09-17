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
