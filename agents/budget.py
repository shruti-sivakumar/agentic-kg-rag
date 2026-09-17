"""
Budget Check: the conditional edge evaluated before every routing
decision in Systems B and C.

Placing this check ahead of the router, rather than folding the budget
condition into the router's own logic, keeps the router's heuristic
focused purely on which action best serves the question, and keeps the
budget enforcement point singular and easy to audit: exactly one place
in the graph decides whether the walk continues.
"""

from __future__ import annotations

from typing import Literal

from agents.state import AgentState


def budget_exhausted(state: AgentState) -> bool:
    """True once the walk has used its allotted hops or LLM calls."""
    return state["hop_count"] >= state["max_hops"] or state["call_count"] >= state["max_calls"]


def route_after_budget_check(state: AgentState) -> Literal["generate", "router"]:
    """LangGraph conditional-edge function.

    A budget-exhausted state routes straight to Generate rather than
    skipping generation outright: every question must still receive a
    scored prediction, produced from whatever context has been gathered
    so far, rather than being left without one because the walk ran out
    of budget before reaching an answer on its own.
    """
    return "generate" if budget_exhausted(state) else "router"


if __name__ == "__main__":
    # The router this check hands off to belongs to System B (Section
    # 7.3), not yet built. This demonstrates only the forced-stop path:
    # a state whose budget is already spent routes straight to Generate
    # and still produces a scored answer, which is what Section 7.2
    # requires and the only part of the mechanism this module owns.
    from langgraph.graph import END, START, StateGraph

    from agents.nodes.generate import generate_node
    from agents.state import initial_state

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate_node)
    graph.add_conditional_edges(START, route_after_budget_check, {"generate": "generate", "router": END})
    graph.add_edge("generate", END)
    compiled = graph.compile()

    state = initial_state("What is the capital of France?", max_hops=0, max_calls=0)
    print("budget_exhausted:", budget_exhausted(state))

    result = compiled.invoke(state)
    print("answer:", result["answer"])
    print("call_count:", result["call_count"])
    print("total_tokens:", result["total_tokens"])
    print("truncated:", result["truncated"])
