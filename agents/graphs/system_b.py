"""
System B: multi-step vector RAG, no knowledge-graph access.

The scientific control between System A and System C: it can iterate
(unlike A), but only by searching text again with a reformulated query
(unlike C, which can also walk the knowledge graph). Any accuracy gain
System C shows over System B is therefore attributable to the graph
specifically, not merely to being allowed more than one retrieval step.

The graph has two nodes, `retrieve` and `generate`, and one router
(`route_system_b`) that also carries the Budget Check: every edge out
of `retrieve`, and the entry edge from `START`, is routed through it,
so the budget is re-evaluated before each decision rather than only at
the start of the walk.
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from agents.nodes.generate import generate_node
from agents.nodes.vector_retrieve import make_vector_retrieve_node
from agents.router import route_system_b
from agents.state import AgentState, initial_state
from data.load_2wiki import Example
from retrieval.faiss_index import PassageIndex

# max_hops sits above route_system_b's own TARGET_RETRIEVALS_B (2), so
# under normal operation the router's heuristic reaches Generate on its
# own and this budget acts only as a backstop. max_calls is nominal in
# the current design: Generate is the only node that ever spends a
# call, and it runs at most once per walk, so no value above 1 changes
# behavior — it is kept as a named, non-trivial budget parameter for
# consistency with System C, where a future design might spend more
# than one call per walk.
DEFAULT_MAX_HOPS = 4
DEFAULT_MAX_CALLS = 3


@dataclass
class SystemResult:
    qid: str
    predicted_answer: str
    llm_calls: int
    total_tokens: int
    latency_seconds: float
    truncated: bool


def build_system_b_graph(passage_index: PassageIndex):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", make_vector_retrieve_node(passage_index))
    graph.add_node("generate", generate_node)

    destinations = {"retrieve": "retrieve", "generate": "generate"}
    graph.add_conditional_edges(START, route_system_b, destinations)
    graph.add_conditional_edges("retrieve", route_system_b, destinations)
    graph.add_edge("generate", END)

    return graph.compile()


def run_system_b(
    example: Example,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_calls: int = DEFAULT_MAX_CALLS,
) -> SystemResult:
    """Answer one question by iterating Vector-Retrieve under System B's router."""
    index = PassageIndex(example.passages)
    compiled = build_system_b_graph(index)

    state = initial_state(example.question, max_hops=max_hops, max_calls=max_calls)
    result = compiled.invoke(state)

    return SystemResult(
        qid=example.qid,
        predicted_answer=result["answer"],
        llm_calls=result["call_count"],
        total_tokens=result["total_tokens"],
        latency_seconds=result["total_latency_seconds"],
        truncated=result["truncated"],
    )


if __name__ == "__main__":
    from data.load_2wiki import load_2wiki

    examples = load_2wiki(n=5, seed=0)
    for example in examples:
        result = run_system_b(example)
        print(f"question: {example.question}")
        print(f"gold: {example.answer!r}  predicted: {result.predicted_answer!r}")
        print(
            f"llm_calls={result.llm_calls} total_tokens={result.total_tokens} "
            f"latency={result.latency_seconds:.2f}s"
        )
        print()
