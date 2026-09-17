"""
System C: the full agentic system.

All three actions the project compares are reachable here: Vector-
Retrieve, Graph-Traverse, and Generate, chosen between at each step by
`route_system_c`. Whatever advantage this system shows over System B
is attributable to the knowledge graph specifically, since B has every
other piece of this design (the shared state, the budget, the same
Generate step) except Graph-Traverse itself.

The dense knowledge graph (see data/kg_2wiki.py) is expensive to build
and shared across every question in a run, unlike a question's
`PassageIndex`, so it is built once by the caller and passed in rather
than constructed inside `run_system_c`.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from langgraph.graph import END, START, StateGraph

from agents.nodes.entity_linker import make_entity_linker_node
from agents.nodes.generate import generate_node
from agents.nodes.graph_traverse import make_graph_traverse_node
from agents.nodes.vector_retrieve import make_vector_retrieve_node
from agents.router import route_system_c
from agents.state import AgentState, initial_state
from data.load_2wiki import Example
from retrieval.faiss_index import PassageIndex

# See agents/router.py for why these are set to 4 and 3: the router's
# own TARGET_HOPS_C (2) reaches Generate on its own under normal
# operation, so max_hops acts as a backstop, and max_calls is nominal
# since Generate is the only call-spending node and runs at most once.
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


def build_system_c_graph(passage_index: PassageIndex, knowledge_graph: nx.MultiDiGraph, example: Example):
    graph = StateGraph(AgentState)
    graph.add_node("link_entities", make_entity_linker_node(example))
    graph.add_node("retrieve", make_vector_retrieve_node(passage_index))
    graph.add_node("traverse", make_graph_traverse_node(knowledge_graph))
    graph.add_node("generate", generate_node)

    # Entity linking spends no hop and no call, and hop_count/call_count
    # are both zero before it runs, so the budget can never already be
    # exhausted at this point — an unconditional edge into it is
    # simpler than a conditional one that could only ever take one path.
    graph.add_edge(START, "link_entities")

    destinations = {"retrieve": "retrieve", "traverse": "traverse", "generate": "generate"}
    graph.add_conditional_edges("link_entities", route_system_c, destinations)
    graph.add_conditional_edges("retrieve", route_system_c, destinations)
    graph.add_conditional_edges("traverse", route_system_c, destinations)
    graph.add_edge("generate", END)

    return graph.compile()


def run_system_c(
    example: Example,
    knowledge_graph: nx.MultiDiGraph,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_calls: int = DEFAULT_MAX_CALLS,
) -> SystemResult:
    """Answer one question with System C.

    `knowledge_graph` must be supplied by the caller (see
    `data.kg_2wiki.load_dense_graph`) rather than built here, since it
    is shared across every question in an evaluation run. To pass this
    system to `eval.harness.run_harness`, which expects a one-argument
    callable, bind it first: `functools.partial(run_system_c,
    knowledge_graph=graph)`.
    """
    index = PassageIndex(example.passages)
    compiled = build_system_c_graph(index, knowledge_graph, example)

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
    from data.kg_2wiki import load_dense_graph
    from data.load_2wiki import load_2wiki

    print("Loading dense knowledge graph (dev + train)...")
    knowledge_graph = load_dense_graph()

    examples = load_2wiki(n=5, seed=0)
    for example in examples:
        result = run_system_c(example, knowledge_graph)
        print(f"question: {example.question}")
        print(f"gold: {example.answer!r}  predicted: {result.predicted_answer!r}")
        print(
            f"llm_calls={result.llm_calls} total_tokens={result.total_tokens} "
            f"latency={result.latency_seconds:.2f}s"
        )
        print()
