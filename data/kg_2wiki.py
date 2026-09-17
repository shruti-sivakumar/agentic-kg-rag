"""
Knowledge-graph construction for System C's Graph-Traverse step.

The graph is built by aggregating gold reasoning-path triples
(`evidences_id`) across many questions, not filtered down to any single
question's own path. This is required for the traversal step to be a
genuine search rather than a lookup of the answer key: if the graph
available at evaluation time contained only a given question's own
gold triples, a frontier entity would generally have exactly one
outgoing edge — its own correct next hop — and "selecting the highest-
scoring candidate" would be selecting among one option, which is the
oracle trap in effect even without directly indexing by question ID.

Aggregating across the validation split alone (12,576 questions) is
not enough in practice. 2WikiMultiHopQA's gold paths are drawn from a
wide spread of biographical and filmographic facts that mostly do not
otherwise overlap between questions, so most entities appear in only
one question's chain. Measured against the seed entities that this
project's own 1,000-question evaluation sample actually starts from:
built from the validation split alone, only 61% of those seed entities
appear in the graph at all, and just 158 of them (11% of the ones with
any outgoing edge) have more than one — meaning traversal from the
other 89% has no real candidate to weigh against the one available
edge. Aggregating the training split in as well (roughly 167,000
further questions) raises coverage to 88% of seed entities and more
than quadruples the number with a genuine choice of outgoing edge (641).
Note that this is a targeted, sample-specific density check, not a
claim that the graph is dense overall: the training split barely moves
the graph's average out-degree (1.07 to 1.11), because most of its
newly added entities are themselves only ever mentioned once. What
matters is not overall density but density exactly where this
project's evaluation questions will actually query the graph, and on
that measure the training split is worth its cost.
"""

from __future__ import annotations

import re
from typing import Iterable

import networkx as nx

from data.load_2wiki import load_raw_records

_QID_PATTERN = re.compile(r"^Q[0-9]+$")

EvidencePair = tuple[list[list[str]], list[list[str]]]  # (evidences, evidences_id)


def _is_qid(value: str) -> bool:
    return bool(_QID_PATTERN.match(value))


def build_knowledge_graph(evidence_pairs: Iterable[EvidencePair]) -> nx.MultiDiGraph:
    """Aggregate (evidences, evidences_id) pairs into one dense multi-relational graph.

    Nodes are keyed by Wikidata QID rather than surface string, since
    distinct entities can share a name; each node carries one observed
    surface-form label (`name`) for verbalizing triples later. Edges
    carry the relation as it appears in the source data.

    Two conditions exclude a triple rather than let it corrupt the
    graph:

    - `evidences_id` shorter than `evidences` for a record means the
      whole reasoning path failed to resolve to entities (the
      all-or-nothing quirk documented in `data/load_2wiki.py`), so the
      record contributes nothing.
    - A subject or object that fails the Wikidata QID pattern is a
      literal (a date, a quantity) rather than an entity — such values
      cannot be graph nodes, so the triple is skipped even when the
      rest of that record's path resolved correctly.

    Repeated identical triples (same subject, relation, and object,
    seen across different questions) are collapsed into a single edge:
    a duplicate contributes no new candidate for the traversal step to
    weigh, only redundant work.
    """
    graph = nx.MultiDiGraph()
    seen_edges: set[tuple[str, str, str]] = set()

    for evidences, evidences_id in evidence_pairs:
        if len(evidences_id) != len(evidences):
            continue
        for (subj_name, relation, obj_name), (subj_qid, _rel_id, obj_qid) in zip(evidences, evidences_id):
            if not (_is_qid(subj_qid) and _is_qid(obj_qid)):
                continue

            if subj_qid not in graph:
                graph.add_node(subj_qid, name=subj_name)
            if obj_qid not in graph:
                graph.add_node(obj_qid, name=obj_name)

            edge_key = (subj_qid, relation, obj_qid)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            graph.add_edge(subj_qid, obj_qid, relation=relation)

    return graph


def load_dense_graph(splits: tuple[str, ...] = ("dev", "train")) -> nx.MultiDiGraph:
    """Build the dense graph from the given splits' raw records."""
    evidence_pairs = (
        (record["evidences"], record["evidences_id"])
        for split in splits
        for record in load_raw_records(split)
    )
    return build_knowledge_graph(evidence_pairs)


if __name__ == "__main__":
    from data.load_2wiki import load_2wiki

    graph = load_dense_graph()
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    out_degrees = [d for _, d in graph.out_degree()]
    nonzero = [d for d in out_degrees if d > 0]
    multi_choice = sum(1 for d in out_degrees if d > 1)
    print(f"Nodes with at least one outgoing edge: {len(nonzero)}")
    print(f"Mean out-degree (over nodes with >=1 edge): {sum(nonzero) / len(nonzero):.2f}")
    print(f"Nodes with more than one outgoing edge: {multi_choice}")

    # The whole-graph statistics above are a weak proxy for what actually
    # matters: how well-connected this graph is exactly where the
    # project's own evaluation sample will query it.
    examples = load_2wiki(n=1000, seed=42)
    seed_entities = {qid for example in examples for qid in example.entity_ids}
    present = [q for q in seed_entities if q in graph]
    multi = [q for q in present if graph.out_degree(q) > 1]
    print(f"\nOf {len(seed_entities)} seed entities in the 1,000-question evaluation sample:")
    print(f"  present in the graph: {len(present)}")
    print(f"  with a genuine choice of outgoing edge: {len(multi)}")

    sample_node = max(graph.nodes, key=lambda n: graph.out_degree(n))
    print(f"\nExample frontier entity: {graph.nodes[sample_node]['name']} ({sample_node})")
    for _, target, data in list(graph.out_edges(sample_node, data=True))[:5]:
        print(f"  -> {data['relation']} -> {graph.nodes[target]['name']} ({target})")
