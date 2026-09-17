# Agentic Knowledge-Graph RAG

Comparing three question-answering strategies on multi-hop QA:

- **System A** — vanilla single-shot vector RAG.
- **System B** — multi-step vector RAG with query reformulation, no knowledge-graph access.
- **System C** — an agentic system that chooses, per step, between vector retrieval, knowledge-graph traversal, or answering, under a hop/call budget.

All three share the same frozen LLM, prompt template, and embedding model, so differences in accuracy and cost are attributable to retrieval strategy rather than to generation. The current phase targets [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) only, since it ships gold entity IDs and gold reasoning-path triples, avoiding the need for knowledge-graph construction from scratch.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | 2WikiMultiHopQA loading and sampling | Done |
| 2 | Embedding + FAISS passage index | Done |
| 3 | System A (single-shot RAG) | Done |
| 4 | Evaluation harness (EM/F1, cost, checkpointing) | Done |
| 5 | Shared LangGraph state schema + Budget Check | Done |
| 6 | System B | Done |
| 7 | Knowledge-graph construction | Done |
| 8 | Graph-Traverse node | Done |
| 9 | System C | Done |
| 10 | Full comparison run | Done |

Paired results, same 1,000-question validation sample, all three systems:

| Metric | A | B | C |
|---|---|---|---|
| EM | 0.236 | 0.269 | **0.349** |
| F1 | 0.411 | **0.504** | 0.433 |
| Avg tokens/question | 377 | 666 | **154** |
| Avg latency/question | 0.94s | 0.88s | 0.84s |

By question type (EM):

| Question type | A | B | C |
|---|---|---|---|
| comparison | 0.307 | 0.280 | **0.427** |
| bridge_comparison | 0.269 | 0.174 | **0.420** |
| inference | 0.196 | 0.206 | **0.402** |
| compositional | 0.194 | **0.325** | 0.263 |

System C reaches the best overall exact-match accuracy at well under half System A's token cost and under a quarter of System B's, and dominates on three of four question types (nearly doubling System A/B's accuracy on inference questions). The exception is compositional (genuine bridge-chain) questions, where System B wins: these questions lean on less-common bridge entities where the knowledge graph is most likely to dead-end partway through the chain (see step 7 below on graph sparsity), leaving less retrieval budget for the fallback than System B gets by committing to two full retrieval rounds from the start. System C's F1 (0.433) also trails System B's (0.504) despite C's higher EM: graph-traversal answers tend to be exactly right or completely wrong, while System B's passage-based answers pick up more partial credit on near-misses.

Iteration alone (System B over A) helps most on compositional questions and slightly hurts comparison-type questions, plausibly from added retrieval noise on questions answerable from one clean passage.

**What these cost figures do and don't include:** all cost numbers above (tokens, latency) are per-question, query-time cost only — the LLM calls and time spent answering a question once the knowledge graph already exists. They exclude the one-time cost of building that graph (downloading the ~700MB training split, parsing ~180,000 questions, producing 165,826 nodes / 171,536 edges — informally observed to take a minute or two, but not rigorously timed or logged) and say nothing about the cost of updating it incrementally versus rebuilding from scratch. Both are explicitly out of scope for this phase (see `CLAUDE.md`) and are planned as a later experiment, not an oversight — System C's apparent cost advantage should be read as an amortized, query-time-only figure until that work exists.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

Generation calls an external LLM API. Create a `.env` file in the repo root:

```
OPENAI_API_KEY=your-key-here
```

## Running

```bash
# Inspect the dataset loader against a sample
python3 -m data.load_2wiki

# Inspect the retrieval index against a sample question
python3 -m retrieval.faiss_index

# Build and inspect the dense knowledge graph (downloads the training
# split on first run, ~700MB, needed for graph density — see
# data/kg_2wiki.py)
python3 -m data.kg_2wiki

# Run a system over a few questions
python3 -m agents.graphs.system_a
python3 -m agents.graphs.system_b
python3 -m agents.graphs.system_c

# Run the full evaluation harness (checkpointed — safe to interrupt and rerun)
python3 -m eval.harness
```

Harness output lands in `results/` as CSV, one row per (question, system): raw and normalized prediction, EM/F1, LLM calls, tokens, latency, benchmark, and question type. A rerun skips any question already recorded for that system, so an interrupted run resumes rather than restarting.

## Repository layout

```
agents/
    state.py                     # shared TypedDict state for Systems B and C
    budget.py                     # Budget Check (forced-stop condition)
    router.py                      # non-LLM routing heuristics for B and C
    nodes/
        entity_linker.py            # seeds entity_frontier (System C)
        vector_retrieve.py           # iterative retrieval with query reformulation
        graph_traverse.py             # scores and selects real graph edges
        generate.py                    # shared frozen-LLM generation step
    graphs/
        system_a.py                    # vanilla single-shot RAG pipeline
        system_b.py                     # multi-step, no graph
        system_c.py                      # full agentic system
data/
    load_2wiki.py                   # dataset loading, sampling, corpus construction
    kg_2wiki.py                      # dense knowledge graph from gold triples
retrieval/
    embed.py                          # shared sentence-transformer encoder
    faiss_index.py                     # per-question dense passage index
eval/
    metrics.py                          # EM/F1 with SQuAD-style normalization
    harness.py                           # runs a system over sampled questions, logs results
results/
    (output CSVs land here, not version controlled)
```
