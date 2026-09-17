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
| 10 | Full comparison run | Pending |

Paired results on the same 1,000-question validation sample (Systems A and B; System C not yet run at full scale):

| Question type | EM (A → B) | F1 (A → B) |
|---|---|---|
| compositional | 0.194 → 0.325 | 0.268 → 0.446 |
| inference | 0.196 → 0.206 | 0.411 → 0.449 |
| comparison | 0.307 → 0.280 | 0.609 → 0.608 |
| bridge_comparison | 0.269 → 0.174 | 0.499 → 0.541 |
| **Overall** | **0.236 → 0.269** | **0.411 → 0.504** |

Iteration (System B) helps most on compositional questions — genuine multi-hop chains — and slightly hurts comparison-type questions, plausibly from added retrieval noise on questions answerable from one clean passage. A 5-question spot check of System C shows it resolving full 2-hop gold chains via genuine graph traversal alone, at substantially lower token cost than A/B's passage-based context; a full 1,000-question run is the remaining step.

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
