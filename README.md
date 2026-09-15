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
| 5 | Shared LangGraph state schema + Budget Check | Pending |
| 6 | System B | Pending |
| 7 | Knowledge-graph construction | Pending |
| 8 | Graph-Traverse node | Pending |
| 9 | System C | Pending |
| 10 | Full comparison run | Pending |

System A's full run on the 1,000-question validation sample: **EM 0.236, F1 0.411**, lowest on compositional and inference (genuine multi-hop) questions, highest on comparison questions — the expected weakness of a single-shot baseline, and the gap Systems B and C are built to close.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

# Run System A over a few questions
python3 -m agents.graphs.system_a

# Run the full evaluation harness (checkpointed — safe to interrupt and rerun)
python3 -m eval.harness
```

Harness output lands in `results/` as CSV, one row per (question, system): raw and normalized prediction, EM/F1, LLM calls, tokens, latency, benchmark, and question type. A rerun skips any question already recorded for that system, so an interrupted run resumes rather than restarting.

## Repository layout

```
agents/
    nodes/generate.py      # shared frozen-LLM generation step
    graphs/system_a.py      # vanilla single-shot RAG pipeline
data/
    load_2wiki.py            # dataset loading, sampling, corpus construction
retrieval/
    embed.py                  # shared sentence-transformer encoder
    faiss_index.py             # per-question dense passage index
eval/
    metrics.py                 # EM/F1 with SQuAD-style normalization
    harness.py                  # runs a system over sampled questions, logs results
results/
    (output CSVs land here, not version controlled)
```
