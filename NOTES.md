# Project Notes — read this when you've forgotten what's going on

This is a plain-English walkthrough of the project: what it is, why it exists, and what's been built so far, in the order it was built. Read top to bottom the first time; after that, skip to "Where things stand right now" at the end to reorient yourself.

---

## Part 1 — What is this project, actually?

You're building and comparing **three different ways of answering questions with an AI system**, to see which approach gets the best accuracy for the least cost (money, time, number of AI calls).

The questions are all **multi-hop questions** — questions that can't be answered from one fact alone, but need two or more facts chained together. For example:

> "Who is the paternal grandmother of Princess Alexandrine of Prussia?"

To answer this, you can't just look up "Princess Alexandrine's grandmother" directly — you have to:
1. Find Princess Alexandrine's father.
2. Find that father's mother.

That's two "hops." Real multi-hop QA datasets are full of questions like this, and they're specifically designed to be hard for systems that only do a single lookup.

### The three systems being compared

- **System A** — the simple baseline. Take the question, search for relevant text passages, hand them to an AI model, get an answer. One search, one answer, done. No matter how complex the question is, it only gets one shot at finding the right information.
- **System B** — smarter searching, but still just text search. If one search isn't enough, it can search again with a refined query, gathering more clues before answering — like System A but allowed to take a few tries.
- **System C** — the most advanced. It has a **knowledge graph** available (a network of facts like "Alexandrine → father → Prince X" and "Prince X → mother → Princess Y") and can choose, at each step, to either search text, walk along the graph to a connected fact, or answer. It decides for itself which action makes sense, under a limited budget of steps.

The whole point of the project is to measure: **does giving the system a knowledge graph to walk actually help, and how much more expensive is it?** System B exists specifically so you can tell the difference between "it got better because it could try more times" (B) versus "it got better because the graph specifically helped" (C). Without B, you couldn't tell those two things apart.

### Why this matters (the actual research contribution)

This isn't about inventing a clever new algorithm. It's about **honestly measuring the tradeoff**: as you go from A → B → C, does accuracy go up, and by how much does cost (AI calls, tokens, time) go up too? That accuracy-vs-cost picture, broken down by question type, is the deliverable.

---

## Part 2 — Vocabulary you'll need

A short glossary, since these terms show up constantly in the code and in this document:

- **RAG (Retrieval-Augmented Generation)**: instead of asking an AI model a question cold, you first *retrieve* relevant text passages (via search), and hand those to the model along with the question. This gives it facts to work from instead of relying purely on what it memorized during training.
- **Embedding**: a way of converting a piece of text into a list of numbers (a vector) such that similar-meaning texts end up with similar numbers. This is how "search by meaning" works — you convert the question and every candidate passage into vectors, and find the passages whose vectors are closest to the question's vector.
- **FAISS**: a library for doing that "find the closest vectors" search quickly.
- **Knowledge graph**: a network of facts, each written as a **triple**: (subject, relation, object) — e.g. (Xawery Żuławski, mother, Małgorzata Braunek). System C can "walk" from one entity to a connected one by following these triples.
- **Entity**: a specific real-world thing — a person, place, film, etc. Each entity in this dataset has a unique ID (a "QID" from Wikidata, e.g. `Q42`), separate from its name, because names can be ambiguous.
- **Hop**: one step of gathering information (one search, or one graph-walk step).
- **LLM call**: one request to an AI language model. These cost money and take time, so counting them matters for the cost side of the comparison.
- **Budget**: a limit on how many hops and how many LLM calls a system is allowed to use per question before it's forced to just answer with whatever it has.
- **EM (Exact Match) / F1**: how answers get graded. EM is strict — the answer must match the correct answer exactly (after cleaning up capitalization/punctuation). F1 is more forgiving — it gives partial credit based on word overlap, like scoring "Louise Mecklenburg" as *mostly* right against the correct answer "Louise of Mecklenburg-Strelitz."
- **The "oracle trap"**: an important trap to avoid. The dataset used here (2WikiMultiHopQA) actually *ships the correct chain of facts* needed to answer each question, because that's how researchers verify systems are reasoning correctly. It would be very easy to accidentally build System C so it just looks up this answer key directly — which would make the numbers meaningless. So the code is deliberately built so the graph-walking step has to genuinely search among several real candidate facts and guess the best one, the same way it would have to in a system that didn't have an answer key at all. The answer key is only used afterward, to check whether the guess was right.

---

## Part 3 — What's been built so far, in order

The project follows a fixed build order (see `CLAUDE.md` if you want the full technical spec). Here's what each completed step actually does and why it came in this order.

### Step 1 — Getting the data

**File:** `data/load_2wiki.py`

Before anything else, you need real questions to work with. This step downloads the **2WikiMultiHopQA** dataset (12,576 real multi-hop trivia questions built from Wikipedia/Wikidata), and:
- Picks a random, fixed sample of 1,000 questions to use for all testing (so results are reproducible — the same 1,000 questions every time).
- For each question, packages up its "corpus": 10 short text passages (2 of them actually contain the answer, 8 are decoys/distractors) — this is what System A/B search over.
- Also pulls out each question's type (`comparison`, `inference`, `compositional`, `bridge_comparison` — four different flavors of multi-hop reasoning) and its gold entity IDs and gold fact-chain, for later use.

**Why first:** everything else needs real data to test against. There's no point building search or AI-answering logic if you don't yet know what the actual questions and passages look like.

### Step 2 — Making text searchable

**Files:** `retrieval/embed.py`, `retrieval/faiss_index.py`

This step builds the "search" half of RAG. It:
- Loads one shared embedding model (a small AI model whose only job is converting text into those "meaning vectors" mentioned above).
- For a given question, converts all 10 of its passages into vectors, and converts the question into a vector too, then uses FAISS to find which passages' vectors are closest to the question's — i.e., which passages are most likely to be relevant.

**Why second:** this is the retrieval engine every system needs (A, B, and C all search text at some point). Built and tested on its own first, before adding any AI-answering on top, so that if something's wrong later, you know it's not the search step.

**Sanity check done:** ran it on 50 real questions — the top search result was one of the 2 relevant passages 88% of the time. That's well above random chance (which would be ~20%, since 2 of 10 passages are relevant), confirming the search actually works.

### Step 3 — The simplest possible system (System A)

**Files:** `agents/nodes/generate.py`, `agents/graphs/system_a.py`

Now that search works, this step adds the "generation" half: take the top few search results, hand them to an AI model along with the question, and get a short answer back. This is System A in full: search once, answer once.

The AI model used is **gpt-4o-mini** (a small, cheap OpenAI model) — chosen because it's inexpensive enough to run 1,000+ questions for a few cents, and because it answers directly without an internal "thinking" step that would complicate cost-tracking (more on why that matters below).

**Why third:** System A is deliberately the simplest possible pipeline, with no fancy control logic — it exists to prove the basic retrieve → generate stack actually works before adding any complexity like multi-step loops or graph-walking on top.

### Step 4 — Grading the answers (the evaluation harness)

**Files:** `eval/metrics.py`, `eval/harness.py`

An AI system that gives answers is useless without a way to check if those answers are *right*, and at what cost. This step builds:
- **Scoring** (`metrics.py`): implements EM and F1 scoring, with text cleanup first (lowercase, remove "a/an/the", strip punctuation) so that trivial formatting differences don't count as wrong answers.
- **The harness** (`harness.py`): runs any system over a batch of questions, scores every answer, and logs one row per question — the answer, whether it was right, and the cost (how many AI calls, how many tokens, how long it took). Results get saved as a CSV file in `results/`.

An important feature: the harness **checkpoints** — it saves each result to disk immediately, and if it gets interrupted (rate limit, crash, closing the laptop), rerunning the same command picks up where it left off instead of starting over or losing work.

**Why fourth, and why "test the harness before building more":** if the *grading* logic has a bug, every number produced afterward is wrong in a way that's very hard to notice later — you'd think System C was underperforming when actually the scorer was broken. So the rule was: don't build System B or C until the harness has been proven correct against System A's real output.

**What this produced:** the first real result. System A was run on all 1,000 questions:

| Question type | Accuracy (EM) |
|---|---|
| comparison | 30.7% |
| bridge_comparison | 26.9% |
| inference | 19.6% |
| compositional | 19.4% |
| **Overall** | **23.6%** |

This shape is exactly what you'd expect: "comparison" questions (e.g. "which came out first, film X or film Y?") can often be answered from one relevant passage, so single-shot search does okay. "Compositional"/"inference" questions need real multi-hop chaining, which a single search can't reliably capture — hence the low scores. This gap is precisely what Systems B and C are meant to close.

*(Side note on the AI model: an earlier attempt used a free-tier model on Groq, which turned out to have a hard daily cost/usage limit that a full 1,000-question run blew through partway. Switched to gpt-4o-mini, which comfortably handles the whole run in one sitting for a few cents — worth knowing in case "Groq" shows up anywhere in old files or git history.)*

### Step 5 — The shared scaffolding for Systems B and C (current step)

**Files:** `agents/state.py`, `agents/budget.py`, plus an addition to `agents/nodes/generate.py`

Systems B and C, unlike A, aren't a straight line — they loop: search (or walk the graph), check if you've done enough, maybe search again, eventually answer. To build a *loop* like that in code, you use a framework called **LangGraph**, which represents the system as a graph of steps ("nodes") connected by rules about what happens next ("edges"). This step builds the shared pieces both B and C's loops will be made of:

- **`state.py`** — defines what information gets carried around and updated at every step of the loop: the question, all the facts gathered so far, how many hops/calls have been used, the budget limits, and (once reached) the final answer. Also tracks running totals of tokens/time/whether any answer got cut off, since those numbers are needed for the same cost comparison System A already produced.
- **`budget.py`** — the "have we run out of budget?" check that runs before every decision in the loop. If the budget (hops or AI calls) is used up, it forces the system straight to the answering step — a partial, budget-exhausted attempt still has to produce *some* answer, never nothing.
- **`generate.py`** — got a small addition (`generate_node`) so the same answering logic System A uses can also plug into a LangGraph loop for B and C.

**Why fifth:** rather than building System B and System C as two separate, duplicated pieces of code, this shared scaffolding gets built once, tested on its own, and then Systems B and C get assembled on top of it in later steps — reusing the same building blocks, so the comparison between them stays fair (same underlying machinery, different available actions).

**Sanity check done:** built a tiny test loop where the budget is set to zero (so it's exhausted immediately), confirmed it correctly skips straight to answering, and confirmed the answer step still worked end-to-end (asked "What is the capital of France?", got "Paris" back, with cost figures correctly recorded).

---

## Part 4 — What's left to build

In order:

6. **System B** — the multi-step, text-only system. Needs a simple "should I search again or answer now?" decision rule (not another AI call — a plain heuristic, so it doesn't distort the cost comparison), plus logic for rephrasing the search query using what's been learned so far.
7. **Building the knowledge graph** — turning the dataset's fact-chains into an actual graph structure System C can walk. Important: built from the *full* set of facts across many questions, not just the answer key for one question — otherwise System C's graph-walking would just be reading the answer key (the "oracle trap" mentioned above).
8. **The graph-walking step** — given a starting entity, look at all its real connected facts, and pick the one that seems most relevant to the question (by comparing meanings, the same "vector similarity" trick used in search) — a genuine guess, not a lookup.
9. **System C** — put it all together: a decision-maker that can choose between searching text, walking the graph, or answering, at each step.
10. **The full comparison** — run all three systems over the same 1,000 questions and produce the final accuracy-vs-cost results table.

---

## Where things stand right now

- Steps 1-5 are done, tested against real data, and committed to git.
- System A has a full result on all 1,000 questions (23.6% exact-match accuracy overall — the number Systems B and C need to beat).
- The AI model in use is `gpt-4o-mini` via OpenAI (you'll need an `OPENAI_API_KEY` in a `.env` file for anything that makes AI calls to work).
- Nothing costs meaningful money — the entire project so far has cost well under a dollar in API usage.
- Next up: Step 6, System B.
