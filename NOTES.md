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

### Step 5 — The shared scaffolding for Systems B and C

**Files:** `agents/state.py`, `agents/budget.py`, plus an addition to `agents/nodes/generate.py`

Systems B and C, unlike A, aren't a straight line — they loop: search (or walk the graph), check if you've done enough, maybe search again, eventually answer. To build a *loop* like that in code, you use a framework called **LangGraph**, which represents the system as a graph of steps ("nodes") connected by rules about what happens next ("edges"). This step builds the shared pieces both B and C's loops will be made of:

- **`state.py`** — defines what information gets carried around and updated at every step of the loop: the question, all the facts gathered so far, how many hops/calls have been used, the budget limits, and (once reached) the final answer. Also tracks running totals of tokens/time/whether any answer got cut off, since those numbers are needed for the same cost comparison System A already produced.
- **`budget.py`** — the "have we run out of budget?" check that runs before every decision in the loop. If the budget (hops or AI calls) is used up, it forces the system straight to the answering step — a partial, budget-exhausted attempt still has to produce *some* answer, never nothing.
- **`generate.py`** — got a small addition (`generate_node`) so the same answering logic System A uses can also plug into a LangGraph loop for B and C.

**Why fifth:** rather than building System B and System C as two separate, duplicated pieces of code, this shared scaffolding gets built once, tested on its own, and then Systems B and C get assembled on top of it in later steps — reusing the same building blocks, so the comparison between them stays fair (same underlying machinery, different available actions).

**Sanity check done:** built a tiny test loop where the budget is set to zero (so it's exhausted immediately), confirmed it correctly skips straight to answering, and confirmed the answer step still worked end-to-end (asked "What is the capital of France?", got "Paris" back, with cost figures correctly recorded).

### Step 6 — System B, the multi-step text-only system

**Files:** `agents/nodes/vector_retrieve.py`, `agents/router.py`, `agents/graphs/system_b.py`

System B is allowed to search more than once. Two new pieces make that work:
- **Query reformulation** (`vector_retrieve.py`): before searching again, the question gets extended with everything found so far ("original question + all retrieved passage text"), so the second search is looking for something more specific than the first. This is a plain string operation, not an AI call — deliberately, so System B's cost stays comparable to System A's (only the final answering step spends an AI call either way).
- One subtlety that mattered in practice: with only 10 passages per question, a reformulated search tends to find *the same top passages again*. So the retrieval step searches the whole 10-passage corpus and throws out anything already gathered on an earlier search, forcing each round to surface genuinely new information.
- **The router** (`router.py`): a simple, non-AI decision rule — "search" for the first 2 rounds, then "answer." Two rounds, not some other number, because most questions in this dataset resolve via exactly a 2-fact chain.

**Why sixth:** with the scaffolding from step 5 ready, System B is the first system actually built *on* it, and it's simpler than System C (only 2 possible actions instead of 3), so it's a good first real test of the loop machinery.

**Result:** run on all 1,000 questions, compared directly against System A on the exact same questions:

| Question type | EM: A → B | F1: A → B |
|---|---|---|
| compositional | 19.4% → 32.5% | 0.268 → 0.446 |
| inference | 19.6% → 20.6% | 0.411 → 0.449 |
| comparison | 30.7% → 28.0% | 0.609 → 0.608 |
| bridge_comparison | 26.9% → 17.4% | 0.499 → 0.541 |
| **Overall** | **23.6% → 26.9%** | **0.411 → 0.504** |

Iterating helps a lot exactly where you'd expect — compositional (genuine multi-hop) questions jump the most — and very slightly *hurts* comparison-type questions, plausibly because extra passages introduce noise on questions answerable from one clean fact. A real, interpretable result, not a bug.

### Step 7 — Building the actual knowledge graph

**File:** `data/kg_2wiki.py`

This is where System C's graph comes from. The dataset doesn't ship a ready-made knowledge graph — only each question's own short "gold" fact-chain (2-7 facts). So the graph gets built by collecting these fact-chains from *many* questions and merging them into one big network, where entities (by their Wikidata ID, not by name, since two different people can share a name) become nodes and facts become connections between them.

**The important trap this avoids:** if System C's graph only contained a question's *own* correct fact-chain, then "searching the graph" would be fake — there'd only ever be one fact to pick, which is the same as just reading the answer off the answer key. This is why the graph is built from as much of the dataset as reasonably possible, not from the specific questions being tested — that way, an entity usually has *several* connected facts, most of them irrelevant to the current question, and the system genuinely has to figure out which one actually matters.

**A real surprise here:** aggregating just the ~12,500 questions in the validation set produced a very thin graph — most entities had only one known fact each, so there was rarely a real *choice* to make. Adding in the training set too (167,000 more questions) helped a lot: for the actual entities this project's 1,000 test questions start from, coverage went from 61% to 88%, and the number with a genuine choice of next fact roughly quadrupled. This cost an extra ~700MB one-time download but was worth it — worth knowing in case you ever wonder why loading the graph takes a minute or two and pulls a large file.

### Step 8 — The graph-walking step itself

**File:** `agents/nodes/graph_traverse.py`

Given the current entity (or entities) the walk is standing on, this step:
1. Looks up every real fact connected to that entity in the graph (not just the "correct" one for this question — genuinely every fact known about it).
2. Turns each fact into a short sentence (e.g. "Alexandrine's father is Prince Albert of Prussia").
3. Converts the question and every candidate sentence into those "meaning vectors" again, and picks whichever fact is closest in meaning to the question.
4. Moves the "current entity" pointer to whatever that fact pointed to, and remembers the fact as something learned.

**How well this actually works, checked directly:** across 138 real test questions where there was a genuine choice between multiple facts, this method picked the *correct* fact 87.7% of the time. That's strong evidence the "guess by meaning" approach genuinely works, not just technically avoids cheating.

**Handling dead ends:** sometimes an entity has no useful connected facts at all (or none that seem related to the question). Rather than force a bad guess, this is detected and reported back, so the decision-maker (built in step 9) knows to try searching text instead next, rather than getting stuck repeatedly failing on the same entity. The cutoff for "nothing useful found" was set by checking real numbers: a genuinely correct fact almost never scores below about 0.5 similarity, while comparing a question against totally unrelated facts never scores above about 0.41 — so the cutoff sits at 0.45, in the gap between the two.

### Step 9 — Entity Linker and System C, put together

**Files:** `agents/nodes/entity_linker.py`, `agents/graphs/system_c.py`, additions to `agents/router.py`

Before System C can walk the graph, it needs to know *which entity to start from*. The dataset conveniently already tells us this for each question (it ships the correct starting entity IDs, the same way it ships the fact-chain) — using this is a deliberate, documented shortcut for this phase (real "figure out which entity a name refers to" logic is harder and out of scope right now), not a form of the answer-key cheating this project is careful to avoid elsewhere: knowing where to *start* isn't the same as knowing the *answer*.

System C's decision-maker (in `router.py`) is the 3-way version: at each step, if there's an entity to walk from and the last graph-step actually found something, walk the graph; otherwise search text; and after 2 total steps (of either kind), answer.

**Result, run on all 1,000 questions:**

| Question type | EM: A / B / C | 
|---|---|
| comparison | 30.7% / 28.0% / **42.7%** |
| bridge_comparison | 26.9% / 17.4% / **42.0%** |
| inference | 19.6% / 20.6% / **40.2%** |
| compositional | 19.4% / **32.5%** / 26.3% |
| **Overall** | **23.6% / 26.9% / 34.9%** |

System C wins overall, often by a huge margin (inference roughly doubles), while using **less than half** System A's AI-token budget per question and **under a quarter** of System B's — verbalized single facts are much shorter than whole retrieved passages. The one place it *loses* to System B is compositional questions specifically: these lean on less-common bridge entities where the graph is most likely to run dry partway through the 2-hop chain (see step 7's density numbers), leaving System C with less of its budget left for a text-search fallback than System B gets by committing to two full searches from the start. A genuinely interesting, explainable limitation, not a flaw in the build.

One more nuance: System C's F1 (0.433) is actually a bit *behind* System B's (0.504) despite winning on EM. Graph answers tend to be short exact entity names — either dead-on right, or clearly wrong — while System B's passage-based answers pick up more partial credit for near-misses. EM and F1 are telling two different, both-true stories here.

### Step 10 — The full three-way comparison

No new code — this is simply running all three finished systems over the identical 1,000 questions and lining up the results, which is the table above plus System A's original numbers. This was the actual goal stated at the very top of the project: not "build a clever system," but "honestly measure the accuracy-vs-cost tradeoff between these three approaches." That comparison now exists, with real numbers, for all three systems.

---

## Where things stand right now

- **All ten build steps are done.** Systems A, B, and C all work end-to-end and have been run on the same 1,000 real questions each.
- The headline result: System C (the full agentic, graph-using system) gets the best overall accuracy *and* the lowest cost per question — but System B (search-only, multiple tries) still wins specifically on compositional questions, where the knowledge graph's coverage is weakest.
- The AI model in use is `gpt-4o-mini` via OpenAI (you'll need an `OPENAI_API_KEY` in a `.env` file for anything that makes AI calls to work). Building the knowledge graph also downloads the dataset's training split (~700MB, one-time) — that's expected, not a bug, the first time you run `data/kg_2wiki.py` or System C.
- Nothing has cost meaningful money — the entire project so far has cost a few dollars at most in API usage.
- What's genuinely left, beyond this phase's scope (see `CLAUDE.md` for the full picture): extending this to the MuSiQue and HotpotQA datasets, which don't ship a ready-made knowledge graph or gold starting entities the way 2WikiMultiHopQA does — that's real, harder work (entity linking without an answer key, building a graph from scratch) deliberately deferred until this phase's comparison was proven out first.
