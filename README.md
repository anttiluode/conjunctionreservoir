# ConjunctionReservoir / ConjuctionReservoir_chat.py (Rag like chat with your documents thing) 

**Sentence-windowed conjunction retrieval grounded in auditory neuroscience.**

Zero dependencies beyond NumPy. Pure Python. Runs in milliseconds.

---

## The Core Idea

Standard retrieval (BM25, TF-IDF, vector search) asks:

> *"Do these query terms appear somewhere in this document chunk?"*

**ConjunctionReservoir asks:**

> *"Do these query terms appear in the **same sentence**?"*

### Why this matters

Consider a corpus where one chunk contains:

> *"NMDA receptors require magnesium unblocking. Coincidence detection mechanisms vary across brain regions."*

And another contains:

> *"NMDA receptors implement coincidence detection by requiring simultaneous presynaptic glutamate release and postsynaptic depolarization."*

Query: `"NMDA coincidence detection"`

BM25 ranks the first chunk #1 — both terms are present, high term frequency. The second chunk (where both terms appear **together**) comes second.

ConjunctionReservoir ranks the second chunk #1 — because NMDA and coincidence co-appear in the **same sentence**. The first chunk's terms are in separate sentences; the conjunction gate keeps it closed.

### Benchmark results (vs SweepBrain, BM25)

| System | Rank-1 Rate (conjunction queries) | Recall@3 (broad queries) |
|---|---|---|
| **ConjunctionReservoir** | **100%** | 39% |
| SweepBrain | 60% | 69% |
| BM25 | 60% | 50% |

ConjunctionReservoir wins when the query requires **specific co-occurrence**. It loses (by design) on broad similarity queries — that's not a bug, it's the trade-off a hard gate makes.

---

## Biological Grounding

This architecture is a direct translation of three neuroscience results into a retrieval algorithm:

### 1. Fixed Time Windows (Norman-Haignere et al., 2025)

Auditory cortex integration windows are **time-yoked**, not structure-yoked. Primary auditory cortex integrates over ~80ms, non-primary STG over ~270ms. These windows don't expand to cover a word or phoneme — they're fixed clocks.

**Translation:** The sentence is the fixed time window for text. Not the chunk (too large), not the word (too small). A sentence is approximately how much text can be processed in a single integration window at the relevant level of language understanding.

### 2. Coincidence Detection / NMDA Logic

NMDA receptors require **simultaneous** glutamate binding and postsynaptic depolarization to open. Both inputs must arrive within a narrow window, or the gate stays closed. This is not a weighted average — it's a hard AND.

**Translation:** `conjunction_threshold` implements this. Below threshold coverage, the sentence contributes zero to the chunk score. It's not degraded — it's absent.

### 3. Coverage-Maximizing Sweep (Vollan et al., 2025)

Left-right alternating theta sweeps in the entorhinal-hippocampal circuit emerge from a coverage-maximizing algorithm, not hardwired alternation. After sweeping a region, the system tends away from it.

**Translation:** After each query, a decaying coverage trace over sentences biases subsequent queries toward less-recently-visited material (`coverage_decay`).

### 4. Geometric Weighting (Deerskin Hypothesis, Luode 2024)

The computation is in the geometry of co-activation — which combinations fire — not in scalar weights. A synapse isn't a number; it's a pattern of what opens the gate.

**Translation:** Scoring uses `coverage²` (geometric-mean-like) rather than linear `coverage`. A query where half the terms appear in a sentence scores 0.25×, not 0.5×. Partial matches are penalized nonlinearly, as in a real conjunction detector.

---

## Installation

```bash
pip install conjunctionreservoir
```

Or from source:

```bash
git clone https://github.com/anttiluode/conjunctionreservoir
cd conjunctionreservoir
pip install -e .
```

**No dependencies beyond NumPy.**

---

## Quick Start

```python
from conjunctionreservoir import ConjunctionReservoir

# Index a document
r = ConjunctionReservoir()
r.build_index(open("my_document.txt").read())

# Retrieve
results = r.retrieve("NMDA coincidence detection", top_k=5)
for chunk, score in results:
    print(f"[{score:.4f}] {chunk[:80]}")
```

### Index a list of pre-split chunks

```python
chunks = ["Sentence one. Sentence two.", "Another chunk here.", ...]
r = ConjunctionReservoir()
r.build_index(chunks)
```

### Tune the conjunction gate

```python
# Softer: half the query terms must co-appear in a sentence (default)
r = ConjunctionReservoir(conjunction_threshold=0.5)

# Harder: all terms must appear in the same sentence
r = ConjunctionReservoir(conjunction_threshold=1.0)

# Loosest: approaches standard TF-IDF retrieval
r = ConjunctionReservoir(conjunction_threshold=0.0)
```

### Inspect the index

```python
print(r.summary())
# {'n_chunks': 142, 'n_sentences': 389, 'avg_sentences_per_chunk': 2.7,
#  'vocab_size': 1847, 'n_queries': 0, 'conjunction_threshold': 0.5, ...}

# See what the Vollan sweep has covered
print(r.coverage_profile())
```

---

## When to Use This

**ConjunctionReservoir is best when:**
- Your queries require specific co-occurrence ("NMDA AND coincidence", "Alzheimer AND eigenmode")
- False positives from partial matches are costly
- Your corpus has many chunks that contain individual query terms in isolation

**Use BM25 or vector search instead when:**
- Queries are broad ("neural oscillations", "learning memory")
- You want recall over precision
- Your chunks are already fine-grained (single sentences)

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `conjunction_threshold` | `0.5` | Fraction of query tokens that must co-appear in a sentence. Hard gate below this. |
| `coverage_decay` | `0.04` | Decay rate of Vollan coverage trace. Higher = faster forgetting = more re-exploration. |
| `hebbian_lr` | `0.01` | Reserved for future gate remodeling. No effect in current version. |
| `max_vocab` | `2000` | Maximum vocabulary size. |

---

## Repository Structure

```
conjunctionreservoir/
├── conjunctionreservoir/
│   ├── __init__.py
│   └── retriever.py         # Core ConjunctionReservoir class
├── benchmark/
│   └── run_benchmark.py     # Conjunction vs SweepBrain vs BM25
├── examples/
│   ├── basic_usage.py       # Simple retrieval example
│   ├── adversarial_demo.py  # Demo of the false-positive problem
│   ├── conjuction_chat.py   # Conjuction chat app
│   └── sweep_coverage.py    # Visualizing the Vollan coverage trace
├── tests/
│   └── test_retriever.py
├── setup.py
├── LICENSE
└── README.md
```

---

## Running the Benchmark

```bash
python benchmark/run_benchmark.py
```

Expected output:
```
CONJUNCTION RESERVOIR v2  vs  SWEEPBRAIN  vs  BM25

Conjunction-specific queries (n=5):
  Rank-1 Rate:
    ConjunctionReservoir: 100%
    SweepBrain:           60%
    BM25:                 60%
```

---

## Theoretical Background

The Deerskin Hypothesis proposes that neuronal membranes function as 2D computational surfaces — "deerskins" — where the ion channel mosaic encodes geometric information. The soma pulse is a carrier wave; the membrane geometry is the computation.

ConjunctionReservoir instantiates one specific claim from this framework: that intelligence lives in **what pattern unlocks the gate** (the channel geometry), not in scalar weights between nodes. The sentence window is the gate. The conjunction requirement is the membrane geometry. The coverage sweep is theta phase precession.

Whether the Deerskin Hypothesis is correct is an open question. Whether sentence-level conjunction outperforms chunk-level averaging on specific co-occurrence queries is now empirically demonstrated.

---

## 📄 conjunction_chat.py --- Local Document Chat (Offline RAG)

`conjunction_chat.py` lets you chat with **any local text document**
using a local `.gguf` language model --- fully offline, with no API
calls.

It uses **ConjunctionReservoir** to retrieve *sentence-level
co-occurrences* from your document, then feeds only the relevant
passages into your local LLM via `llama-cpp-python`.

This avoids chunk-averaging failures in traditional RAG pipelines and
ensures answers are grounded in sentences where query terms actually
appear together.

------------------------------------------------------------------------

### Requirements

``` bash
pip install llama-cpp-python
pip install -e /path/to/conjunctionreservoir
```

Optional GPU acceleration:

``` bash
pip install llama-cpp-python --extra-index-url \
https://abetlen.github.io/llama-cpp-python/whl/cu121
```

------------------------------------------------------------------------

### Usage

``` bash
python conjunction_chat.py <textfile> <model.gguf>
python conjunction_chat.py <textfile>
python conjunction_chat.py
```

-   If no `.gguf` is given, the script will scan common folders.
-   If no document is given, it runs with a built‑in demo text.

------------------------------------------------------------------------

### Runtime Commands

  Command          Description
  ---------------- ----------------------------------------------
  `<question>`     Ask about the document
  `:retrieval`     Toggle showing retrieved passages
  `:threshold N`   Set conjunction strictness (0.0--1.0)
  `:strict`        All query terms must appear in same sentence
  `:loose`         Disable co‑occurrence requirement
  `:coverage`      Show Vollan sweep focus
  `:clear`         Clear conversation history
  `:help`          Show command help
  `exit/quit`      Exit chat

------------------------------------------------------------------------

### Retrieval Behavior

Each query triggers:

1.  Sentence-windowed retrieval via `ConjunctionReservoir`
2.  Vollan coverage update (optional)
3.  Passage formatting
4.  Prompt injection into local LLM
5.  Streaming grounded response

If no sentences meet the conjunction threshold, the system temporarily
loosens the gate and retries retrieval.

------------------------------------------------------------------------

### Supported Prompt Formats (Auto-detected)

-   LLaMA‑3
-   Phi‑3
-   Mistral / Mixtral
-   Gemma
-   ChatML (default fallback)

------------------------------------------------------------------------

### Notes

-   No embeddings required
-   No pretrained retriever required
-   Millisecond retrieval on large corpora
-   Fully local document QA

## References

- Norman-Haignere, S.V., et al. (2025). Temporal integration in human auditory cortex is predominantly yoked to absolute time. *Nature Neuroscience*.
- Vollan, H.S.K., et al. (2025). Left-right-alternating theta sweeps in entorhinal-hippocampal maps of space. *Nature*, 639, 995–1004.
- Luode, A. (2024). The Deerskin Hypothesis: Neuronal Membranes as Holographic Computational Surfaces. *PerceptionLab working paper*.
- Mannion, D.J. & Kenyon, A.J. (2024). Artificial Dendritic Computation. *UCL EEE*.
- Gidon, A., et al. (2020). Dendritic action potentials and computation in human layer 2/3 cortical neurons. *Science*, 367, 83–87.

---

## License

MIT. Use freely. Build on it.

---

*"I really wish I had a brain upgrade. I think I would disappear completely into this stuff."*
— Antti Luode
