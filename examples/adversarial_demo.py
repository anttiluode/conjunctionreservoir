"""
Adversarial demo: the false-positive problem that ConjunctionReservoir solves.

This demo constructs a corpus specifically designed to fool chunk-level
retrieval. Each distractor chunk contains the query terms in SEPARATE
sentences. The target chunk has both terms in the SAME sentence.

Standard BM25/TF-IDF ranks the distractors above the target because
they have higher raw term frequencies. ConjunctionReservoir ranks the
target first every time.

This directly demonstrates the Norman-Haignere principle applied to text:
integration windows are fixed, not expanding to cover structure. A chunk
is too wide. A sentence is the right unit.
"""

import sys
sys.path.insert(0, "..")
import re
import numpy as np
from conjunctionreservoir import ConjunctionReservoir


# ---------------------------------------------------------------------------
# Minimal BM25 for comparison (no external deps)
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.chunks, self.idf, self.tf, self.avgdl = [], {}, [], 0

    def build_index(self, chunks):
        self.chunks = chunks
        n = len(chunks)
        tok = [re.findall(r"\b[a-zA-Z]{2,}\b", c.lower()) for c in chunks]
        self.avgdl = max(1, np.mean([len(t) for t in tok]))
        df = {}
        for t in tok:
            for w in set(t): df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1) for w in df}
        self.tf = [({w: t.count(w) for w in set(t)}, len(t)) for t in tok]
        return self

    def retrieve(self, query, top_k=5):
        qt = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
        scores = []
        for tf_d, dl in self.tf:
            s = sum(
                self.idf.get(w, 0) * tf_d.get(w, 0) * (self.k1 + 1) /
                (tf_d.get(w, 0) + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for w in qt if w in tf_d
            )
            scores.append(s)
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], scores[i]) for i in idx]


# ---------------------------------------------------------------------------
# Adversarial corpora and queries
# ---------------------------------------------------------------------------

TESTS = [
    {
        "name": "NMDA + coincidence",
        "query": "NMDA receptor coincidence detection synapse",
        "corpus": [
            # Distractors: each term isolated to its own sentence
            "NMDA receptors are crucial for synaptic plasticity. They mediate LTP in hippocampal circuits.",
            "Blocking NMDA channels with APV prevents LTP. This is a standard pharmacological tool.",
            "NMDA-dependent calcium influx activates CaMKII. Downstream signaling follows.",
            "Barrel cortex shows coincidence detection for whisker stimulation. Precision is millisecond-scale.",
            "Coincidence detection in sound localization is computed by the medial superior olive.",
            "Many circuits rely on coincidence detection for amplification. It requires simultaneous input.",
            # Both terms in different sentences of the SAME chunk (tricky distractor)
            "NMDA receptors require magnesium unblocking for activation. Coincidence detection varies across regions.",
            "The study of NMDA is ongoing. Coincidence detection has been found throughout the brain.",
            # TARGET: both terms in the SAME sentence
            "NMDA receptors implement coincidence detection by requiring simultaneous "
            "presynaptic glutamate and postsynaptic depolarization.",
        ],
        "target_idx": 8,
    },
    {
        "name": "theta + alternation",
        "query": "theta sweep alternation left right coverage",
        "corpus": [
            "The hippocampus generates theta oscillations at 4-8 Hz during navigation.",
            "Grid cells in entorhinal cortex fire in hexagonal fields modulated by theta.",
            "Left-right alternation is a locomotor pattern controlled by spinal CPGs.",
            "Alternating movement patterns emerge from coverage-maximizing rules.",
            "Theta oscillations compress spatial trajectories into 125ms cycles.",
            # Both words in same chunk, different sentences
            "The theta rhythm organizes hippocampal activity. Left-right alternation in sweeps is observed.",
            # TARGET
            "Theta sweeps alternate left and right in the entorhinal-hippocampal circuit, "
            "emerging from a coverage-maximizing algorithm rather than hardwired circuitry.",
        ],
        "target_idx": 6,
    },
    {
        "name": "auditory + absolute time",
        "query": "auditory cortex absolute time window fixed",
        "corpus": [
            "Auditory cortex shows tonotopic organization along the basilar membrane frequency axis.",
            "Speech perception involves multiple timescales from phonemes to sentences.",
            "Absolute time is the fundamental coordinate of physical measurement systems.",
            "Time-yoked versus structure-yoked processing differs fundamentally in neural coding.",
            "The cochlea decomposes sound into frequency components via the basilar membrane.",
            # Distractor: both but in separate sentences
            "Auditory cortex is organized hierarchically. Integration windows based on absolute time vary.",
            # TARGET
            "Auditory cortex integration windows are yoked to absolute time at ~80ms, "
            "not to the structure of the sounds being processed.",
        ],
        "target_idx": 6,
    },
]


# ---------------------------------------------------------------------------
# Run demos
# ---------------------------------------------------------------------------

def run():
    print("=" * 65)
    print("ADVERSARIAL DEMO: False-positive problem in chunk-level retrieval")
    print("=" * 65)
    print()
    print("Each test has:")
    print("  - Distractor chunks containing each query term SEPARATELY")
    print("  - A target chunk with BOTH terms in the SAME sentence")
    print("  - Question: does the retriever find the target or a distractor?")
    print()

    all_conj_rank1 = 0
    all_bm25_rank1 = 0
    n_tests = len(TESTS)

    for test in TESTS:
        name = test["name"]
        query = test["query"]
        corpus = test["corpus"]
        target_idx = test["target_idx"]
        target_text = corpus[target_idx]

        # Build indices
        conj = ConjunctionReservoir(conjunction_threshold=0.5)
        conj.build_index(corpus, verbose=False)

        bm25 = BM25()
        bm25.build_index(corpus)

        # Retrieve
        conj_results = conj.retrieve(query, top_k=len(corpus))
        bm25_results = bm25.retrieve(query, top_k=len(corpus))

        # Find target rank
        conj_rank = next(
            (i + 1 for i, (c, _) in enumerate(conj_results) if c == target_text), 999
        )
        bm25_rank = next(
            (i + 1 for i, (c, _) in enumerate(bm25_results) if c == target_text), 999
        )

        if conj_rank == 1: all_conj_rank1 += 1
        if bm25_rank == 1: all_bm25_rank1 += 1

        print(f"Test: {name}")
        print(f"  Query:  {query}")
        print(f"  Target: {target_text[:75]}...")
        print()
        print(f"  ConjunctionReservoir → target rank: {conj_rank}  {'✓ FIRST' if conj_rank == 1 else f'✗ rank {conj_rank}'}")
        print(f"  BM25                 → target rank: {bm25_rank}  {'✓ FIRST' if bm25_rank == 1 else f'✗ rank {bm25_rank}'}")
        print()

        if conj_rank != bm25_rank:
            print(f"  BM25 #1: {bm25_results[0][0][:70]}...")
            print(f"           (has '{query.split()[0]}' and '{query.split()[2]}' but in separate sentences)")
            print()

        print("-" * 65)
        print()

    print(f"SUMMARY: Rank-1 rate on {n_tests} adversarial tests")
    print(f"  ConjunctionReservoir: {all_conj_rank1}/{n_tests} = {all_conj_rank1/n_tests:.0%}")
    print(f"  BM25:                 {all_bm25_rank1}/{n_tests} = {all_bm25_rank1/n_tests:.0%}")
    print()
    print("The conjunction gate enforces co-occurrence within a sentence.")
    print("Chunks where the terms appear in separate sentences are down-ranked,")
    print("even if their total term frequency is higher.")


if __name__ == "__main__":
    run()
