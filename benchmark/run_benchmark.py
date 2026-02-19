"""
Benchmark: ConjunctionReservoir vs SweepBrain vs BM25
======================================================
Tests two query types:

  conjunction_specific: query requires BOTH terms to appear in the
    same sentence. Target chunk has them together; distractors have
    each term isolated. Measures: target rank (lower = better).

  broad_similarity: query has relevant chunks spread across the corpus.
    No single "right" answer. Measures: Recall@3.

Run with:
    python benchmark/run_benchmark.py
"""

import sys
import re
import time
import numpy as np

sys.path.insert(0, "..")
from conjunctionreservoir import ConjunctionReservoir


# ---------------------------------------------------------------------------
# Minimal BM25
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
        scores = [
            sum(
                self.idf.get(w, 0) * td.get(w, 0) * (self.k1 + 1) /
                (td.get(w, 0) + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for w in qt if w in td
            )
            for td, dl in self.tf
        ]
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], scores[i]) for i in idx]


# ---------------------------------------------------------------------------
# Minimal SweepBrain
# ---------------------------------------------------------------------------

class SweepBrain:
    def __init__(self, seed=42):
        self.seed = seed
        self.chunks = []
        self.vocab = None
        self.idf = None
        self.W_in = None
        self.feats = None

    def build_index(self, chunks):
        self.chunks = chunks
        counts = {}
        for c in chunks:
            for w in re.findall(r"\b[a-zA-Z]{2,}\b", c.lower()):
                counts[w] = counts.get(w, 0) + 1
        self.vocab = {
            w: i for i, (w, _) in enumerate(
                sorted(counts.items(), key=lambda x: -x[1])[:1500]
            )
        }
        n = len(chunks)
        df = np.zeros(len(self.vocab))
        for c in chunks:
            for w in set(re.findall(r"\b[a-zA-Z]{2,}\b", c.lower())):
                if w in self.vocab: df[self.vocab[w]] += 1
        self.idf = np.log((n + 1) / (df + 1)) + 1
        rng = np.random.default_rng(self.seed)
        d = len(self.vocab)
        self.W_in = rng.standard_normal((d, 256)) * np.sqrt(1.0 / d)
        raw = np.array([self._enc(c) for c in chunks])
        proj = np.tanh(raw @ self.W_in)
        states = np.zeros_like(proj)
        for k in range(len(chunks)):
            diff = raw - raw[k][np.newaxis, :]
            gate = np.exp(-0.5 * np.sum(diff ** 2, axis=1) / 1.44)
            gate /= gate.sum() + 1e-8
            states[k] = proj.T @ gate
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        self.feats = states / (norms + 1e-8)
        return self

    def _enc(self, text):
        vec = np.zeros(len(self.vocab))
        for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()):
            if w in self.vocab: vec[self.vocab[w]] += 1
        vec *= self.idf
        return vec / (np.linalg.norm(vec) + 1e-8)

    def retrieve(self, query, top_k=5):
        q = np.tanh(self._enc(query) @ self.W_in)
        q /= np.linalg.norm(q) + 1e-8
        sims = self.feats @ q
        idx = sims.argsort()[-top_k:][::-1]
        return [(self.chunks[i], float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# Test corpus — adversarial for conjunction queries
# ---------------------------------------------------------------------------

CORPUS = [
    # Cluster 1: NMDA + coincidence
    "NMDA receptors are crucial for synaptic plasticity. They mediate LTP in hippocampal circuits.",
    "Blocking NMDA channels with APV prevents LTP. This pharmacological tool is widely used.",
    "NMDA-dependent calcium influx activates CaMKII. Downstream signaling leads to AMPA insertion.",
    "Barrel cortex shows coincidence detection for whisker stimulation. Precision is millisecond-scale.",
    "Coincidence detection in sound localization is computed by the medial superior olive.",
    "NMDA receptors require magnesium unblocking. Coincidence detection mechanisms vary across regions.",
    # TARGET 6
    "NMDA receptors implement coincidence detection by requiring simultaneous presynaptic "
    "glutamate release and postsynaptic depolarization.",

    # Cluster 2: theta + alternation
    "The hippocampus generates theta oscillations at 4-8 Hz during active navigation.",
    "Grid cells in entorhinal cortex fire in hexagonal fields modulated by theta phase.",
    "Left-right alternation is a locomotor pattern controlled by spinal central pattern generators.",
    "Alternating sweeps maximize spatial coverage without repeated visits.",
    "The theta rhythm organizes hippocampal activity into 125ms cycles. Left-right alternation is efficient.",
    # TARGET 12
    "Theta sweeps alternate left and right in the entorhinal-hippocampal circuit, "
    "implementing coverage-maximizing exploration.",

    # Cluster 3: Alzheimer + eigenmode
    "Alzheimer disease involves progressive neurodegeneration starting in entorhinal cortex.",
    "EEG shows slowing in Alzheimer patients with decreased alpha and increased delta power.",
    "Eigenmode analysis decomposes brain activity into spatially structured resonant patterns.",
    "Connectome eigenmodes form a natural basis for brain activity decomposition.",
    "Alzheimer patients show memory deficits. Eigenmode analysis is a promising biomarker approach.",
    # TARGET 18
    "Eigenmode dwell time analysis differentiates Alzheimer disease from healthy controls "
    "with statistical significance in EEG recordings.",

    # Cluster 4: auditory + absolute time
    "Auditory cortex is tonotopically organized with high frequencies anteriorly.",
    "Speech perception involves processing at multiple timescales from phonemes to sentences.",
    "Absolute time measurements require a stable reference frame independent of local dynamics.",
    "Time-yoked versus structure-yoked processing differs fundamentally in sensory neuroscience.",
    "Auditory processing is hierarchical. Absolute time windows vary across cortical regions.",
    # TARGET 24
    "Auditory cortex integration windows are yoked to absolute time at approximately 80ms, "
    "not to the structure of the sounds being processed.",

    # Cluster 5: reservoir + spectral radius
    "Echo state networks use a fixed random reservoir with only readout weights trained.",
    "Reservoir computing avoids backpropagation through time for efficient learning.",
    "The spectral radius of a matrix determines whether its powers grow or decay over time.",
    "Matrix eigenvalues control stability in linear dynamical systems.",
    "Reservoir networks have fixed recurrent weights. The spectral radius requires careful tuning.",
    # TARGET 30
    "The spectral radius of the reservoir weight matrix controls the memory duration, "
    "with values near 1.0 providing the longest temporal echo.",
]

QUERIES = [
    {"q": "NMDA receptor coincidence detection synapse",    "target": 6,  "type": "conj"},
    {"q": "theta sweep alternation left right entorhinal",  "target": 12, "type": "conj"},
    {"q": "Alzheimer eigenmode dwell time EEG",             "target": 18, "type": "conj"},
    {"q": "auditory cortex absolute time window not struct","target": 24, "type": "conj"},
    {"q": "reservoir spectral radius memory echo duration", "target": 30, "type": "conj"},
    {"q": "hippocampus theta oscillation navigation",       "relevant": [7,8,11,12], "type": "broad"},
    {"q": "neural coincidence temporal precision circuit",  "relevant": [3,4,5,6],  "type": "broad"},
    {"q": "brain EEG biomarker disease",                    "relevant": [13,14,17,18], "type": "broad"},
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def target_rank(results, target_chunk):
    for i, (c, _) in enumerate(results):
        if c == target_chunk:
            return i + 1
    return len(CORPUS) + 1


def recall_at_k(results, relevant_indices, k=3):
    retrieved = {c for c, _ in results[:k]}
    relevant = {CORPUS[i] for i in relevant_indices}
    return len(retrieved & relevant) / len(relevant)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print()
    print("=" * 65)
    print("ConjunctionReservoir  vs  SweepBrain  vs  BM25")
    print("=" * 65)
    print(f"Corpus: {len(CORPUS)} chunks  |  Queries: {len(QUERIES)}")
    print()

    t0 = time.perf_counter()
    conj = ConjunctionReservoir(conjunction_threshold=0.5)
    conj.build_index(CORPUS, verbose=False)
    tc = time.perf_counter() - t0

    t0 = time.perf_counter()
    sweep = SweepBrain()
    sweep.build_index(CORPUS)
    ts = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25 = BM25()
    bm25.build_index(CORPUS)
    tb = time.perf_counter() - t0

    print(f"Build  Conjunction={tc*1000:.0f}ms  SweepBrain={ts*1000:.0f}ms  BM25={tb*1000:.0f}ms")
    print()

    conj_ranks, sweep_ranks, bm25_ranks = [], [], []
    conj_rec, sweep_rec, bm25_rec = [], [], []

    print(f"{'Query':<50}  Conj  Sweep  BM25")
    print("-" * 70)

    for qd in QUERIES:
        q = qd["q"]
        cr = conj.retrieve(q, top_k=5)
        sr = sweep.retrieve(q, top_k=5)
        br = bm25.retrieve(q, top_k=5)

        if qd["type"] == "conj":
            tc2 = CORPUS[qd["target"]]
            r_c = target_rank(cr, tc2)
            r_s = target_rank(sr, tc2)
            r_b = target_rank(br, tc2)
            conj_ranks.append(r_c)
            sweep_ranks.append(r_s)
            bm25_ranks.append(r_b)
            label = f"rank {r_c:2d} / {r_s:2d} / {r_b:2d}"
        else:
            c_s = recall_at_k(cr, qd["relevant"])
            s_s = recall_at_k(sr, qd["relevant"])
            b_s = recall_at_k(br, qd["relevant"])
            conj_rec.append(c_s)
            sweep_rec.append(s_s)
            bm25_rec.append(b_s)
            label = f"R@3  {c_s:.2f} /  {s_s:.2f} /  {b_s:.2f}"

        print(f"{q[:50]:<50}  {label}")

    print("-" * 70)
    print()

    print("CONJUNCTION QUERIES — Mean Target Rank (lower = better):")
    print(f"  ConjunctionReservoir : {np.mean(conj_ranks):.2f}")
    print(f"  SweepBrain           : {np.mean(sweep_ranks):.2f}")
    print(f"  BM25                 : {np.mean(bm25_ranks):.2f}")
    print()

    r1_c = sum(r == 1 for r in conj_ranks) / len(conj_ranks)
    r1_s = sum(r == 1 for r in sweep_ranks) / len(sweep_ranks)
    r1_b = sum(r == 1 for r in bm25_ranks) / len(bm25_ranks)
    print("  Rank-1 Rate:")
    print(f"  ConjunctionReservoir : {r1_c:.0%}")
    print(f"  SweepBrain           : {r1_s:.0%}")
    print(f"  BM25                 : {r1_b:.0%}")
    print()

    if conj_rec:
        print("BROAD QUERIES — Recall@3:")
        print(f"  ConjunctionReservoir : {np.mean(conj_rec):.3f}")
        print(f"  SweepBrain           : {np.mean(sweep_rec):.3f}")
        print(f"  BM25                 : {np.mean(bm25_rec):.3f}")
        print()

    print("=" * 65)
    if np.mean(conj_ranks) <= min(np.mean(sweep_ranks), np.mean(bm25_ranks)):
        print("✓ ConjunctionReservoir wins on conjunction-specific queries.")
        print("  Sentence-level co-occurrence requirement beats chunk-level")
        print("  term frequency when specificity matters.")
    else:
        print("Result is mixed — see per-query breakdown above.")

    print()
    print("Note: ConjunctionReservoir intentionally trades broad-query")
    print("recall for precision on specific co-occurrence queries.")
    print("Use conjunction_threshold=0.0 to approach standard TF-IDF.")


if __name__ == "__main__":
    run()
