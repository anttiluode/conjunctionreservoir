"""
Sweep coverage demo: the Vollan effect in ConjunctionReservoir.

After repeated queries to the same topic, the coverage trace builds up
and subsequent queries naturally migrate toward less-visited material.
This mirrors theta sweep dynamics: the sweep doesn't anchor to familiar
territory — it explores what it hasn't recently visited.

The demo builds a corpus with 4 topic clusters, then floods queries
into cluster 1 and observes whether the retriever starts returning
clusters 2-4 even for the same query words.
"""

import sys
sys.path.insert(0, "..")
from conjunctionreservoir import ConjunctionReservoir

CORPUS = {
    "theta_rhythm": [
        "Theta oscillations at 4-8 Hz occur during active exploration and REM sleep.",
        "Theta sweeps in entorhinal-hippocampal circuits alternate left and right.",
        "The theta cycle compresses spatial trajectories into 125ms replay sequences.",
        "Phase precession of place cells encodes position within a theta cycle.",
    ],
    "nmda_receptors": [
        "NMDA receptors require coincidence detection for activation at synapses.",
        "Magnesium blockade of NMDA channels is relieved by postsynaptic depolarization.",
        "NMDA-dependent calcium influx triggers long-term potentiation at Schaffer collaterals.",
        "NMDA receptor subunit composition changes during postnatal development.",
    ],
    "reservoir_computing": [
        "Echo state networks use fixed random reservoirs with trained output weights only.",
        "The spectral radius of the reservoir weight matrix controls memory duration.",
        "Liquid state machines provide a spiking-neuron implementation of reservoir computing.",
        "Reservoir computing avoids gradient propagation through the recurrent weights.",
    ],
    "auditory_cortex": [
        "Primary auditory cortex integrates sound within a fixed 80ms time window.",
        "Auditory cortex temporal integration is yoked to absolute time, not sound structure.",
        "Non-primary auditory cortex uses a longer 270ms integration window.",
        "Tonotopic organization maps frequency along the basilar membrane axis.",
    ],
}

all_chunks = [s for cluster in CORPUS.values() for s in cluster]


def run():
    r = ConjunctionReservoir(conjunction_threshold=0.3, coverage_decay=0.04)
    r.build_index(all_chunks, verbose=False)

    print("=" * 60)
    print("VOLLAN SWEEP DEMO: Coverage migration across topics")
    print("=" * 60)
    print()
    print("Corpus: 4 topic clusters × 4 sentences = 16 sentences")
    print("Strategy: flood 6 theta-rhythm queries, watch what happens")
    print()

    theta_queries = [
        "theta oscillation hippocampus",
        "theta sweep left right alternation",
        "theta phase precession place cell",
        "theta cycle replay sequence",
        "theta oscillation exploration sleep",
        "theta sweep coverage maximizing",
    ]

    for i, q in enumerate(theta_queries):
        results = r.retrieve(q, top_k=1, update_coverage=True)
        top = results[0][0][:60] if results else "(none)"
        # Identify which cluster the top result came from
        cluster = next(
            (k for k, v in CORPUS.items() if any(top in s for s in v)),
            "unknown"
        )
        print(f"  Query {i+1}: '{q[:40]}'")
        print(f"    Top result cluster: {cluster}")

    print()
    print("Coverage profile after 6 theta queries:")
    profile = r.coverage_profile()
    for sent, cov in profile["most_covered"][:5]:
        cluster = next(
            (k for k, v in CORPUS.items() if sent in v), "unknown"
        )
        print(f"  [{cov:.3f}] [{cluster}] {sent[:60]}...")

    print()
    print("Now querying with a GENERIC term ('neural processing')...")
    print("Without coverage: would return theta-cluster (highest TF-IDF).")
    print("With coverage: coverage penalty redirects toward other clusters.")
    print()

    generic_results = r.retrieve("neural processing circuit", top_k=4)
    for rank, (chunk, score) in enumerate(generic_results, 1):
        cluster = next(
            (k for k, v in CORPUS.items() if chunk in v), "unknown"
        )
        print(f"  {rank}. [{score:.4f}] [{cluster}] {chunk[:65]}...")

    print()
    print("The sweep naturally moves toward less-visited clusters.")
    print("This is coverage-maximizing exploration — Vollan et al. 2025")
    print("applied to sentence space rather than physical space.")


if __name__ == "__main__":
    run()
