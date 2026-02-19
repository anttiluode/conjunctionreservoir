"""
Basic usage example for ConjunctionReservoir.
"""

import sys
sys.path.insert(0, "..")

from conjunctionreservoir import ConjunctionReservoir

# ---------------------------------------------------------------------------
# 1. Index a list of pre-split chunks
# ---------------------------------------------------------------------------

chunks = [
    # Two chunks that each contain ONE of the query terms in isolation
    "NMDA receptors are voltage-dependent ion channels requiring magnesium unblocking. "
    "They play a central role in synaptic plasticity and LTP.",

    "Coincidence detection is a fundamental property of many neural circuits. "
    "It allows neurons to respond only to simultaneous inputs from multiple sources.",

    # The target: BOTH terms in the SAME sentence
    "NMDA receptors implement coincidence detection by requiring simultaneous "
    "presynaptic glutamate release and postsynaptic depolarization to open.",

    # Unrelated
    "The hippocampus generates theta oscillations during spatial navigation. "
    "Place cells fire at specific phases of the theta cycle.",

    "Reservoir computing uses a fixed random recurrent network. "
    "Only the output weights are trained.",
]

r = ConjunctionReservoir(conjunction_threshold=0.5)
r.build_index(chunks, verbose=False)

print("Query: 'NMDA coincidence detection synapse'")
print()

results = r.retrieve("NMDA coincidence detection synapse", top_k=3)
for rank, (chunk, score) in enumerate(results, 1):
    print(f"  {rank}. [{score:.4f}]  {chunk[:90]}...")

print()
print("Note: chunk #3 (both terms in same sentence) should rank first.")
print()

# ---------------------------------------------------------------------------
# 2. Inspect the index
# ---------------------------------------------------------------------------

print("Index summary:")
for k, v in r.summary().items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 3. Coverage profile after multiple queries
# ---------------------------------------------------------------------------

r.retrieve("NMDA receptor plasticity")
r.retrieve("NMDA calcium influx")
r.retrieve("NMDA magnesium block")

print()
print("Coverage profile after 3 NMDA queries:")
profile = r.coverage_profile()
print(f"  Mean coverage: {profile['mean_coverage']}")
print(f"  Most-visited sentences:")
for sent, cov in profile["most_covered"][:3]:
    print(f"    [{cov:.4f}] {sent[:70]}...")

# ---------------------------------------------------------------------------
# 4. Tune the conjunction threshold
# ---------------------------------------------------------------------------

print()
print("Effect of conjunction_threshold on scoring:")
for thresh in [0.0, 0.3, 0.5, 1.0]:
    r2 = ConjunctionReservoir(conjunction_threshold=thresh)
    r2.build_index(chunks, verbose=False)
    results2 = r2.retrieve("NMDA coincidence detection", top_k=3)
    top_chunk = results2[0][0][:60] if results2 else "(none)"
    has_both = "NMDA" in top_chunk and "coincidence" in top_chunk.lower()
    mark = "✓" if has_both else "✗"
    print(f"  threshold={thresh:.1f}  top result {mark}: {top_chunk}...")
