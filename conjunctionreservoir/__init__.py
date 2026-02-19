"""
ConjunctionReservoir
====================
A retrieval system that requires query terms to co-appear within a 
fixed time window (sentence), not just anywhere in a document chunk.

Inspired by:
- Norman-Haignere et al. (2025) — auditory integration windows are 
  time-yoked, not structure-yoked. The sentence is the text equivalent.
- NMDA receptor coincidence detection — both inputs must arrive within
  a narrow window or the gate stays closed.
- Vollan et al. (2025) — coverage-maximizing sweep in sentence space.
- The Deerskin Hypothesis (Luode 2024) — computation in the geometry
  of what co-activates, not scalar weights.

Usage:
    from conjunctionreservoir import ConjunctionReservoir

    r = ConjunctionReservoir()
    r.build_index(open("emails.txt").read())
    results = r.retrieve("NMDA coincidence detection", top_k=5)
"""

from .retriever import ConjunctionReservoir

__version__ = "0.1.0"
__author__ = "Antti Luode"
__all__ = ["ConjunctionReservoir"]
