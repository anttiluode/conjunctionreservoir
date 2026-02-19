"""
Tests for ConjunctionReservoir.

Run with:  pytest tests/
"""

import pytest
import sys
sys.path.insert(0, "..")

from conjunctionreservoir import ConjunctionReservoir
from conjunctionreservoir.retriever import (
    split_sentences,
    chunk_document,
    build_vocab,
    encode_text,
    tfidf_weights,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    text = "This is the first sentence here. This is the second sentence here. And the third."
    sents = split_sentences(text, min_len=10)
    assert len(sents) == 3
    assert "first" in sents[0]


def test_split_sentences_min_len():
    text = "Hi. This is a proper sentence with enough words."
    sents = split_sentences(text, min_len=15)
    assert len(sents) == 1  # "Hi" is too short


def test_build_vocab():
    texts = ["hello world", "hello again world"]
    vocab = build_vocab(texts, max_vocab=10)
    assert "hello" in vocab
    assert "world" in vocab


def test_encode_text_normalized():
    import numpy as np
    vocab = {"hello": 0, "world": 1}
    idf = tfidf_weights(["hello world"], vocab)
    vec = encode_text("hello world", vocab, idf)
    assert abs(float((vec ** 2).sum() ** 0.5) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Build index tests
# ---------------------------------------------------------------------------

def test_build_from_list():
    chunks = ["The cat sat on the mat.", "Dogs bark loudly at night."]
    r = ConjunctionReservoir()
    r.build_index(chunks, verbose=False)
    assert r.vocab is not None
    assert len(r.chunk_texts) == 2
    assert len(r.all_sentences) >= 2


def test_build_from_string():
    text = "First chunk text here. It has two sentences.\n\nSecond chunk is different."
    r = ConjunctionReservoir()
    r.build_index(text, verbose=False)
    assert len(r.chunk_texts) >= 1


def test_build_error_empty():
    r = ConjunctionReservoir()
    with pytest.raises(ValueError):
        r.build_index([], verbose=False)


def test_retrieve_before_index_raises():
    r = ConjunctionReservoir()
    with pytest.raises(RuntimeError):
        r.retrieve("query")


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

ADVERSARIAL_CORPUS = [
    # Distractor: NMDA in sentence 1, coincidence in sentence 2
    "NMDA receptors are crucial for plasticity. Coincidence detection varies across regions.",
    # Distractor: NMDA only
    "NMDA channels require magnesium unblocking for activation.",
    # Distractor: coincidence only
    "Coincidence detection occurs in the medial superior olive for sound localization.",
    # TARGET: both in same sentence
    "NMDA receptors implement coincidence detection by requiring simultaneous inputs.",
    # Unrelated
    "The hippocampus generates theta oscillations during navigation.",
]
TARGET = ADVERSARIAL_CORPUS[3]


def test_conjunction_ranks_target_first():
    r = ConjunctionReservoir(conjunction_threshold=0.5)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    results = r.retrieve("NMDA coincidence detection", top_k=5)
    assert results[0][0] == TARGET, (
        f"Expected target at rank 1, got: {results[0][0][:60]}"
    )


def test_top_k_respected():
    r = ConjunctionReservoir()
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    results = r.retrieve("NMDA coincidence", top_k=3)
    assert len(results) == 3


def test_fallback_when_threshold_too_high():
    """With threshold=1.0 and a query with 4 tokens, most sentences fail.
    Should fall back to TF-IDF and still return results."""
    r = ConjunctionReservoir(conjunction_threshold=1.0)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    results = r.retrieve("NMDA receptor coincidence detection activation", top_k=3)
    assert len(results) > 0  # fallback fires


def test_threshold_zero_approaches_tfidf():
    """With threshold=0 all sentences pass, behavior approaches TF-IDF."""
    r = ConjunctionReservoir(conjunction_threshold=0.0)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    results = r.retrieve("NMDA", top_k=5)
    assert len(results) > 0
    # Some NMDA chunk should be in top 2
    top2_texts = " ".join(c for c, _ in results[:2])
    assert "NMDA" in top2_texts


# ---------------------------------------------------------------------------
# Coverage / Vollan tests
# ---------------------------------------------------------------------------

def test_coverage_updates():
    r = ConjunctionReservoir(coverage_decay=0.1)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    assert r.n_queries == 0
    r.retrieve("NMDA coincidence", update_coverage=True)
    assert r.n_queries == 1
    assert r.sentence_coverage.max() > 0


def test_coverage_no_update():
    r = ConjunctionReservoir()
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    r.retrieve("NMDA", update_coverage=False)
    assert r.n_queries == 0
    assert r.sentence_coverage.max() == 0


def test_coverage_decays():
    import numpy as np
    r = ConjunctionReservoir(coverage_decay=0.5)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    r.retrieve("NMDA coincidence")
    cov1 = r.sentence_coverage.copy()
    r.retrieve("theta oscillation")  # different topic
    cov2 = r.sentence_coverage
    # Previous NMDA coverage should have decayed
    assert np.sum(cov2) != np.sum(cov1)


# ---------------------------------------------------------------------------
# Summary / introspection tests
# ---------------------------------------------------------------------------

def test_summary_fields():
    r = ConjunctionReservoir(conjunction_threshold=0.4)
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    s = r.summary()
    assert s["n_chunks"] == len(ADVERSARIAL_CORPUS)
    assert s["conjunction_threshold"] == 0.4
    assert s["n_queries"] == 0
    assert s["vocab_size"] > 0


def test_coverage_profile():
    r = ConjunctionReservoir()
    r.build_index(ADVERSARIAL_CORPUS, verbose=False)
    r.retrieve("NMDA coincidence")
    profile = r.coverage_profile()
    assert "most_covered" in profile
    assert "mean_coverage" in profile
    assert len(profile["most_covered"]) > 0
