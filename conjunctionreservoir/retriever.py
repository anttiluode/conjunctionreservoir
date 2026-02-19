"""
ConjunctionReservoir — core retriever
"""

import numpy as np
import re
import time
from typing import Dict, List, Optional, Tuple, Union


def split_sentences(text: str, min_len: int = 15) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) >= min_len]


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    sections = re.split(r"\n(?=From:|Subject:|Date:|---)", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            for i in range(0, len(section), chunk_size - overlap):
                chunk = section[i : i + chunk_size].strip()
                if len(chunk) > 50:
                    chunks.append(chunk)
    return chunks


def build_vocab(texts: List[str], max_vocab: int = 2000) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"\b[a-zA-Z]{2,}\b", t.lower()):
            counts[w] = counts.get(w, 0) + 1
    return {
        word: idx
        for idx, (word, _) in enumerate(
            sorted(counts.items(), key=lambda x: -x[1])[:max_vocab]
        )
    }


def tfidf_weights(sentences: List[str], vocab: Dict[str, int]) -> np.ndarray:
    n = len(sentences)
    df = np.zeros(len(vocab))
    for s in sentences:
        for w in set(re.findall(r"\b[a-zA-Z]{2,}\b", s.lower())):
            if w in vocab:
                df[vocab[w]] += 1
    return np.log((n + 1) / (df + 1)) + 1.0


def encode_text(text: str, vocab: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    vec = np.zeros(len(vocab))
    for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()):
        if w in vocab:
            vec[vocab[w]] += 1.0
    vec *= idf
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


class ConjunctionReservoir:
    """
    Document retriever with sentence-level conjunction scoring.

    Instead of asking "do query terms appear anywhere in this chunk?",
    asks "do query terms appear in the SAME SENTENCE?"

    Biological grounding
    --------------------
    - Norman-Haignere et al. (2025): auditory cortex integration windows
      are time-yoked (~80ms), not structure-yoked. The sentence is the
      text analog of this fixed window.
    - NMDA receptor logic: both inputs must arrive simultaneously or the
      gate stays closed. conjunction_threshold implements this.
    - Vollan et al. (2025): coverage-maximizing theta sweep. After visiting
      a region the system tends away from it (coverage_decay trace).
    - Deerskin Hypothesis (Luode 2024): coverage² weighting = geometric
      penalty for partial matches, like a real coincidence detector.

    Parameters
    ----------
    conjunction_threshold : float
        Fraction of query tokens (≥3 chars) that must co-appear in a
        sentence for it to contribute. 0.5 = half (default). 1.0 = all.
    coverage_decay : float
        Decay rate for the Vollan coverage trace. 0.04 default.
    max_vocab : int
        Maximum vocabulary size.

    Example
    -------
    >>> r = ConjunctionReservoir()
    >>> r.build_index(open("notes.txt").read())
    >>> for chunk, score in r.retrieve("NMDA coincidence detection"):
    ...     print(f"[{score:.4f}]  {chunk[:80]}")
    """

    def __init__(
        self,
        conjunction_threshold: float = 0.5,
        coverage_decay: float = 0.04,
        hebbian_lr: float = 0.01,
        max_vocab: int = 2000,
    ) -> None:
        self.conjunction_threshold = conjunction_threshold
        self.coverage_decay = coverage_decay
        self.hebbian_lr = hebbian_lr
        self.max_vocab = max_vocab

        self.vocab: Optional[Dict[str, int]] = None
        self.idf: Optional[np.ndarray] = None
        self.chunk_texts: List[str] = []
        self.all_sentences: List[str] = []
        self.sentence_to_chunk: List[int] = []
        self.sent_feats: Optional[np.ndarray] = None
        self.chunk_feats: Optional[np.ndarray] = None
        self.sentence_coverage: Optional[np.ndarray] = None
        self.n_queries: int = 0
        self.index_time: float = 0.0

    def build_index(
        self,
        text_or_chunks: Union[str, List[str]],
        verbose: bool = True,
    ) -> "ConjunctionReservoir":
        """
        Index a document or list of pre-split chunks.

        Parameters
        ----------
        text_or_chunks : str or list[str]
        verbose : bool

        Returns self.
        """
        t0 = time.perf_counter()

        if isinstance(text_or_chunks, str):
            if verbose:
                print(f"  Chunking ({len(text_or_chunks):,} chars)...")
            self.chunk_texts = chunk_document(text_or_chunks)
        else:
            self.chunk_texts = list(text_or_chunks)

        if not self.chunk_texts:
            raise ValueError("No chunks found.")

        if verbose:
            print(f"  Chunks: {len(self.chunk_texts)}")

        self.all_sentences = []
        self.sentence_to_chunk = []
        for chunk_idx, chunk in enumerate(self.chunk_texts):
            for s in split_sentences(chunk):
                self.all_sentences.append(s)
                self.sentence_to_chunk.append(chunk_idx)

        if not self.all_sentences:
            raise ValueError("No sentences extracted.")

        self.vocab = build_vocab(
            self.all_sentences + self.chunk_texts, max_vocab=self.max_vocab
        )
        self.idf = tfidf_weights(self.all_sentences, self.vocab)

        if verbose:
            avg = len(self.all_sentences) / len(self.chunk_texts)
            print(f"  Sentences: {len(self.all_sentences)} ({avg:.1f}/chunk) | Vocab: {len(self.vocab)}")

        self.sent_feats = np.array(
            [encode_text(s, self.vocab, self.idf) for s in self.all_sentences]
        )
        self.chunk_feats = np.array(
            [encode_text(c, self.vocab, self.idf) for c in self.chunk_texts]
        )
        self.sentence_coverage = np.zeros(len(self.all_sentences))

        self.index_time = time.perf_counter() - t0
        if verbose:
            print(f"  Index built in {self.index_time * 1000:.1f}ms")

        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        update_coverage: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the top_k most relevant chunks.

        Scoring per sentence s:
          token_coverage = overlap(query_tokens, sentence_tokens) / |query_tokens|
          if token_coverage < threshold: score = 0  [hard gate]
          else: score = tfidf_sim * coverage² * vollan_weight

        Chunk score = max sentence score. Falls back to TF-IDF if no
        sentence passes the gate.
        """
        if self.vocab is None:
            raise RuntimeError("Call build_index() before retrieve().")

        q_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        q_feat = encode_text(query, self.vocab, self.idf)
        sent_scores = np.zeros(len(self.all_sentences))

        for s_idx, sentence in enumerate(self.all_sentences):
            s_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower()))
            matches = sum(
                1 for qt in q_tokens
                if any(qt in st or st in qt for st in s_tokens)
            )
            token_coverage = matches / len(q_tokens) if q_tokens else 0.0

            if token_coverage < self.conjunction_threshold:
                continue

            tfidf_sim = float(self.sent_feats[s_idx] @ q_feat)
            conj_weight = token_coverage ** 2
            vollan_w = 1.0 / (1.0 + self.sentence_coverage[s_idx])
            sent_scores[s_idx] = tfidf_sim * conj_weight * vollan_w

        chunk_scores = np.zeros(len(self.chunk_texts))
        for s_idx, (score, chunk_idx) in enumerate(zip(sent_scores, self.sentence_to_chunk)):
            if score > chunk_scores[chunk_idx]:
                chunk_scores[chunk_idx] = score

        if chunk_scores.max() == 0.0:
            chunk_scores = self.chunk_feats @ q_feat

        top_idx = chunk_scores.argsort()[-top_k:][::-1]
        results = [(self.chunk_texts[i], float(chunk_scores[i])) for i in top_idx]

        if update_coverage and sent_scores.max() > 0.0:
            norm = sent_scores / (sent_scores.max() + 1e-8)
            self.sentence_coverage = (
                self.sentence_coverage * (1.0 - self.coverage_decay) + norm
            )
            self.n_queries += 1

        return results

    def summary(self) -> Dict:
        return {
            "n_chunks": len(self.chunk_texts),
            "n_sentences": len(self.all_sentences),
            "avg_sentences_per_chunk": round(
                len(self.all_sentences) / max(1, len(self.chunk_texts)), 2
            ),
            "vocab_size": len(self.vocab) if self.vocab else 0,
            "conjunction_threshold": self.conjunction_threshold,
            "coverage_decay": self.coverage_decay,
            "n_queries": self.n_queries,
            "index_time_ms": round(self.index_time * 1000, 1),
        }

    def coverage_profile(self) -> Dict:
        if self.sentence_coverage is None:
            return {}
        top_idx = self.sentence_coverage.argsort()[-10:][::-1]
        return {
            "most_covered": [
                (self.all_sentences[i], round(float(self.sentence_coverage[i]), 4))
                for i in top_idx
                if self.sentence_coverage[i] > 0
            ],
            "mean_coverage": round(float(self.sentence_coverage.mean()), 6),
            "n_queries": self.n_queries,
        }
