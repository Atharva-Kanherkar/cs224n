"""Step 1: turn raw tokenized text into an integer-indexed vocabulary.

Everything downstream (pair generation, negative sampling, the embedding
matrices themselves) is indexed by integer word id, not by string. This is
the foundation the rest of the pipeline is built on.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Vocab:
    """Container for a built vocabulary. This class is complete — you don't
    need to change it, just return one from build_vocab().

    word2idx: maps word string -> integer id
    idx2word: maps integer id -> word string (idx2word[word2idx[w]] == w)
    counts:   counts[i] is the true corpus frequency of idx2word[i] (words
              dropped by min_count simply have no entry anywhere in this
              object — the survivors' counts are unaffected by the drop)
    """

    word2idx: dict
    idx2word: list
    counts: np.ndarray


def build_vocab(tokenized_sentences: list[list[str]], min_count: int = 1) -> Vocab:
    """Build a Vocab from a list of tokenized sentences.

    tokenized_sentences: e.g. [["the", "king", "ruled"], ["the", "queen", ...], ...]
    min_count: words appearing fewer than `min_count` times across the whole
        corpus are dropped entirely (no UNK token for this exercise — just
        omit them from word2idx/idx2word as if they never existed).

    Requirements:
      - word2idx and idx2word must be consistent: idx2word[word2idx[w]] == w
        for every w in word2idx.
      - counts must be aligned with idx2word: counts[i] is the true corpus
        frequency of idx2word[i] (count first, then filter out words below
        min_count — don't let filtering change the surviving words' counts).
      - Iteration order isn't tested, but it's good practice to make id
        assignment deterministic (e.g. sort by first-seen order or by word).

    Returns:
        Vocab(word2idx, idx2word, counts)
    """
    raise NotImplementedError("TODO: build_vocab")
