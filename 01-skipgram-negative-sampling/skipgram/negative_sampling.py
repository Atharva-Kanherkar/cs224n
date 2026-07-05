"""Step 3: the negative sampling distribution and sampler.

Negative Sampling replaces a full softmax over the vocabulary with: "is this
context word real, or is it one of k words drawn at random?" The catch is
that "at random" doesn't mean uniform, and it doesn't mean proportional to
raw frequency either — see build_negative_sampling_distribution below.
"""

import numpy as np


def build_negative_sampling_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Turn raw word counts into a sampling distribution over the vocabulary.

    Word2Vec raises each count to the 0.75 power before normalizing:

        P(w) = count(w)^power / sum_j count(j)^power

    Why not just sample proportional to raw frequency? Because a handful of
    words ("the", "a", "of", ...) dominate raw frequency counts, and if
    negatives are drawn proportional to raw frequency, nearly every negative
    sample would be one of those few words — not a useful training signal.
    Raising to the 0.75 power flattens the distribution: very frequent words
    get sampled less often than their raw frequency would suggest, and rare
    words get sampled more often than their raw frequency would suggest,
    relative to a uniform baseline.

    counts: shape (vocab_size,), aligned with a Vocab's idx2word/word2idx.
    Returns: shape (vocab_size,) array of probabilities summing to 1.0.
    """
    powered = counts ** power 
    return powered/powered.sum()




def sample_negatives(
    distribution: np.ndarray,
    positive_idx: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw k negative sample word ids from `distribution`.

    The true positive context word (positive_idx) must never appear in the
    output — if you happen to sample it, resample that slot. Duplicate
    negative ids among the k draws (other than positive_idx) are fine and
    expected; you are not required to deduplicate them, and downstream code
    (sgd_update) needs to handle duplicates correctly, not avoid them.

    rng: use this np.random.Generator for all randomness here (so callers
        can control reproducibility). Don't call np.random.* module-level
        functions or create your own Generator inside this function.

    Returns: integer array of shape (k,).
    """
    negatives = []
    while(len(negatives)) < k:
        candidate = rng.choice(len(distribution), p=distribution)
        if candidate == positive_idx:
            continue
        negatives.append(candidate)
       
    return np.array(negatives)
    
