"""Step 5: the negative log likelihood loss for Skip-Gram with Negative
Sampling.
"""

import numpy as np


def negative_log_likelihood(pos_score, neg_scores: np.ndarray) -> float:
    """Given the RAW dot-product scores (not yet passed through sigmoid)
    for the positive pair and the k negative samples, compute:

        L = -log( sigmoid(pos_score) )
            - sum_i log( sigmoid(-neg_scores[i]) )

    Intuition: sigmoid(pos_score) should be close to 1 (real pair), so
    -log(...) of it should be close to 0 when training is going well.
    sigmoid(-neg_scores[i]) should also be close to 1 (i.e. sigmoid(neg_scores[i])
    close to 0 — negatives should score low), for the same reason.

    Numerical stability: for large |score|, sigmoid(score) can round to
    exactly 0.0 or 1.0 in float64, and log(0.0) is -inf, which will poison
    training. Don't compute sigmoid(x) and then log(...) of it naively.
    Instead express log(sigmoid(x)) in a form that never takes log of
    something that can underflow to zero — look up "log-sigmoid" or
    "softplus" (np.logaddexp is useful here). Verify this yourself with
    pos_score = 1000.0 or -1000.0 before you consider this done.

    Returns: a single float (the loss for this one training example).
    """
    pos_term= -np.logaddexp(0,-pos_score)
    neg_term= -np.logaddexp(0, neg_scores)

    loss = -pos_term -sum(neg_term)
    return loss