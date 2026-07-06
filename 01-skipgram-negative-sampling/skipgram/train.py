"""Step 8: wire vocab + pairs + negative sampling + model into an actual
training loop.
"""

import numpy as np

from skipgram.loss import negative_log_likelihood
from skipgram.model import SkipGramNegSamplingModel
from skipgram.negative_sampling import sample_negatives


def train_one_step(
    model: SkipGramNegSamplingModel,
    center_idx: int,
    context_idx: int,
    negative_idxs: np.ndarray,
    lr: float,
) -> float:
    """Run one full training example through the model: forward -> loss ->
    backward -> sgd_update. This is a thin composition of methods you've
    already implemented on `model` and the loss function — no new math here.

    Returns: the scalar loss (float) for this example, computed BEFORE the
    update is applied (i.e. the loss at the parameters you started this
    call with).
    """

    pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)
    loss = negative_log_likelihood(pos_score, neg_scores)

    grad_v_c, grad_v_o, grad_v_neg = model.backward(
      center_idx,
      context_idx,
      negative_idxs,
      pos_score,
      neg_scores,
    )
    model.sgd_update(
      center_idx,
      context_idx,
      negative_idxs,
      grad_v_c,
      grad_v_o,
      grad_v_neg,
      lr,
    )

    return float(loss)

def train(
    pairs: list[tuple[int, int]],
    distribution: np.ndarray,
    model: SkipGramNegSamplingModel,
    k_negatives: int,
    epochs: int,
    lr: float,
    seed: int = 42,
) -> list[float]:
    """Train `model` on `pairs` for `epochs` passes.

    Each epoch:
      - shuffle `pairs` (use a np.random.Generator seeded from `seed`,
        re-derived deterministically per call so results are reproducible)
      - for each (center_idx, context_idx) pair:
          - sample k_negatives negatives from `distribution`, excluding
            context_idx
          - call train_one_step(...)
      - record the average loss over all pairs seen this epoch

    Returns:
        A list of length `epochs`, the average loss per epoch. This should
        trend downward if everything above is implemented correctly —
        tests/test_train.py checks exactly that on a small end-to-end run.
    """
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []

    for _ in range(epochs):
      shuffled_pairs = list(pairs)
      rng.shuffle(shuffled_pairs)

      total_loss = 0.0
      for center_idx, context_idx in shuffled_pairs:
        negative_idxs = sample_negatives(
          distribution=distribution,
          positive_idx=context_idx,
          k=k_negatives,
          rng=rng,
        )
        total_loss += train_one_step(
          model,
          center_idx,
          context_idx,
          negative_idxs,
          lr,
        )

      epoch_losses.append(total_loss / len(shuffled_pairs))

    return epoch_losses
