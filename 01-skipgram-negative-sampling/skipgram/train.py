"""Step 8: wire vocab + pairs + negative sampling + model into an actual
training loop.
"""

import numpy as np

from skipgram.model import SkipGramNegSamplingModel


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
    raise NotImplementedError("TODO: train_one_step")


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
    raise NotImplementedError("TODO: train")
