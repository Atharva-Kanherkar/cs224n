"""Step 4/6/7: the two embedding matrices, the forward pass, the gradients,
and the sparse update rule.

This file is the heart of the exercise. Read the whole README before
touching backward() — the loss and gradient math are laid out there.
"""

import numpy as np


class SkipGramNegSamplingModel:
    def __init__(self, vocab_size: int, embedding_dim: int, rng: np.random.Generator):
        """Initialize two SEPARATE embedding matrices, each shape
        (vocab_size, embedding_dim):

          self.W_center  — the "input"/center-word embeddings. This is what
                            you'd export as your final word vectors.
          self.W_context — the "output"/context-word embeddings. Used only
                            during training, normally discarded afterward.

        Why two matrices instead of one shared matrix? A word plays two
        different roles in this objective: it's a center word in some
        training pairs and a context word in others. Tying those to the same
        vector creates degenerate optimization dynamics. Word2Vec keeps them
        separate.

        Initialize both with small random values (e.g. rng.normal(scale=0.01,
        size=(vocab_size, embedding_dim))) — not zeros (every word's initial
        gradient would be identical) and not large values (dot products
        would saturate the sigmoid immediately, killing the gradient signal
        before training starts).

        Store vocab_size and embedding_dim on self too; tests and train.py
        may read them.
        """
        raise NotImplementedError("TODO: SkipGramNegSamplingModel.__init__")

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: 1 / (1 + exp(-x)).

        Must not produce nan/inf/overflow warnings for large |x| (try
        x = 1000 and x = -1000). Must work elementwise on arrays as well as
        on Python floats.
        """
        raise NotImplementedError("TODO: sigmoid")

    def forward(self, center_idx: int, context_idx: int, negative_idxs: np.ndarray):
        """Compute the raw (pre-sigmoid) dot-product scores for one training
        example: one positive (center, context) pair plus its k negative
        samples.

            v_c = self.W_center[center_idx]              # shape (dim,)
            v_o = self.W_context[context_idx]             # shape (dim,)
            v_neg = self.W_context[negative_idxs]         # shape (k, dim)

            pos_score = v_c . v_o                          # scalar
            neg_scores = v_neg @ v_c                        # shape (k,)

        Returns:
            (pos_score, neg_scores) — pos_score is a Python float or 0-d
            array, neg_scores is shape (k,). These are RAW scores, not yet
            passed through sigmoid — that happens in loss.py.
        """
        raise NotImplementedError("TODO: forward")

    def backward(
        self,
        center_idx: int,
        context_idx: int,
        negative_idxs: np.ndarray,
        pos_score,
        neg_scores: np.ndarray,
    ):
        """Compute dL/d(v_c), dL/d(v_o), dL/d(v_neg) for the negative log
        likelihood loss (see loss.py / README for the loss formula), given
        the scores from forward().

        This is the actual exercise — derive it yourself with the chain
        rule. You'll need d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        twice: once for the positive term, once for each negative term.

        No closed-form is given here on purpose. tests/test_model.py checks
        your result against a numerical gradient (finite differences), not
        against a hardcoded formula — so there's no way to "match the test"
        without actually getting the calculus right.

        Returns:
            grad_v_c:   shape (dim,)   — gradient w.r.t. W_center[center_idx]
            grad_v_o:   shape (dim,)   — gradient w.r.t. W_context[context_idx]
            grad_v_neg: shape (k, dim) — gradient w.r.t. W_context[negative_idxs],
                        row i is the gradient for negative_idxs[i]
        """
        raise NotImplementedError("TODO: backward")

    def sgd_update(
        self,
        center_idx: int,
        context_idx: int,
        negative_idxs: np.ndarray,
        grad_v_c: np.ndarray,
        grad_v_o: np.ndarray,
        grad_v_neg: np.ndarray,
        lr: float,
    ) -> None:
        """Apply one SGD step in place using the gradients from backward().

        This is the sparse update: out of a (vocab_size, dim) matrix, this
        call must only ever modify:
          - row `center_idx` of self.W_center
          - row `context_idx` and all rows in `negative_idxs` of self.W_context

        Every other row of both matrices must be bit-for-bit unchanged after
        this call returns. tests/test_model.py verifies this directly by
        snapshotting both matrices before the call and diffing every row
        after.

        Gotcha: `negative_idxs` can contain duplicate ids (e.g. the same
        word sampled twice as a negative for this example). If you use plain
        fancy-index assignment like `self.W_context[negative_idxs] -= lr *
        grad_v_neg`, NumPy will NOT accumulate the two updates for a
        repeated index — it silently keeps only one of them. You need the
        two (or more) gradient contributions for a repeated index to add
        together before being applied. Look into `np.add.at`, or accumulate
        the per-row updates yourself before writing them.
        """
        raise NotImplementedError("TODO: sgd_update")
