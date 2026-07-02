import numpy as np

from skipgram.loss import negative_log_likelihood
from skipgram.model import SkipGramNegSamplingModel


def test_sigmoid_basic_properties():
    assert np.isclose(SkipGramNegSamplingModel.sigmoid(0.0), 0.5)

    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = SkipGramNegSamplingModel.sigmoid(x)

    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    # sigmoid must be monotonically increasing
    assert np.all(np.diff(y) >= 0)


def test_init_produces_two_distinct_small_matrices():
    rng = np.random.default_rng(0)
    model = SkipGramNegSamplingModel(vocab_size=15, embedding_dim=6, rng=rng)

    assert model.W_center.shape == (15, 6)
    assert model.W_context.shape == (15, 6)
    assert not np.array_equal(model.W_center, model.W_context)
    # small init: rules out zeros-init and rules out large-scale init
    assert np.all(np.abs(model.W_center) < 1.0)
    assert not np.all(model.W_center == 0)


def test_forward_matches_manual_dot_products():
    rng = np.random.default_rng(1)
    model = SkipGramNegSamplingModel(vocab_size=10, embedding_dim=4, rng=rng)
    center_idx, context_idx = 2, 7
    negative_idxs = np.array([0, 3, 9])

    pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)

    v_c = model.W_center[center_idx]
    expected_pos = v_c @ model.W_context[context_idx]
    expected_neg = model.W_context[negative_idxs] @ v_c

    assert np.isclose(float(pos_score), expected_pos)
    assert np.allclose(neg_scores, expected_neg)
    assert neg_scores.shape == (3,)


def test_gradient_check_matches_numerical():
    """The core correctness check for backward(). No hardcoded formula here:
    your analytic gradient must match a finite-difference numerical gradient
    computed directly on the model's parameters.
    """
    rng = np.random.default_rng(7)
    model = SkipGramNegSamplingModel(vocab_size=12, embedding_dim=5, rng=rng)
    center_idx, context_idx = 3, 8
    negative_idxs = np.array([1, 5, 11])
    eps = 1e-5

    def loss_at_current_params():
        pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)
        return negative_log_likelihood(pos_score, neg_scores)

    pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)
    grad_v_c, grad_v_o, grad_v_neg = model.backward(
        center_idx, context_idx, negative_idxs, pos_score, neg_scores
    )

    assert grad_v_c.shape == (5,)
    assert grad_v_o.shape == (5,)
    assert grad_v_neg.shape == (3, 5)

    def numerical_grad(matrix, row_idx, analytic_row):
        numerical = np.zeros_like(analytic_row)
        for d in range(matrix.shape[1]):
            original = matrix[row_idx, d]
            matrix[row_idx, d] = original + eps
            loss_plus = loss_at_current_params()
            matrix[row_idx, d] = original - eps
            loss_minus = loss_at_current_params()
            matrix[row_idx, d] = original
            numerical[d] = (loss_plus - loss_minus) / (2 * eps)
        return numerical

    numerical_v_c = numerical_grad(model.W_center, center_idx, grad_v_c)
    assert np.allclose(grad_v_c, numerical_v_c, atol=1e-4), (
        f"grad_v_c mismatch: analytic={grad_v_c}, numerical={numerical_v_c}"
    )

    numerical_v_o = numerical_grad(model.W_context, context_idx, grad_v_o)
    assert np.allclose(grad_v_o, numerical_v_o, atol=1e-4), (
        f"grad_v_o mismatch: analytic={grad_v_o}, numerical={numerical_v_o}"
    )

    for i, neg_idx in enumerate(negative_idxs):
        numerical_v_neg_i = numerical_grad(model.W_context, neg_idx, grad_v_neg[i])
        assert np.allclose(grad_v_neg[i], numerical_v_neg_i, atol=1e-4), (
            f"grad_v_neg[{i}] mismatch: analytic={grad_v_neg[i]}, numerical={numerical_v_neg_i}"
        )


def test_sgd_update_only_touches_relevant_rows():
    rng = np.random.default_rng(3)
    model = SkipGramNegSamplingModel(vocab_size=20, embedding_dim=8, rng=rng)
    W_center_before = model.W_center.copy()
    W_context_before = model.W_context.copy()

    center_idx, context_idx = 4, 9
    negative_idxs = np.array([2, 11, 15])

    pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)
    grad_v_c, grad_v_o, grad_v_neg = model.backward(
        center_idx, context_idx, negative_idxs, pos_score, neg_scores
    )
    model.sgd_update(center_idx, context_idx, negative_idxs, grad_v_c, grad_v_o, grad_v_neg, lr=0.1)

    touched_context_rows = set(negative_idxs.tolist()) | {context_idx}

    for i in range(20):
        if i == center_idx:
            assert not np.array_equal(model.W_center[i], W_center_before[i]), (
                "the center word's row should have been updated"
            )
        else:
            assert np.array_equal(model.W_center[i], W_center_before[i]), (
                f"row {i} of W_center should be untouched by this update"
            )

        if i in touched_context_rows:
            assert not np.array_equal(model.W_context[i], W_context_before[i]), (
                f"row {i} of W_context (context/negative) should have been updated"
            )
        else:
            assert np.array_equal(model.W_context[i], W_context_before[i]), (
                f"row {i} of W_context should be untouched by this update"
            )


def test_sgd_update_accumulates_duplicate_negative_indices():
    """If the same word is sampled as a negative twice in one step, both
    gradient contributions must be applied — a naive fancy-index assignment
    silently drops one of them.
    """
    rng = np.random.default_rng(4)
    model = SkipGramNegSamplingModel(vocab_size=10, embedding_dim=4, rng=rng)
    center_idx, context_idx = 0, 1
    negative_idxs = np.array([3, 3, 5])  # word 3 sampled twice on purpose

    pos_score, neg_scores = model.forward(center_idx, context_idx, negative_idxs)
    grad_v_c, grad_v_o, grad_v_neg = model.backward(
        center_idx, context_idx, negative_idxs, pos_score, neg_scores
    )

    W_context_before = model.W_context.copy()
    lr = 0.5
    model.sgd_update(center_idx, context_idx, negative_idxs, grad_v_c, grad_v_o, grad_v_neg, lr)

    expected_row_3 = W_context_before[3] - lr * (grad_v_neg[0] + grad_v_neg[1])
    expected_row_5 = W_context_before[5] - lr * grad_v_neg[2]

    assert np.allclose(model.W_context[3], expected_row_3), (
        "duplicate negative-sample indices must have their gradients accumulated, not overwritten"
    )
    assert np.allclose(model.W_context[5], expected_row_5)
