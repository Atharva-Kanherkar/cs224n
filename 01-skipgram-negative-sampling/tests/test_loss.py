import numpy as np

from skipgram.loss import negative_log_likelihood


def test_known_value_at_zero_scores():
    # sigmoid(0) == 0.5 for every term, so:
    # L = -log(0.5) - log(0.5) - log(0.5) = -3 * log(0.5)
    pos_score = 0.0
    neg_scores = np.array([0.0, 0.0])

    loss = negative_log_likelihood(pos_score, neg_scores)

    expected = -3 * np.log(0.5)
    assert np.isclose(loss, expected, atol=1e-6)


def test_loss_is_nonnegative():
    rng = np.random.default_rng(2)
    for _ in range(30):
        pos_score = float(rng.normal(scale=5))
        neg_scores = rng.normal(scale=5, size=5)
        loss = negative_log_likelihood(pos_score, neg_scores)
        assert loss >= 0


def test_loss_rewards_correct_direction():
    # a confidently-correct positive (high score) and confidently-correct
    # negatives (very negative score) should give a much smaller loss than
    # a confidently-WRONG positive and confidently-wrong negatives
    good_loss = negative_log_likelihood(10.0, np.array([-10.0, -10.0]))
    bad_loss = negative_log_likelihood(-10.0, np.array([10.0, 10.0]))
    assert good_loss < bad_loss
    assert good_loss < 1e-3


def test_loss_is_numerically_stable_for_extreme_scores():
    loss_a = negative_log_likelihood(1000.0, np.array([-1000.0, -1000.0]))
    loss_b = negative_log_likelihood(-1000.0, np.array([1000.0]))

    assert np.isfinite(loss_a)
    assert np.isfinite(loss_b)
