import numpy as np

from skipgram.negative_sampling import (
    build_negative_sampling_distribution,
    sample_negatives,
)


def test_distribution_sums_to_one_and_matches_power_law():
    counts = np.array([100.0, 10.0, 1.0])
    dist = build_negative_sampling_distribution(counts, power=0.75)

    assert dist.shape == (3,)
    assert np.isclose(dist.sum(), 1.0)

    unnormalized = counts ** 0.75
    expected = unnormalized / unnormalized.sum()
    assert np.allclose(dist, expected)


def test_distribution_flattens_relative_to_raw_frequency():
    counts = np.array([1000.0, 1.0])
    dist = build_negative_sampling_distribution(counts, power=0.75)
    raw = counts / counts.sum()

    # the dominant word's share under the 0.75-power distribution must be
    # smaller than its share under raw frequency
    assert dist[0] < raw[0]
    assert dist[1] > raw[1]


def test_sample_negatives_shape_and_excludes_positive():
    rng = np.random.default_rng(0)
    dist = np.array([0.98, 0.01, 0.01])

    samples = sample_negatives(dist, positive_idx=0, k=200, rng=rng)

    assert samples.shape == (200,)
    assert not np.any(samples == 0)


def test_sample_negatives_roughly_follows_distribution():
    rng = np.random.default_rng(1)
    dist = np.array([0.0, 0.9, 0.1])

    samples = sample_negatives(dist, positive_idx=0, k=3000, rng=rng)

    frac_word_1 = np.mean(samples == 1)
    assert 0.75 < frac_word_1 < 1.0
