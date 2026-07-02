import numpy as np

from skipgram.dataset import generate_skipgram_pairs
from skipgram.model import SkipGramNegSamplingModel
from skipgram.negative_sampling import build_negative_sampling_distribution
from skipgram.train import train, train_one_step
from skipgram.vocab import build_vocab


def test_train_one_step_returns_finite_loss_and_updates_model():
    rng = np.random.default_rng(5)
    model = SkipGramNegSamplingModel(vocab_size=10, embedding_dim=4, rng=rng)
    W_center_before = model.W_center.copy()

    loss = train_one_step(model, center_idx=2, context_idx=6, negative_idxs=np.array([1, 8]), lr=0.1)

    assert isinstance(loss, float) or np.isscalar(loss)
    assert np.isfinite(loss)
    assert loss > 0
    assert not np.array_equal(model.W_center[2], W_center_before[2])


def test_loss_decreases_over_training_on_toy_corpus():
    sentences = [
        "the quick brown fox jumps over the lazy dog".split(),
        "the dog barks at the fox".split(),
        "the fox runs from the dog".split(),
        "the dog and the fox are enemies".split(),
    ] * 8

    vocab = build_vocab(sentences, min_count=1)
    pairs = generate_skipgram_pairs(sentences, vocab.word2idx, window_size=2)
    distribution = build_negative_sampling_distribution(vocab.counts)

    rng = np.random.default_rng(42)
    model = SkipGramNegSamplingModel(len(vocab.idx2word), embedding_dim=10, rng=rng)

    losses = train(pairs, distribution, model, k_negatives=5, epochs=40, lr=0.05, seed=42)

    assert len(losses) == 40
    assert all(np.isfinite(l) for l in losses)
    # a real training loop should cut the loss substantially over 40 epochs
    # on a corpus this small and repetitive
    assert losses[-1] < losses[0] * 0.7, f"loss barely moved: {losses[0]} -> {losses[-1]}"
