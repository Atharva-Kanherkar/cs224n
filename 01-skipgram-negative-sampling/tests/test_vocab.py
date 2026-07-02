from skipgram.vocab import build_vocab


def test_build_vocab_basic():
    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["a", "cat", "ran"],
    ]
    vocab = build_vocab(sentences, min_count=1)

    assert set(vocab.word2idx.keys()) == {"the", "cat", "sat", "dog", "a", "ran"}
    assert len(vocab.idx2word) == len(vocab.word2idx)
    assert len(vocab.counts) == len(vocab.word2idx)

    for word, idx in vocab.word2idx.items():
        assert vocab.idx2word[idx] == word

    assert vocab.counts[vocab.word2idx["the"]] == 2
    assert vocab.counts[vocab.word2idx["cat"]] == 2
    assert vocab.counts[vocab.word2idx["sat"]] == 2
    assert vocab.counts[vocab.word2idx["dog"]] == 1
    assert vocab.counts[vocab.word2idx["a"]] == 1
    assert vocab.counts[vocab.word2idx["ran"]] == 1


def test_build_vocab_min_count_filters_rare_words():
    sentences = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["the", "rare", "word"],
    ]
    vocab = build_vocab(sentences, min_count=2)

    assert "the" in vocab.word2idx
    assert "sat" in vocab.word2idx
    assert "cat" not in vocab.word2idx
    assert "dog" not in vocab.word2idx
    assert "rare" not in vocab.word2idx
    assert "word" not in vocab.word2idx


def test_build_vocab_empty_input():
    vocab = build_vocab([], min_count=1)
    assert vocab.word2idx == {}
    assert vocab.idx2word == []
    assert len(vocab.counts) == 0
