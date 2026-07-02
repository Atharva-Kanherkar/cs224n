from skipgram.dataset import generate_skipgram_pairs


def test_generate_pairs_window_1():
    sentences = [["the", "quick", "brown", "fox"]]
    word2idx = {"the": 0, "quick": 1, "brown": 2, "fox": 3}

    pairs = generate_skipgram_pairs(sentences, word2idx, window_size=1)

    expected = {
        (0, 1),  # the -> quick
        (1, 0),  # quick -> the
        (1, 2),  # quick -> brown
        (2, 1),  # brown -> quick
        (2, 3),  # brown -> fox
        (3, 2),  # fox -> brown
    }
    assert set(pairs) == expected
    assert len(pairs) == len(expected)


def test_generate_pairs_window_2_includes_farther_context():
    sentences = [["a", "b", "c", "d", "e"]]
    word2idx = {w: i for i, w in enumerate("abcde")}

    pairs = generate_skipgram_pairs(sentences, word2idx, window_size=2)

    # center "c" (idx 2) should see a,b,d,e as context (all within distance 2)
    c_contexts = {ctx for center, ctx in pairs if center == word2idx["c"]}
    assert c_contexts == {word2idx["a"], word2idx["b"], word2idx["d"], word2idx["e"]}


def test_generate_pairs_respects_sentence_boundaries():
    sentences = [["a", "b"], ["c", "d"]]
    word2idx = {"a": 0, "b": 1, "c": 2, "d": 3}

    pairs = generate_skipgram_pairs(sentences, word2idx, window_size=5)

    first_sentence_ids = {0, 1}
    second_sentence_ids = {2, 3}
    for center, context in pairs:
        crosses = (center in first_sentence_ids and context in second_sentence_ids) or (
            center in second_sentence_ids and context in first_sentence_ids
        )
        assert not crosses, f"pair ({center}, {context}) crosses a sentence boundary"


def test_generate_pairs_skips_words_not_in_vocab():
    sentences = [["the", "rare", "cat"]]
    word2idx = {"the": 0, "cat": 1}  # "rare" was dropped by min_count upstream

    pairs = generate_skipgram_pairs(sentences, word2idx, window_size=2)

    for center, context in pairs:
        assert center in word2idx.values()
        assert context in word2idx.values()
    assert (0, 1) in pairs
    assert (1, 0) in pairs
