"""Demo/orchestration script — not part of the exercise, nothing to implement
here. This just wires together the modules you built and shows the payoff:
training on the toy corpus and inspecting which words ended up nearby in
vector space.

Run with: python main.py
"""

import numpy as np

from skipgram.dataset import generate_skipgram_pairs
from skipgram.model import SkipGramNegSamplingModel
from skipgram.negative_sampling import build_negative_sampling_distribution
from skipgram.train import train
from skipgram.vocab import build_vocab

CORPUS_PATH = "data/toy_corpus.txt"
EMBEDDING_DIM = 20
WINDOW_SIZE = 2
K_NEGATIVES = 8
EPOCHS = 200
LEARNING_RATE = 0.05
QUERY_WORDS = ["king", "queen", "dog", "cat", "paris", "france"]
TOP_N = 5


def load_sentences(path: str) -> list[list[str]]:
    sentences = []
    with open(path) as f:
        for line in f:
            tokens = line.strip().lower().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def nearest_neighbors(word, vocab, embeddings, top_n=TOP_N):
    if word not in vocab.word2idx:
        return []
    idx = vocab.word2idx[word]
    query_vec = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-9
    sims = (embeddings @ query_vec) / norms
    ranked = np.argsort(-sims)
    results = []
    for other_idx in ranked:
        if other_idx == idx:
            continue
        results.append((vocab.idx2word[other_idx], float(sims[other_idx])))
        if len(results) == top_n:
            break
    return results


def main():
    sentences = load_sentences(CORPUS_PATH)
    vocab = build_vocab(sentences, min_count=1)
    pairs = generate_skipgram_pairs(sentences, vocab.word2idx, WINDOW_SIZE)
    distribution = build_negative_sampling_distribution(vocab.counts)

    print(f"vocab size: {len(vocab.idx2word)}")
    print(f"training pairs: {len(pairs)}")

    rng = np.random.default_rng(42)
    model = SkipGramNegSamplingModel(len(vocab.idx2word), EMBEDDING_DIM, rng)

    losses = train(pairs, distribution, model, K_NEGATIVES, EPOCHS, LEARNING_RATE)

    print(f"loss: epoch 0 = {losses[0]:.4f}, epoch {EPOCHS - 1} = {losses[-1]:.4f}")
    print()
    print("nearest neighbors (by cosine similarity on W_center):")
    for word in QUERY_WORDS:
        neighbors = nearest_neighbors(word, vocab, model.W_center)
        if not neighbors:
            print(f"  {word!r}: not in vocab")
            continue
        formatted = ", ".join(f"{w} ({s:.2f})" for w, s in neighbors)
        print(f"  {word:10s} -> {formatted}")


if __name__ == "__main__":
    main()
