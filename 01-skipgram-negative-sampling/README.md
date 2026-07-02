# Skip-Gram with Negative Sampling — from scratch

Goal: build the actual training pipeline behind word2vec's Skip-Gram model,
by hand, in NumPy. No autodiff, no frameworks. You derive the gradients,
you write the update rule, you watch the loss go down.

This is not about getting good word vectors — the toy corpus is 25 sentences,
that's not enough data to learn anything meaningful. It's about seeing,
mechanically, how the pieces fit: how a (center word, context word) pair
turns into a scalar loss, and how that loss turns into an update that only
touches a handful of rows in a giant embedding matrix.

Nobody wrote the implementation for you. Every file in `skipgram/` has
function/class signatures, type hints, and docstrings — the *what* and
sometimes the *why*, never the *how*. `tests/` is the thing you're actually
being graded against. Get every test green and you've built a correct
Skip-Gram + Negative Sampling trainer.

## The idea, in one paragraph

Skip-Gram tries to predict context words from a center word. Instead of a
full softmax over the vocabulary (expensive — you'd need a sum over every
word for every single training example), Negative Sampling turns it into a
binary classification problem: "is `context` really a neighbor of `center`,
or is it one of `k` randomly sampled words that mean nothing to `center`?"
You maximize the probability of the real pair and minimize the probability
of the fake (negative) pairs.

## The math you need

Every word gets **two** vectors, not one:

- `v_c` — the word's vector when it's acting as a **center** word (row `c` of `W_center`)
- `v_o` — the word's vector when it's acting as a **context/output** word (row `o` of `W_context`)

For one training example — one positive pair `(center=c, context=o)` plus
`k` sampled negative words `n_1 ... n_k` — the loss is:

```
score_pos   = v_c · v_o
score_neg_i = v_c · v_{n_i}          for i = 1..k

L = -log( sigmoid(score_pos) )  -  sum_i log( sigmoid(-score_neg_i) )
```

Read that as: push `sigmoid(v_c · v_o)` toward 1 (real pairs should score
high), push `sigmoid(v_c · v_{n_i})` toward 0 (fake pairs should score low).

`L` is a scalar. Your job in `model.py`/`loss.py` is to compute it, and in
`backward()` to compute `dL/d(v_c)`, `dL/d(v_o)`, and `dL/d(v_{n_i})` for each
negative sample. Nobody hands you the closed-form derivative here — derive
it yourself with the chain rule (it's two applications of
`d(sigmoid(x))/dx = sigmoid(x)(1-sigmoid(x))`), or come find me and I'll walk
through the derivation with you on a whiteboard, verbally, without touching
your code. Either way, `tests/test_model.py` checks your analytic gradient
against a **numerical gradient** (finite differences) — if they match, your
calculus was right, whether you derived it alone or with help.

## Why "sparse"

`W_center` and `W_context` are each `(vocab_size, embedding_dim)`. For one
training example, you only ever read/write:

- 1 row of `W_center` (the center word)
- `1 + k` rows of `W_context` (the context word + the negatives)

Every other row of both matrices must be **bit-for-bit unchanged** after an
update step. That's the whole point of negative sampling as an optimization:
you never touch the full matrix, so training stays cheap even with a
100,000-word vocabulary. `tests/test_model.py` verifies this directly by
snapshotting the matrices before an update and diffing every row after.

There's a real gotcha hiding here: if your `k` negative samples happen to
include the same word twice, a naive `W_context[negative_idxs] -= lr * grad`
will silently **drop** one of the two gradient contributions instead of
accumulating them (NumPy fancy-index assignment with duplicate indices keeps
only the last write). One of the tests exists specifically to catch this.

## Task list — do these in order

Each step only needs the ones before it. Run the matching test file after
each step; don't move on until it's green.

1. **`skipgram/vocab.py` — `build_vocab`**
   Turn tokenized sentences into `word2idx`, `idx2word`, and per-word
   counts. Everything downstream is indexed by integer id, not string.
   → `pytest tests/test_vocab.py`

2. **`skipgram/dataset.py` — `generate_skipgram_pairs`**
   Slide a window over each sentence and emit `(center_id, context_id)`
   pairs. Don't cross sentence boundaries.
   → `pytest tests/test_dataset.py`

3. **`skipgram/negative_sampling.py`**
   Build the sampling distribution (`count^0.75`, normalized — this is the
   detail that keeps "the" from dominating every negative sample), then
   sample `k` negatives that exclude the true positive.
   → `pytest tests/test_negative_sampling.py`

4. **`skipgram/model.py` — `sigmoid`, `__init__`, `forward`**
   Two embedding matrices, small random init. `forward` computes the raw
   dot-product scores (not yet the loss).

5. **`skipgram/loss.py` — `negative_log_likelihood`**
   Implement the formula above. Watch out for `log(0)` — `sigmoid` can
   saturate to exactly `0.0` or `1.0` in float64 for large `|score|`. You
   need a numerically stable formulation (hint: relate `log(sigmoid(x))` to
   `logaddexp`, or look up the "log-sum-exp trick" / "softplus").
   → `pytest tests/test_loss.py`

6. **`skipgram/model.py` — `backward`**
   Derive and implement the three gradients. No shortcuts — this is the
   actual exercise.
   → `pytest tests/test_model.py::test_gradient_check_matches_numerical`

7. **`skipgram/model.py` — `sgd_update`**
   Apply the update to exactly the right rows. Handle duplicate negative
   indices correctly (accumulate, don't overwrite — look at `np.add.at`).
   → `pytest tests/test_model.py` (the rest of it)

8. **`skipgram/train.py` — `train_one_step`, `train`**
   Wire it all together: sample negatives, forward, loss, backward, update,
   repeat over epochs.
   → `pytest tests/test_train.py`

9. Run everything: `pytest -v` from this directory. Then run `python
   main.py` — it trains on `data/toy_corpus.txt` and prints, for a few
   query words, which other words ended up nearest in vector space. On this
   tiny corpus don't expect magic, but "king" and "queen" landing close, or
   "dog" and "cat" landing close, is a real signal that the mechanism works.

## Running tests

```bash
cd 01-skipgram-negative-sampling
pip install -r requirements.txt
pytest -v                    # everything
pytest tests/test_vocab.py   # one file at a time, in task order
```

## What's already done for you

- `main.py` (orchestration/demo only, not part of the exercise)
- Test files in `tests/`
- The toy corpus in `data/`

## Further reading (search, don't guess links)

- Mikolov et al., 2013, "Distributed Representations of Words and Phrases
  and their Compositionality" — arXiv:1310.4546. This is the actual
  Negative Sampling paper; Section 2.2 has the objective above.
- CS224N lecture notes / Assignment 2 (word2vec) cover the same derivation
  if you want a second explanation of the gradient.
