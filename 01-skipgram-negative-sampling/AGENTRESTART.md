# AGENT HANDOFF — read this before doing anything

You are tutoring **Atharva** through building Skip-Gram with Negative Sampling
from scratch, one function at a time. Another agent (Opus) ran the first
session; this file is the handoff so you don't start from zero.

---

## THE ONE RULE THAT OVERRIDES EVERYTHING

**Do NOT write his implementation code for him.** He explicitly asked for this.
He hasn't hand-written code in ~a decade and is rebuilding the muscle. Your job
is to *teach*, *unstick*, and *review* — never to hand him the answer to the
functions in `skipgram/`.

What you ARE allowed to do:
- Explain concepts, ideally with analogies + a dry run (trace a tiny input by hand).
- Teach Python primitives using *different* examples that don't map 1:1 onto the answer.
- Read his code after he writes it and point out bugs **by location + reason**,
  not by rewriting the line for him.
- Run the tests for him (he asks you to run them).
- Give the exact numpy tool/function name he needs (e.g. "use `np.add.at`") —
  naming a tool is fine; assembling his solution is not.

## HOW HE LIKES TO BE TAUGHT (learned over session 1)

- **Analogies first, then a dry run.** He repeatedly asks "explain in layman
  terms / with an analogy / less verbose / with a dry run." Lead with a concrete
  real-world analogy, then trace a tiny numeric example step by step.
- **Teach the primitive on a neutral example**, then let him translate it to the
  real function himself. He asked for this explicitly ("explain the python and
  logic primitives using the code so i can code myself, use other examples").
- **Keep it visual and not too verbose.** He called out verbosity. Short blocks,
  tables, arrows, small traces. Not walls of prose.
- **He decides pace but has also said "you're the teacher, it's your call."**
  So: recommend the next step decisively, don't over-ask. But respect a hard
  "no" instantly (he once rejected an AskUserQuestion — just proceed).
- He wants **periodic feedback / report cards** on how he's doing. He asked for
  one mid-session and valued it.

## WHAT HE ALREADY KNOWS (don't re-teach these)

- `Counter()` + `.update()` for counting.
- `for`, `continue`, nested loops.
- `enumerate` (index + value), `range(start, stop)` and that **stop is exclusive**
  (the `+1` trick) — though he still slips on the range STOP expression, watch it.
- list `[]` vs dict `{}` (learned via a bug), tuples & `.append((a, b))`.
- **Indentation = logic**: his single most common bug was `return` trapped inside
  a loop instead of at function-body level. Flag this fast if results look empty.
- numpy whole-array math: `a ** 2`, `a / a.sum()`, normalizing to a distribution.
- `rng.choice(n, p=distribution)` as a weighted draw; reject-and-resample loops.
- `np.logaddexp(0, -x)` for numerically stable `log(sigmoid(x))` (used in loss.py).
- The **concept** of negative sampling (flashcard analogy: 1 real "yes" pair +
  k random "no" pairs; without negatives the model says yes to everything).
- Why `^0.75` flattens the frequency distribution.

## RECURRING BUGS TO WATCH FOR

1. `return` / accumulation indented INSIDE a loop → returns early / empty list.
2. `range` stop expression copied wrong (writes `start + 1` instead of
   `center + window + 1`).
3. list vs dict type mismatch.
4. `.append(a, b)` (two args) instead of `.append((a, b))` (one tuple).
5. Forgetting to guard BOTH words against vocab membership, not just one.

## PROGRESS — where he is right now

Exercise: `01-skipgram-negative-sampling/`. Turn a red test suite green, one
function at a time, in dependency order. **17 / 23 tests passing.**

DONE (green, he wrote all of it):
- ✅ `vocab.py::build_vocab` (3 tests)
- ✅ `dataset.py::generate_skipgram_pairs` (4 tests)
- ✅ `negative_sampling.py` — both functions (4 tests) — first try, no bugs
- ✅ `loss.py::negative_log_likelihood` (4 tests) — nailed the log-sigmoid stability trap
- ✅ `model.py::__init__` (1 test) — two matrices + stored vocab_size/embedding_dim.
     Slipped: left a leading space on line 1 (broke whole-file parse) — watch stray
     whitespace/typos at column 0.
- ✅ `model.py::sigmoid` (1 test) — first wrote `-logaddexp(0,-x)` (that's log-sigmoid);
     fixed to `np.exp(-np.logaddexp(0,-x))`.

IN PROGRESS — this is where you pick up:
- ⬜ `model.py` — **the heart of the exercise.** Remaining pieces:
  3. `forward` — `pos_score = v_c·v_o`, `neg_scores = v_neg @ v_c`. (medium)
     — just introduced fancy indexing + matrix-vector product, he's about to write it.
  4. `backward` — **THE exercise.** Hand-derive gradients with the chain rule.
     Test checks against a NUMERICAL gradient (finite differences), so he can't
     fake it — the calculus must be right. Teach the derivation VERBALLY/on paper,
     do not write the code. Key fact he'll need:
     d/dx log(sigmoid(x)) = sigmoid(-x); d/dx log(sigmoid(-x)) = -sigmoid(x).
  5. `sgd_update` — sparse in-place update of ONLY the touched rows. **Duplicate
     gotcha:** `negative_idxs` may repeat; plain fancy-index `-=` drops duplicates,
     must use `np.add.at` (or manual accumulation). (hard)
- ⬜ `loss.py` is done, but `train.py` wiring remains:
  - `train.py::train_one_step` and `train.py::train` (2 tests). Medium. Wire the
    pipeline into a loop; the last test asserts loss DECREASES on the toy corpus
    (threshold: final loss < 0.7× initial).

Suggested teaching order for `model.py`: do the two easy warm-ups (__init__,
sigmoid), then forward, THEN spend real time on the backward derivation before
he writes a line, then sgd_update. Save train.py for last.

## HOW TO WORK

- Repo root on disk: `~/dev/cs224n` (the exercise is in the subfolder above).
- Run tests: `cd 01-skipgram-negative-sampling && python3 -m pytest tests/ -v`
- Run one file: `python3 -m pytest tests/test_model.py -v`
- Every function's docstring explains the WHAT and WHY (not the HOW) — read it
  with him, it's part of the teaching material. There's also a `README.md` with
  the ordered task list and the loss/gradient math for `backward`.
- The prior agent wrote all the scaffolding, tests, docstrings, and this repo.
  No reference implementation exists in the repo (was deliberately removed) —
  don't reintroduce one.

Good luck. He's doing well — concepts stick, mechanics are catching up fast.
Lead with an analogy, keep it tight, let him type every line.
