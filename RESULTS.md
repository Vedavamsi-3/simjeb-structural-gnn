# Run log

Every training run gets a config in `configs/`, a row here, and its outputs committed
under `outputs/<run_name>/`. Nothing is edited in place — a run that has happened is a
fact, and changing the code that produced it retroactively would make the comparison
meaningless.

## Benchmark

SimJEB's own paper (Whalen, Beyene & Mueller 2021, Table 2) gives a naive surrogate as
the floor: a **degree-three polynomial in x, y, z** fitted to the *average* field across
the training set. It never looks at the geometry.

| load case | naive MAE (MPa) |
|---|---|
| **vertical** | **60.1** |
| horizontal | 89.3 |
| diagonal | 36.1 |
| torsional | 84.4 |

> "Surrogate models that fail to beat this naive model effectively have no predictive
> value." — SimJEB paper, §6

All runs here use the **vertical** case, so **60.1 MPa is the number to beat**.

## Runs

| run | val R² | val MAE | test R² | test MAE | vs. 60.1 | epochs | notes |
|---|---|---|---|---|---|---|---|
| `c1_baseline` | 0.507 | 59.5 | *pending* | *pending* | −1% | 398 (best 197) | Overfit badly after epoch 197 |
| `c2_regularised` | — | — | — | — | — | — | not yet run |

### c1_baseline

- **Data:** 328 graphs, `grouped_split_v1` (236 train / 25 val / 67 test)
- **Stopped:** early stopping, 200 epochs without improvement
- **Best epoch:** 197 of 398 — 7 h 14 min on a T4, ~65 s/epoch

**What happened.** Validation loss bottomed at epoch 197 (0.480) and then rose to 0.55
while training loss kept falling from 0.56 to 0.37. A 34% improvement on data it had
seen, paired with a 15% regression on data it had not: the model had capacity to
memorise 236 brackets and used it.

The val/train loss ratio of **1.52** is the same statement in one number.

**Read.** The model is **data-limited, not capacity-limited**. At 59.5 MPa it sits
level with a polynomial that ignores geometry entirely, which by the paper's own
standard is the threshold of having no predictive value. It learned *something* —
R² 0.51 against 0 for a constant predictor — but not much that a positional prior
does not already capture.

**Cause of the overfit:** `weight_decay=1e-5` is effectively no regularisation.

### c2_regularised — planned

Only the regularisation changes, so any difference is attributable to it.

| | c1 | c2 |
|---|---|---|
| `weight_decay` | 1e-5 | **1e-3** |
| `dropout` | 0.0 | **0.1** |
| `patience` | 200 | **60** |

Architecture, data, split and seed are identical.

**Prediction, recorded before running:** the train/val gap narrows and validation MAE
improves. If it does not, the problem is data volume rather than regularisation, and
the next lever is recovering the 47 models excluded as rotation outliers (+20% data)
or training on all four load cases.

Writing the prediction down first is what makes the result informative either way.

## If regularisation is not enough

In order of expected value:

1. **Recover the rotation outliers.** 47 models excluded on a 2° / 2 mm tolerance that
   may be tighter than necessary — median rotation across the dataset was 0.12°. Worth
   inspecting `outputs/qa/alignment_report.csv` to see whether they cluster just past
   the threshold or sit far out.
2. **Train on all four load cases** — 4× the data. Needs a load-direction node feature,
   since the four cases are currently indistinguishable to the model.
3. **A virtual global node.** Message passing reaches 8 hops; the lug is 54 hops from
   the nearest bolt. A single node connected to all others makes any pair 2 hops apart
   for ~0.1 GB.
