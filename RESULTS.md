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

| run | val MAE | **test MAE** | test R² | vs. 60.1 baseline | epochs | verdict |
|---|---|---|---|---|---|---|
| `c1_baseline` | 59.5 | **84.0** | 0.227 | **40% worse** | 398 (best 197) | Does not beat the benchmark |
| `c2_regularised` | — | — | — | — | — | not yet run |

All test figures are on `grouped_split_v1`, 67 held-out brackets, vertical load case,
metrics in MPa after inverting the log transform.

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

#### Test results

| | grouped split | official split |
|---|---|---|
| **MAE** | **84.0 MPa** | 62.8 MPa |
| RMSE | 174.6 MPa | 120.7 MPa |
| R² | 0.227 | 0.466 |
| 95% CI on R² | [0.153, 0.305] | [0.327, 0.470] |
| trivial baseline R² | −0.0001 | −0.0063 |

**The model does not beat the paper's naive baseline.** 84.0 MPa against 60.1 MPa is
40% worse, on a benchmark whose authors state that failing to beat it means having no
predictive value. That is the honest headline.

The official-split number is better but is **not** a fair comparison: the model was
trained on the grouped split, so some of the official split's test brackets were in its
training set. It is reported for context only.

#### The validation set lied

Validation MAE 59.5 -> test MAE 84.0, a 41% degradation. Two causes, both structural:

1. **25 validation models is too few.** With per-model MAE varying by roughly 50 MPa,
   the standard error on a 25-model mean is about ±10 MPa -- larger than the effects
   being chased.
2. **Taking the minimum of a noisy series is biased low.** Early stopping picked the
   best of 398 noisy validation scores; the minimum of noise sits below the true mean
   by roughly the noise scale, whatever the set size.

Fixed for c2 by enlarging validation to 50 and stopping on a 10-epoch moving average
rather than the raw value.

#### Where it fails, and why that is informative

| worst models | R² | peak true stress |
|---|---|---|
| 587 | −0.72 | 4,792 MPa |
| 419 | −0.41 | 2,557 MPa |
| 398 | −0.31 | 4,341 MPa |

| best models | R² | peak true stress |
|---|---|---|
| 506 | 0.70 | 1,481 MPa |
| 169 | 0.69 | 870 MPa |
| 597 | 0.62 | 819 MPa |

**Failure concentrates on brackets with extreme stress singularities** -- peaks above
5x the 880 MPa yield. RMSE (174.6) is more than double MAE (84.0), which says the same
thing: a small number of nodes carry most of the error. The `log1p` transform
compresses that tail but does not tame it.

#### One thing that worked

**The QA stage predicted its own failures.** Median R² on models flagged by the outlier
detectors was **0.183**, against **0.381** on unflagged models -- identified before any
training happened, from geometry and mesh statistics alone.

And the trivial baseline scored −0.0001, i.e. exactly zero as designed. The evaluation
is sound; the model is simply weak.

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

## Planned work, in order

Each is a separate run with its own config, so every change stays attributable.

### 1. Recover the rotation outliers (`c3`)

47 of 380 models -- 12% of the dataset -- were excluded on a 2 deg / 2 mm alignment
tolerance. Median rotation across the whole dataset was 0.12 deg, so the dataset is
genuinely well aligned and the threshold may simply be tighter than it needs to be.

The exclusion was defensible when written: the load is a fixed global vector that does
not rotate with the part, so a genuinely rotated bracket is a different physical
problem. But `outputs/qa/alignment_report.csv` will show whether the 47 cluster just
past the threshold -- in which case they are mesh discretisation noise, not rotated
designs -- or sit far out.

At 236 training graphs, +12% data is worth more than almost any architectural change.

### 2. Drop the material features

Ti-6Al-4V for all 381 models, so the three columns are constant and carry exactly zero
information. They were included to keep the pipeline extensible to a multi-material
dataset, and that reasoning still holds -- but they add input width for nothing, and
removing them is free.

Predicted effect: no measurable change. Worth doing because it makes the feature set
honest, not because it will help.

### 3. Regularisation (`c2`, next)

`weight_decay` 1e-5 -> 1e-3, `dropout` 0 -> 0.1, `patience` 200 -> 60.
Plus validation enlarged 25 -> 50 and early stopping on a 10-epoch moving average.

### 4. Huber loss on the log target (`c4`)

The evidence for this is specific: RMSE (174.6) is more than double MAE (84.0), and the
worst-scoring brackets are exactly those with peaks above 5x yield. A small number of
singular nodes dominates the gradient even after `log1p`.

Huber is quadratic near zero and linear in the tail, so a node that is wildly wrong
contributes a bounded gradient instead of an unbounded one. That is precisely the
right shape for a target whose extremes are numerical artefacts of the solver rather
than physics worth fitting.

Deliberately **after** regularisation, so the two are not confounded.

## If none of that is enough

The remaining levers are about data volume, which is the suspected binding constraint:

1. **All four load cases** -- 4x the data. Requires a load-direction node feature, since
   the four cases are currently indistinguishable to the model.
2. **A virtual global node.** Message passing reaches 8 hops; the lug is 54 hops from
   the nearest bolt hole. One node connected to all others makes any pair 2 hops apart
   for roughly 0.1 GB.
