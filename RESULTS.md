# Run log

Every training run gets a config in `configs/`, a row here, and its outputs committed
under `outputs/<run_name>/`. Nothing is edited in place — a run that has happened is a
fact, and changing the code that produced it retroactively would make the comparison
meaningless. That is why the planned changes below are described rather than already
applied to the defaults.

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

All runs here use the **vertical** case, so **60.1 MPa is the reference point**.

### But not yet on equal terms

Reading the paper closely turns up four differences between its protocol and ours:

| | this repo | the paper |
|---|---|---|
| **Nodes scored** | **surface only** (~40% of the mesh) | "vertex-valued" — apparently **all** vertices |
| **Splits** | one **grouped** split | mean over **three random 80/20** splits |
| **Models** | 328 (53 excluded by QA, each with a reason) | all 381 |
| **Targets** | 1 field, 1 load case | 5 fields x 4 cases x 3 splits = 60 models |

Two of these make our task *harder*, not easier. A grouped split removes the same-family
overlap a random split leaves in — **9 of 24 design groups straddle train and test in
`official_split_0`** — and averaging three splits smooths variance we do not get to
smooth.

**Surface-versus-volume is the open one**, and the paper does not settle it. Interior and
surface nodes have different stress distributions, so 60.1 MPa may simply not be the
right target for a surface-only model.

**Next measurement, before any further training:** fit the same degree-three polynomial
on surface nodes only, on our own splits. It runs on the cached data in minutes and costs
no GPU time. Until then, 60.1 is a reference point rather than a verdict.

### The paper also predicts our failure mode

> "several meshes contain one or more elements with large aspect ratios. While
> displacement prediction is generally robust to the presence of a few distorted
> elements, **the accuracy of stress predictions could be improved by improving mesh
> quality, using second-order elements, and replacing sharp corners in the geometry with
> small fillets.**" — SimJEB paper, section 4.2

The error analysis reached the same conclusion from the other direction: the worst
brackets are those peaking above 5x yield, and RMSE is more than double MAE. Von Mises on
SimJEB is a harder target than displacement, and its authors say so. That is context for
the result, not an excuse for it.

One more detail worth knowing: the design categories we stratify on are **hand-assigned**
— the paper calls them "subjective and imperfect". Still the best available grouping
variable, but judgement rather than measurement.

## Runs

| run | val MAE | **test MAE** | test R² | vs. 60.1 baseline | epochs | verdict |
|---|---|---|---|---|---|---|
| `c1_baseline` | 59.5 | **84.0** | 0.227 | not like-for-like &mdash; see above | 398 (best 197) | Overfit; a working pipeline, not yet a competitive model |

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

**Read.** The model is **data-limited, not capacity-limited**. It learned something
real — R² 0.51 on validation against 0 for a constant predictor — but on a stricter
split, scoring only surface nodes, it does not yet clearly separate from a positional
prior. With 236 training geometries and no regularisation to speak of, that is the
expected outcome rather than a surprising one.

**Cause of the overfit:** `weight_decay=1e-5` is effectively no regularisation.

#### Test results

| | grouped split | official split |
|---|---|---|
| **MAE** | **84.0 MPa** | 62.8 MPa |
| RMSE | 174.6 MPa | 120.7 MPa |
| R² | 0.227 | 0.466 |
| 95% CI on R² | [0.153, 0.305] | [0.327, 0.470] |
| trivial baseline R² | −0.0001 | −0.0063 |

**84.0 MPa against a 60.1 MPa reference.** The two are not measured the same way — see
the comparability table above — so this is not yet a clean win or loss. What is not in
doubt is that the model overfit and that its errors concentrate where the paper's own
authors say stress prediction is hardest.

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

The fix for both: enlarge the validation set and stop on a moving average of the
validation loss rather than its raw per-epoch value.

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

## Planned work, in order

Each is a separate run with its own config, so every change stays attributable.

### 1. Recover the rotation outliers

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

### 3. Regularisation

`weight_decay` 1e-5 -> 1e-3, `dropout` 0 -> 0.1, and a shorter patience: the first
run spent 3.6 GPU-hours training past its own peak. Validation should also be
enlarged, and early stopping should compare a moving average rather than the raw
per-epoch value.

### 4. Huber loss on the log target

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
