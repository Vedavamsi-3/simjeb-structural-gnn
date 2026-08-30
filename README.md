# SimJEB structural surrogate

A graph neural network that predicts **surface von Mises stress** on jet-engine
brackets, trained end to end from raw solver decks with open-source tooling only —
Python, PyTorch, PyTorch Geometric.

The dataset is [SimJEB](https://simjeb.github.io/): 381 crowd-sourced CAD brackets from
the GE Jet Engine Bracket Challenge, meshed and solved with OptiStruct.

> **Current status: the model does not yet beat the benchmark.** Test MAE is
> **84.0 MPa** against the SimJEB paper's naive baseline of **60.1 MPa**. The diagnosis
> and the planned fixes are in [`RESULTS.md`](RESULTS.md). This README documents the
> pipeline and reports the numbers as they stand.

---

## The pipeline

```
Harvard Dataverse  ──  7 GB of archives, streamed member-by-member, never extracted
        │
        ▼
1. QA and dataset build            [Kaggle CPU, ~40 min, run once]
        │   profile 381 models · verify frame alignment · detect outliers
        │   build cached surface graphs · freeze the splits
        ▼
   328 cached graphs (1.48 GB)  +  two verified splits
        │
        ▼
2. Training                        [Kaggle T4, resumable across sessions]
        │
        ▼
3. Evaluation                      [Kaggle T4, run once at the end]
```

Everything that touches all 381 models runs on Kaggle. The two steps needing judgement
— deciding exclusions and defining the split — are local and human.
Setup instructions: [`SETUP.md`](SETUP.md).

---

## What the model sees

Each bracket becomes a graph over its **surface** nodes (39.5% of the mesh — these are
thin, perforated parts).

| per node | 11 values |
|---|---|
| node type one-hot | 3 — plain surface / bolt interface / load lug |
| surface normal | 3 |
| distance to nearest clamp | 1 |
| distance to nearest loaded node | 1 |
| material (E, ν, ρ) | 3 |

| per edge | 4 values |
|---|---|
| relative position `x_j − x_i` | 3 |
| edge length | 1 |

**Why edge features are not optional.** Strain is a spatial derivative of displacement,
and stress follows from strain. A spatial derivative is built from differences between
neighbouring points — which is exactly what a message carrying `x_j − x_i` lets a layer
compute. An architecture without edge features (GCN, GraphSAGE, GAT) can record that
two nodes are connected but not how far apart or in which direction, so it cannot
represent the derivative at all. That requirement, not preference, selects this family.

Edges come from **tetrahedron connectivity**, not k-nearest-neighbours. Two surfaces a
millimetre apart across a gap share no tetrahedron, so no edge appears — because no
material joins them and no load passes between them.

---

## Model

MeshGraphNet-style encode–process–decode
([Pfaff et al., ICLR 2021](https://arxiv.org/abs/2010.03409)), adapted in two ways:
this is a **single static solve**, not time-stepping rollout, and there is **one edge
type**, since nothing collides.

```
x  (N, 11)        ──encode──▶  h (N, 64)
edge_attr (E, 4)  ──encode──▶  e (E, 64)

    ×8 message-passing blocks, residual:
        edge:  Δe = MLP([e, h_src, h_dst])
        node:  Δh = MLP([h, Σ Δe])

h ──decode──▶  log1p(stress)  + 3 auxiliary displacement channels
```

**247,556 parameters.**

### Why 64 × 8 and not the paper's 128 × 15

The usual justification for depth is that a node's receptive field should span the load
path. Measured on a real SimJEB graph, **the load lug is 54 hops from the nearest bolt
hole** — so 15 blocks reaches barely a quarter of it, and 54 blocks would need 10.6 GB
per graph. The paper's meshes are an order of magnitude coarser; the default does not
transfer.

Spanning the load path is the wrong target anyway. Stress concentration is **local**,
set by fillet radii and thickness changes — 8 hops at 0.95 mm mean edge length reaches
7.6 mm, which covers that scale. The **global** context comes from the
distance-to-clamp and distance-to-load features directly, with no message passing
required. Those features are load-bearing, not conveniences.

### Why the stress target is log-transformed

Peak von Mises across the dataset runs from 301 to **14,902 MPa**, against a Ti-6Al-4V
yield of ~880 MPa. **62% of models exceed yield at their peak node.** These are
linear-elastic solves, so that is the signature of stress singularities at sharp
corners — numerical artefacts that grow with mesh refinement rather than converging.

Under a plain z-score and MSE, a handful of singular nodes would supply most of the
gradient. `log1p` compresses that tail without discarding it. **All reported metrics
invert the transform first** — R² on a log target flatters the model by compressing
exactly the large errors that matter.

---

## Dataset QA

SimJEB is crowd-sourced CAD — 381 designs by different people, meshed and solved
automatically. A single corrupt model distorts the normalisation statistics every other
model is scaled by, so profiling comes before training.

**53 of 381 models excluded**, each with a written reason in `outputs/qa/excluded.csv`:

| reason | count |
|---|---|
| rotation outlier | 47 |
| negative genus — disconnected or non-manifold mesh | 4 |
| unparseable deck / non-contiguous node numbering | 2 |

**Negative genus** cannot occur for a closed manifold surface; from the Euler
characteristic it means disconnected components. That is fatal for a GNN specifically —
message passing cannot carry load across a gap.

**Rotation outliers are removed, never re-oriented.** The load is a fixed global vector
that does *not* rotate with the part, so a turned bracket under that same load carries
force along a different internal path: a different problem, not a different view.
Rotating the geometry back would not rotate the load that produced the stress field,
leaving an aligned mesh whose answers came from a different loading — a corrupt sample
no longer detectable as an outlier.

Statistical outliers (95 models) were **flagged and kept**. Dropping every unusual
sample removes exactly the hard cases and inflates the test score.

---

## Splits

Two, and both are reported.

| | train / val / test | design families straddling | max SMD |
|---|---|---|---|
| `official_split_0` (SimJEB's own) | 238 / 23 / 67 | **8** | 0.161 |
| **`grouped_split_v1`** (this repo) | 236 / 25 / 67 | **0** | **0.093** |

SimJEB ships three official splits, which makes results comparable across work — but
they are random over models. Measured here: **9 of 24 same-submission design groups
straddle train and test** in split 0, and category shares range from 8.7% (beam) to
30.2% (butterfly) against a 20% target.

The grouped split uses `StratifiedGroupKFold`, grouping on GrabCAD submission and
stratifying on SimJEB's own shape categories. Design families are found by combining
metadata (submission URL, author) with **geometric confirmation** — *metadata proposes,
geometry disposes* — so the method degrades gracefully to pure geometry on a dataset
with no metadata at all.

The seed was chosen from ten candidates by lowest worst-case standardised mean
difference. That is legitimate only because the criterion was **fixed before any model
was trained**; the seed list and the winning score are recorded in the split file.

---

## Results

Vertical load case, 67 held-out brackets, metrics in MPa.

| | grouped split | official split | paper's naive baseline |
|---|---|---|---|
| **MAE** | **84.0** | 62.8 | **60.1** |
| RMSE | 174.6 | 120.7 | — |
| R² | 0.227 | 0.466 | — |
| 95% CI on R² | [0.153, 0.305] | [0.327, 0.470] | — |
| trivial baseline R² | −0.0001 | −0.0063 | — |

The paper's baseline is a degree-three polynomial in x, y, z fitted to the *average*
field — it never looks at the geometry. Its authors write that models failing to beat
it "effectively have no predictive value". **This model does not beat it.**

The official-split figure is **not** a fair comparison: the model trained on the grouped
split, so some official test brackets were in its training set.

The bootstrap interval resamples whole **models**, not nodes — nodes within a bracket
are highly correlated, and resampling them would treat millions of dependent points as
independent and produce an interval far too narrow to be honest.

### What went wrong

**Overfitting.** Validation loss bottomed at epoch 197 and rose to 0.55 while training
loss kept falling from 0.56 to 0.37 — a 34% improvement on seen data paired with a 15%
regression on unseen. `weight_decay=1e-5` was effectively no regularisation.

**The validation set lied.** Val MAE 59.5 → test MAE 84.0. Two structural causes: 25
validation models gives roughly ±10 MPa of noise, and taking the *minimum* of 398 noisy
scores is biased low regardless of set size.

**Extreme singularities dominate the error.** The worst brackets have peaks above 4,700
MPa; the best top out near 1,400. RMSE at more than double MAE says the same thing.

### What worked

**The QA stage predicted its own failures.** Median R² on flagged models: **0.183**,
against **0.381** on unflagged — identified before training, from geometry and mesh
statistics alone.

**The trivial baseline scored −0.0001** — exactly zero, as designed. The evaluation is
sound; the model is weak.

Full diagnosis and planned fixes: [`RESULTS.md`](RESULTS.md).

---

## Repository

```
src/data/      fetch · parse .fem decks · build graphs · dataset · splits · normalisation
src/qa/        frame alignment · outlier detection · figures
src/models/    the network
src/train.py   resumable training loop
src/evaluate.py
notebooks/     the three Kaggle notebooks
configs/       one file per run
tests/         228 tests
```

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
pytest -q
```

Tests run against a single sample model (31 MB) — the full dataset is not needed.

## Acknowledgement

Whalen, Beyene & Mueller, *SimJEB: Simulated Jet Engine Bracket Dataset*,
Computer Graphics Forum, 2021. [arXiv:2105.03534](https://arxiv.org/abs/2105.03534)
