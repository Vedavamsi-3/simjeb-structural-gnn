# Handover

Where the project stands, what has been tried, and what to do next. Written so a
session — or a person — starting cold can pick it up without reconstructing anything.

**Repo:** https://github.com/Vedavamsi-3/simjeb-structural-gnn
**Local:** `D:\Vamsi_courses\Projects\simjeb-structural-gnn`
**Data:** `D:\Vamsi_courses\Projects\3D_deep_learning_project_2` (raw archives, sample
model 148, `all_bracket_metadata.tab`)

---

## State

| | |
|---|---|
| Pipeline | complete — fetch, QA, graph build, splits, model, train, evaluate |
| Tests | **228 passing** (`pytest -q`) |
| Runs completed | **one** (`c1_baseline`) |
| Test result | **84.0 MPa** MAE, R² 0.227, on 67 held-out brackets |
| Reference point | 60.1 MPa (SimJEB paper's naive baseline) — **not measured the same way** |

### Read these first, in order

1. `README.md` — pipeline, method rationale, results, and the comparability caveats
2. `RESULTS.md` — c1's full diagnosis and the planned work with evidence for each item
3. `configs/c1_baseline.json` — every setting that produced the result
4. `SETUP.md` — how to run the three Kaggle notebooks

Two design documents live outside the repo, deliberately (they are personal notes, not
project deliverables):

- `..\3D_deep_learning_project_2\docs\simjeb-plan.html` — the full design reasoning,
  every decision with its justification
- `..\3D_deep_learning_project_2\docs\simjeb-findings.html` — 22 findings, each with how
  it was found, why it mattered, and a sentence to say out loud

---

## The immediate next step

**Not a training run.** Settle whether the benchmark comparison is even valid.

The SimJEB paper reports 60.1 MPa for a degree-three polynomial in x, y, z fitted to the
average field. But its fields are described as *"vertex-valued"* — apparently **all mesh
vertices**, while this project scores **surface nodes only** (~40% of the mesh). Interior
and surface stress distributions differ, so 60.1 may simply not be the right target.

**Settling it is a script, not a GPU run:** fit the same polynomial on surface nodes
only, on our splits, and compare like with like. Minutes on the cached data.

Until that is done, treat 60.1 MPa as a reference point rather than a verdict.

---

## What has been tried, and what it showed

### c1_baseline — the only committed run

398 epochs, 7 h 14 min on a T4. Best epoch 197.

**It overfit.** Validation loss bottomed at epoch 197 and rose to 0.55 while training
loss kept falling from 0.56 to 0.37 — a 34% improvement on seen data alongside a 15%
regression on unseen. `weight_decay=1e-5` was effectively no regularisation.

**Its validation set lied.** Validation MAE 59.5 became test MAE 84.0. Two structural
causes: 25 validation models carry roughly ±10 MPa of noise, and taking the *minimum* of
398 noisy scores is biased low regardless of set size.

**Where it fails is legible.** The worst brackets peak above 4,700 MPa; the best top out
near 1,400. RMSE at more than double MAE says the same thing — a few nodes carry most of
the error.

**One thing worked well.** The QA stage predicted its own failures: median R² on
QA-flagged models was 0.183 against 0.381 on unflagged, identified before training from
geometry and mesh statistics alone.

### A second run was attempted and then reverted

Six regularisation changes at once — weight decay 1e-3, dropout 0.1, Huber loss, no
material features, smoothed early stopping, shorter patience — plus a larger validation
set and a looser rotation tolerance recovering ~47 previously excluded models.

**It produced no measurable improvement.** MAE 81.2 against c1's 84.0, R² 0.178 against
0.227, with heavily overlapping bootstrap intervals.

That run and its code were deliberately removed from the repo to keep the record at the
c1 state. **The finding is worth knowing anyway:** stronger regularisation alone did not
help, which points at **data volume** rather than regularisation as the binding
constraint.

A second observation from its per-model results, which also applies to c1: **mean MAE is
dominated by a single catastrophic bracket.** Median MAE across test models was 61.5 MPa
while the mean was 81.2; one model with 7,835 MPa peak stress contributed 1,049 MPa of
error on its own. Reporting median alongside mean would be more honest.

---

## Planned work, in order

Full reasoning and evidence for each is in `RESULTS.md`.

1. **Settle surface-vs-volume** (above) — free, and it decides what the target even is
2. **Recover the rotation outliers** — 47 models, 12% more data, excluded on a 2° / 5 mm
   tolerance that is probably tighter than needed (median rotation across the dataset was
   0.12°, maximum 8.74°)
3. **Drop the material features** — constant across all 381 models, so exactly zero
   information; free to remove
4. **Regularisation** — weight decay 1e-3, dropout 0.1, shorter patience, larger
   validation set, early stopping on a moving average
5. **Huber loss on the log target** — RMSE at double MAE says a few singular nodes
   dominate the gradient even after `log1p`

If none of that is enough, the remaining levers are about **data volume**, which is the
suspected constraint:

- **All four load cases** — 4× the data, but needs a load-direction node feature, since
  the four cases are currently indistinguishable to the model
- **A virtual global node** — message passing reaches 8 hops while the lug is 54 hops
  from the nearest bolt; one node connected to all others makes any pair 2 hops apart

---

## Practical notes

**Kaggle.** GPU quota is 30 h/week. Notebook 1 (data build) runs on CPU and is free;
notebook 2 (training) is the only real consumer at roughly 65 s/epoch. Sessions cap at
12 h, so `max_hours` is set to 10.5 and the loop checkpoints every epoch and resumes.

**The notebooks clone `src/` from GitHub on every run.** A code change reaches Kaggle by
pushing it. A change to a `.ipynb` file needs the notebook re-imported.

**The `.pt` cache holds raw MPa.** `log1p` is applied at load, normalisation inside the
training loop. So the cached data can be inspected in physical units directly — which is
what makes distribution analysis tractable.

**Local machine has 7.7 GB RAM** and segfaults on a full-dataset backward pass. Local
work is code authoring and unit tests against the model-148 fixture only.

**Environment.** `.venv` with CPU-only torch. If pip or git fails with a certificate
error, that is Avast intercepting HTTPS — see `SETUP.md` for the flags.

---

## Open questions

- Is 60.1 MPa the right target for a surface-only model? *(step 1 above)*
- Are the 47 rotation outliers genuinely misaligned, or discretisation noise? The
  evidence is in `outputs/qa/alignment_report.csv` on the Kaggle dataset — if they
  cluster just past the threshold, they should come back
- Should median MAE be the headline rather than mean, given one bracket dominates?
