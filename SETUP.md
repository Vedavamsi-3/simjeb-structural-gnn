# Running this on Kaggle

The raw SimJEB distribution is ~7 GB of archives that expand past 20 GB. **You download
and process it once.** Everything afterwards runs against a 2.4 GB cache of prebuilt
graphs that Kaggle mounts read-only in seconds.

```
notebook 1  ── run ONCE ──▶  download 7 GB, QA, build 381 graphs   (~40 min, CPU)
                                        │
                                        ▼
                              Kaggle Dataset: simjeb-graphs
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
              notebook 2 (train, GPU)             notebook 3 (evaluate, GPU)
              run as many times as needed         run once at the end
```

The source data never changes, so step 1 never needs repeating -- including across
resumed training sessions and any later ablations.

---

## One-time: build the dataset

1. **Import** `notebooks/kaggle_1_data.ipynb` into Kaggle
   (*Create -> Notebook -> File -> Import Notebook*).
2. **Settings:** Accelerator **None**, Internet **ON**.
   Internet is required -- the notebook fetches from Harvard Dataverse.
3. Run with **Save & Run All (Commit)**, not interactively. A committed run survives a
   browser disconnect; an interactive one does not.
4. When it finishes, open the output and click **New Dataset**. Name it
   **`simjeb-graphs`**.

A named dataset rather than a raw notebook output: it is stable, it does not depend on
chasing version numbers, and it survives independently of the notebook.

The output contains:

| path | contents |
|---|---|
| `graphs/*.pt` | 381 cached surface graphs, ~6 MB each |
| `splits/grouped_split_v1.json` | the strict split -- no design family straddles it |
| `splits/official_split_0.json` | SimJEB's own split, for comparability |
| `qa/*.csv` | alignment report, flagged models, exclusions with reasons |
| `qa/figures/*.png` | the QA figures |

### Why not keep the cache in git

| | |
|---|---|
| Total size | 2.4 GB -- GitHub warns above 1 GB and clones become slow |
| Git LFS free tier | 1 GB storage, 1 GB/month bandwidth -- not enough |
| Clone time per Kaggle run | minutes, every single run |

Kaggle Datasets are built for this: 100 GB free, mounted rather than downloaded, and
they live next to the compute.

---

## Training

1. Import `notebooks/kaggle_2_train.ipynb`.
2. **Add Data** -> your `simjeb-graphs` dataset.
3. **Settings:** Accelerator **GPU T4**, Internet **ON** (for `pip install`).
4. Check the mount path Kaggle shows and set `DATA` at the top of the notebook to
   match -- usually `/kaggle/input/simjeb-graphs`.
5. **Save & Run All (Commit)**.

A timing probe runs first and reports the real seconds-per-epoch, so you learn how many
sessions the run needs before spending ten hours finding out.

### Continuing a long run

At 20-60 s per epoch, a 3,000-epoch budget spans several sessions. The loop stops
cleanly on a wall-clock limit and checkpoints every epoch, so it resumes rather than
restarting.

To continue:

1. **Add Data** -> the previous run's output, alongside `simjeb-graphs`.
2. Set `RESUME_FROM` to that mount path.
3. Run again. The log will say `resumed C from epoch N`.

Early stopping may fire first -- 3,000 is a ceiling, not a target.

---

## Evaluation

1. Import `notebooks/kaggle_3_evaluate.ipynb`.
2. **Add Data** -> `simjeb-graphs` **and** the final training run's output.
3. Set `DATA` and `TRAIN` to their mount paths. Accelerator **GPU T4**.

Run this **once**, at the end. Every look at the test set during development turns it
into a second validation set, and model selection slowly overfits to it.

---

## Running the tests locally

The full dataset is not needed. Everything is tested against SimJEB model 148, which
ships in the 31 MB sample bundle.

```bash
python -m venv .venv
.venv/Scripts/activate                       # Windows
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
pytest -q
```

Point `SIMJEB_SAMPLE_DIR` at the extracted sample bundle if it is not at the default
path in `tests/conftest.py`. Tests skip rather than fail when the fixture is absent.

CPU-only torch is deliberate: nothing trains locally, so the CUDA build would be
2.5 GB of nothing.

### If pip fails with a certificate error

Antivirus that intercepts HTTPS (Avast, for instance) presents its own certificate,
which pip does not trust by default:

```bash
pip install --cert "C:\ProgramData\Avast Software\Avast\wscert.pem" -r requirements.txt
```

For git, `git config http.sslBackend schannel` makes it use the Windows trust store,
which already trusts the interceptor.
