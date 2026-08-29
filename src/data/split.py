"""Train / validation / test splits, grouped, stratified and verified.

A split is a *claim*: "performance on these held-out brackets predicts performance on
designs nobody has drawn yet." Random sampling does not make that claim true, it
assumes it. This module builds the claim deliberately and then measures whether it
holds.

Two splits are produced, and both are reported:

**Official** -- SimJEB ships ``test_split_0/1/2`` in its metadata table, each holding
out 77 of 381 models. Using one keeps results comparable with other published work,
so it is the primary. It is random over models, though: measured on
``test_split_0``, 9 of the 24 same-submission design groups straddle train and test,
and the category shares range from 8.7% (beam) to 30.2% (butterfly) against a 20%
target.

**Grouped** -- built here: no design family straddles the boundary, and the shape
categories are balanced. The stricter number.

Reporting both, and the gap between them, is the point. The official number answers
"how well does it do on held-out models"; the grouped number answers "how well does
it do on a design nobody in the training set drew". Those are different questions and
only one of them is what a surrogate is judged on.

The five-step recipe is deliberately domain-independent -- describe each sample, find
the leakage groups, stratify, verify, freeze -- so the only parts specific to SimJEB
are *what describes a bracket* and *what makes two brackets dependent*.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# The metadata table quotes its string fields.
_QUOTED = ("category", "author_id", "author", "link_name", "download_file",
           "test_split_0", "test_split_1", "test_split_2")


@dataclass
class Split:
    """One frozen train/val/test assignment, with the evidence that produced it."""

    name: str
    train: list[int]
    val: list[int]
    test: list[int]
    seed: int
    provenance: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)

    @property
    def all_ids(self) -> list[int]:
        return self.train + self.val + self.test

    def split_of(self, model_id: int) -> str:
        for name in ("train", "val", "test"):
            if model_id in set(getattr(self, name)):
                return name
        raise KeyError(f"model {model_id} is not in split {self.name!r}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=_jsonable))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        return cls(**json.loads(Path(path).read_text()))


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)}")


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read ``all_bracket_metadata.tab`` and strip the quoting from string columns."""
    df = pd.read_csv(path, sep="\t")
    for col in _QUOTED:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip('"')
    return df.set_index("id", drop=False)


def leakage_groups(meta: pd.DataFrame,
                   geometric_links: list[tuple[int, int]] | None = None,
                   use_author: bool = False) -> dict[int, int]:
    """Assign each model a group id; models in one group never straddle a split.

    Three signals, of decreasing reliability:

    ``link_name`` (the GrabCAD submission)
        Definitive and accepted outright. Models sharing a submission are variants of
        one entry -- 20 pairs and 4 triples across the 381, covering 52 models.

    ``geometric_links``
        Pairs confirmed near-identical in descriptor space, passed in by the QA stage.
        This is the "metadata proposes, geometry disposes" rule: filename and author
        only ever generate *candidates*, and the data confirms them. It is also what
        keeps the method portable -- with no metadata at all, geometry alone still
        works.

    ``author_id`` (optional, off by default)
        Reaches much further -- 81 authors with more than one model, covering 246 of
        the 381 -- but over-merges, because the same designer often submits genuinely
        different designs. Available as a stricter variant to report alongside.

    Groups are the connected components of the resulting graph.
    """
    ids = list(meta["id"])
    parent = {i: i for i in ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for key in ["link_name"] + (["author_id"] if use_author else []):
        for _, rows in meta.groupby(key):
            members = list(rows["id"])
            for other in members[1:]:
                union(members[0], other)

    for a, b in geometric_links or []:
        union(a, b)

    roots = sorted({find(i) for i in ids})
    renumber = {root: n for n, root in enumerate(roots)}
    return {i: renumber[find(i)] for i in ids}


def _carve_validation(train_ids: list[int], groups: dict[int, int],
                      strata: dict[int, str], fraction: float,
                      seed: int) -> tuple[list[int], list[int]]:
    """Split a training pool into train and validation, keeping groups intact.

    The official splits define train and test only. Carving validation out of train
    still has to respect the grouping, or the early-stopping decision leaks even
    though the test set does not.
    """
    n_folds = max(2, int(round(1 / fraction)))
    y = np.array([strata[i] for i in train_ids])
    g = np.array([groups[i] for i in train_ids])
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    keep_idx, val_idx = next(splitter.split(np.zeros(len(train_ids)), y, g))
    keep = sorted(train_ids[i] for i in keep_idx)
    val = sorted(train_ids[i] for i in val_idx)
    return keep, val


def official_split(meta: pd.DataFrame, groups: dict[int, int], index: int = 0,
                   val_fraction: float = 0.1, seed: int = 0,
                   exclude: set[int] | None = None) -> Split:
    """Use SimJEB's own ``test_split_<index>``, carving validation out of its train."""
    column = f"test_split_{index}"
    if column not in meta.columns:
        raise KeyError(f"{column} not in the metadata table")

    exclude = exclude or set()
    usable = meta[~meta["id"].isin(exclude)]
    strata = dict(zip(usable["id"], usable["category"]))

    is_test = usable[column].astype(str).str.lower() == "true"
    test = sorted(usable.loc[is_test, "id"])
    pool = sorted(usable.loc[~is_test, "id"])
    train, val = _carve_validation(pool, groups, strata, val_fraction, seed)

    return Split(
        name=f"official_split_{index}",
        train=train, val=val, test=test, seed=seed,
        provenance={
            "source": f"all_bracket_metadata.tab :: {column}",
            "val_carved_from_train": True,
            "val_fraction": val_fraction,
            "excluded": sorted(exclude),
        },
    )


def grouped_split(meta: pd.DataFrame, groups: dict[int, int],
                  test_fraction: float = 0.2, val_fraction: float = 0.1,
                  seed: int = 0, exclude: set[int] | None = None) -> Split:
    """Group-preserving, category-stratified split.

    Stratified on SimJEB's own ``category`` -- flat / block / beam / butterfly / arch /
    other -- rather than on clusters computed here. It is the dataset's own semantics,
    it needs no cluster count to justify, and ``other`` has only 9 models, so
    proportional sampling is what stops a class vanishing from a split entirely.

    ``test_fraction`` defaults to 0.2 to match the official split size, so the two
    numbers are compared on equally sized test sets.
    """
    exclude = exclude or set()
    usable = meta[~meta["id"].isin(exclude)]
    ids = np.array(sorted(usable["id"]))
    y = np.array([usable.loc[i, "category"] for i in ids])
    g = np.array([groups[i] for i in ids])

    outer = StratifiedGroupKFold(n_splits=max(2, int(round(1 / test_fraction))),
                                 shuffle=True, random_state=seed)
    pool_idx, test_idx = next(outer.split(np.zeros(len(ids)), y, g))

    test = sorted(int(i) for i in ids[test_idx])
    pool = sorted(int(i) for i in ids[pool_idx])
    strata = dict(zip(usable["id"], usable["category"]))
    train, val = _carve_validation(pool, groups, strata, val_fraction, seed)

    return Split(
        name="grouped_split_v1",
        train=train, val=val, test=test, seed=seed,
        provenance={
            "source": "StratifiedGroupKFold",
            "stratified_on": "category",
            "grouped_on": "leakage_groups",
            "test_fraction": test_fraction,
            "val_fraction": val_fraction,
            "excluded": sorted(exclude),
        },
    )


def select_grouped_split(meta: pd.DataFrame, groups: dict[int, int],
                         seeds: range | list[int] = range(10),
                         test_fraction: float = 0.2, val_fraction: float = 0.1,
                         exclude: set[int] | None = None) -> Split:
    """Try several seeds and keep the best-balanced split.

    ``grouped_split`` stratifies on ``category``, which is a label. That says nothing
    about the *continuous* descriptors -- volume, node count, mass -- and with whole
    design families that must move together and only 77 test models, those can drift
    even when the categories are near-perfect.

    Trying a handful of seeds and keeping the one with the smallest worst-case
    standardised mean difference costs nothing and fixes it.

    **This is only legitimate because the criterion is fixed before any model is
    trained.** Choosing a split after seeing which one produces a nicer R^2 would be
    selecting on the outcome. The seed list, the criterion and the winning score are
    all recorded in ``provenance`` so the decision is auditable.
    """
    candidates = []
    for seed in seeds:
        split = grouped_split(meta, groups, test_fraction, val_fraction, seed, exclude)
        report = verify_split(split, meta, groups)
        if report["groups_straddling"]:      # must never happen; refuse it if it does
            continue
        candidates.append((report["smd_max"], seed, split, report))

    if not candidates:
        raise RuntimeError("no leakage-free split found across the given seeds")

    candidates.sort(key=lambda c: c[0])
    best_smd, best_seed, split, report = candidates[0]

    split.verification = report
    split.provenance.update({
        "seed_selection": {
            "criterion": "minimise max standardised mean difference (train vs test)",
            "decided_before_training": True,
            "seeds_tried": [int(s) for _, s, _, _ in candidates],
            "smd_by_seed": {int(s): round(float(v), 4)
                            for v, s, _, _ in sorted(candidates, key=lambda c: c[1])},
            "chosen_seed": int(best_seed),
            "chosen_smd_max": round(float(best_smd), 4),
        }
    })
    return split


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """SMD between two samples of one variable.

    Preferred over a Kolmogorov-Smirnov test for judging balance here: at ~77 test
    models KS has very little power, so a non-significant result is weak evidence of
    anything. SMD reports effect size directly, and |SMD| < 0.1 is the conventional
    threshold for "balanced".
    """
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled < 1e-12:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled)


def verify_split(split: Split, meta: pd.DataFrame, groups: dict[int, int],
                 descriptor_columns: list[str] | None = None) -> dict:
    """Measure whether the split has the properties it was built to have.

    Every construction step above is a heuristic; this is what turns "I stratified it"
    into "here are the balance statistics". Run on the official splits too -- that is
    how their group straddling was quantified in the first place.
    """
    descriptor_columns = descriptor_columns or [
        "num_vertices", "num_faces", "num_tets", "volume", "surface_area",
        "average_edge_length", "genus", "mass",
    ]
    train, val, test = set(split.train), set(split.val), set(split.test)

    # 1. Leakage: does any group appear in more than one split?
    straddling = []
    by_group: dict[int, set[str]] = {}
    for model_id in split.all_ids:
        where = "train" if model_id in train else "val" if model_id in val else "test"
        by_group.setdefault(groups[model_id], set()).add(where)
    for gid, wheres in by_group.items():
        if len(wheres) > 1:
            straddling.append({"group": int(gid), "splits": sorted(wheres)})

    # 2. Balance: category proportions, train vs test.
    def shares(ids: set[int]) -> dict[str, float]:
        cats = meta.loc[sorted(ids), "category"]
        return {str(k): float(v) for k, v in cats.value_counts(normalize=True).items()}

    # 3. Distribution parity on the descriptors.
    smd = {}
    for col in descriptor_columns:
        if col in meta.columns:
            smd[col] = standardized_mean_difference(
                meta.loc[sorted(train), col].to_numpy(dtype=float),
                meta.loc[sorted(test), col].to_numpy(dtype=float),
            )

    return {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "groups_straddling": len(straddling),
        "straddling_detail": straddling[:20],
        "category_share_train": shares(train),
        "category_share_test": shares(test),
        "smd": smd,
        "smd_max": max(smd.values()) if smd else None,
        "balanced": bool(smd) and max(smd.values()) < 0.1,
    }
