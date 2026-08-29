"""Flag models worth a second look before any of them reaches the training set.

SimJEB is crowd-sourced CAD -- 381 designs by different people, meshed and solved
automatically -- so bad geometry and non-convergent stress are realistic. A single
corrupt model also distorts the normalisation statistics that every *other* model is
scaled by, which is why this runs before training rather than after a disappointing
result.

Two separate mechanisms, deliberately:

**Hard checks** are physical impossibilities -- NaN results, negative genus, no nodes.
These are not "unusual", they are broken, and they are excluded automatically.

**Statistical flags** are merely unusual. They are reported for review and kept by
default. Dropping every unusual sample removes exactly the hard cases and quietly
inflates the test score; the honest version is to flag, look, and record a reason for
anything actually removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Iglewicz-Hoaglin cutoff for the modified z-score.
MAD_THRESHOLD = 3.5
MAD_SCALE = 0.6745      # rescales MAD so the threshold matches a normal z-score
MEAN_AD_SCALE = 1.253314  # the equivalent constant for mean absolute deviation

# Ti-6Al-4V yield. Peak von Mises above this in a *linear-elastic* solve is a stress
# singularity at a sharp corner, not a strength prediction -- the value grows with mesh
# refinement instead of converging. Used to rank severity, not to delete models: across
# SimJEB the mean peak is ~1,383 MPa, so exceeding yield is the norm here.
YIELD_MPA = 880.0


@dataclass
class OutlierReport:
    flags: pd.DataFrame                                   # model x metric booleans
    hard_failures: dict[int, str] = field(default_factory=dict)
    statistical: list[int] = field(default_factory=list)  # union of the two detectors
    by_method: dict[str, list[int]] = field(default_factory=dict)

    @property
    def must_exclude(self) -> set[int]:
        """Only the physical impossibilities. Statistical flags are for review."""
        return set(self.hard_failures)


def modified_z_scores(values: np.ndarray) -> np.ndarray:
    """Robust z-score: ``0.6745 * (x - median) / MAD``.

    The mean and standard deviation are themselves dragged by outliers, so a few
    extreme models inflate the threshold until nothing gets flagged -- the masking
    effect. Median and median-absolute-deviation are unmoved by up to half the data
    being corrupt. The 0.6745 factor rescales MAD so the threshold is comparable to a
    normal-distribution z-score.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    deviations = np.abs(values - median)
    mad = np.median(deviations)

    if mad >= 1e-12:
        return MAD_SCALE * (values - median) / mad

    # MAD is zero whenever more than half the values are identical -- which does not
    # mean there are no outliers, only that the median absolute deviation cannot see
    # them. Iglewicz and Hoaglin's documented fallback is the *mean* absolute
    # deviation, with its own consistency constant. Without this, a column like
    # "number of load cases" (mostly 4, occasionally not) would report nothing.
    mean_ad = deviations.mean()
    if mean_ad < 1e-12:
        return np.zeros_like(values)     # genuinely constant: nothing to flag
    return (values - median) / (MEAN_AD_SCALE * mean_ad)


def hard_checks(meta: pd.DataFrame) -> dict[int, str]:
    """Physical impossibilities. Excluded automatically, with the reason recorded."""
    failures: dict[int, str] = {}
    stress_columns = [c for c in meta.columns if c.endswith("_stress")]
    disp_columns = [c for c in meta.columns if c.endswith("_magdisp")]

    for model_id, row in meta.iterrows():
        reasons = []

        # Genus cannot be negative for a single closed manifold surface. From the Euler
        # characteristic, a negative value means disconnected components or non-manifold
        # geometry -- fatal for a GNN specifically, because message passing cannot carry
        # load across a gap no matter how long it trains.
        if "genus" in meta and row["genus"] < 0:
            reasons.append(f"negative genus ({row['genus']:.0f}): mesh is disconnected "
                           "or non-manifold")

        if any(not np.isfinite(row[c]) for c in stress_columns + disp_columns):
            reasons.append("non-finite value in the result fields")

        if "num_vertices" in meta and row["num_vertices"] <= 0:
            reasons.append("no vertices")

        # A load that produced no movement means it was never applied.
        if disp_columns and all(abs(row[c]) < 1e-12 for c in disp_columns):
            reasons.append("zero displacement in every load case: load not applied")

        if reasons:
            failures[int(model_id)] = "; ".join(reasons)
    return failures


def detect(meta: pd.DataFrame, columns: list[str] | None = None,
           contamination: float = 0.05, seed: int = 0) -> OutlierReport:
    """Run both detectors over the metadata table and report their union.

    Two methods because they see different things. The modified z-score works one
    metric at a time and cannot notice a model that is unremarkable on every axis
    individually but implausible in combination -- a very large volume with a very low
    node count, say, meaning a coarse mesh on a big part. Isolation Forest scores whole
    feature vectors and assumes no distribution.

    Reporting the union, and both methods separately, is more honest than picking
    whichever gives the tidier answer.
    """
    columns = columns or [
        c for c in [
            "num_vertices", "num_faces", "num_tets", "volume", "surface_area",
            "average_edge_length", "genus", "mass",
            "max_ver_stress", "max_ver_magdisp",
        ] if c in meta.columns
    ]

    frame = meta[columns].astype(float)
    flags = pd.DataFrame(index=meta.index)
    for column in columns:
        flags[column] = np.abs(modified_z_scores(frame[column].to_numpy())) > MAD_THRESHOLD

    z_flagged = sorted(int(i) for i in flags.index[flags.any(axis=1)])

    scaled = StandardScaler().fit_transform(frame.to_numpy())
    forest = IsolationForest(contamination=contamination, random_state=seed)
    forest_flagged = sorted(
        int(i) for i, keep in zip(meta.index, forest.fit_predict(scaled)) if keep == -1
    )

    flags["stress_above_yield"] = meta["max_ver_stress"] > YIELD_MPA \
        if "max_ver_stress" in meta else False

    return OutlierReport(
        flags=flags,
        hard_failures=hard_checks(meta),
        statistical=sorted(set(z_flagged) | set(forest_flagged)),
        by_method={"modified_z": z_flagged, "isolation_forest": forest_flagged},
    )


def summarise(report: OutlierReport, meta: pd.DataFrame) -> str:
    """A short text summary for the notebook and the write-up."""
    lines = [
        f"{len(meta)} models scanned",
        f"  hard failures (excluded)      : {len(report.hard_failures)}",
        f"  modified z-score flags        : {len(report.by_method['modified_z'])}",
        f"  isolation forest flags        : {len(report.by_method['isolation_forest'])}",
        f"  union, for review (kept)      : {len(report.statistical)}",
    ]
    if "stress_above_yield" in report.flags:
        above = int(report.flags["stress_above_yield"].sum())
        lines.append(
            f"  peak stress above {YIELD_MPA:.0f} MPa yield : {above} "
            f"({100 * above / len(meta):.0f}%) -- singular corners are the norm here, "
            "which is why the target is log-transformed rather than clipped"
        )
    for model_id, reason in report.hard_failures.items():
        lines.append(f"    excluded {model_id}: {reason}")
    return "\n".join(lines)
