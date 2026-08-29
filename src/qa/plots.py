"""Figures for the QA report.

The flags say *which* models are unusual; these say *why*, and whether the thresholds
are sensible. They also go straight into the write-up -- a histogram spiking at zero is
itself a result when the claim being tested is "the models are pre-aligned".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.qa.alignment import RigidFit
from src.qa.outliers import YIELD_MPA, OutlierReport


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_distributions(meta: pd.DataFrame, report: OutlierReport,
                              path: str | Path, columns: list[str] | None = None) -> Path:
    """Histogram per metric with flagged models marked.

    The flags alone cannot tell you whether a threshold is reasonable. Seeing where the
    flagged models actually sit in the distribution can.
    """
    columns = columns or [c for c in ["num_vertices", "num_tets", "volume",
                                      "surface_area", "genus", "mass",
                                      "max_ver_stress", "max_ver_magdisp"]
                          if c in meta.columns]
    rows = int(np.ceil(len(columns) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        values = meta[column].astype(float)
        ax.hist(values, bins=40, color="tab:blue", alpha=0.75)
        flagged = report.flags[column] if column in report.flags else None
        if flagged is not None and flagged.any():
            for v in values[flagged]:
                ax.axvline(v, color="tab:red", alpha=0.6, linewidth=0.8)
        ax.set_title(column, fontsize=10)
        ax.grid(alpha=0.3)

    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle("metric distributions -- red lines are flagged models", y=1.01)
    return _save(fig, path)


def plot_alignment(fits: list[RigidFit], path: str | Path,
                   rotation_tol_deg: float = 2.0) -> Path:
    """Rotation and translation of every model relative to the reference.

    SimJEB claims the models are pre-aligned, so the expected picture is a spike at
    zero and an empty flagged region. That is worth plotting precisely because a
    confirmed assumption is still a result -- and because the alternative, discovering
    later that it was false, is expensive.
    """
    rotation = np.array([f.rotation_deg for f in fits])
    translation = np.array([f.translation_mm for f in fits])
    rmsd = np.array([f.rmsd_mm for f in fits])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, values, title, unit in (
        (axes[0], rotation, "rotation vs reference", "degrees"),
        (axes[1], translation, "translation vs reference", "mm"),
        (axes[2], rmsd, "residual after best rigid fit", "mm"),
    ):
        ax.hist(values, bins=40, color="tab:blue", alpha=0.8)
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].axvline(rotation_tol_deg, color="tab:red", linestyle="--",
                    label=f"{rotation_tol_deg}° tolerance")
    axes[0].legend(fontsize=8)

    reflected = sum(f.is_reflected for f in fits)
    fig.suptitle(
        f"frame check: max rotation {rotation.max():.2f}°, "
        f"max residual {rmsd.max():.2f} mm, {reflected} reflected",
        y=1.02,
    )
    return _save(fig, path)


def plot_landmarks(landmarks_by_model: dict[int, np.ndarray], path: str | Path) -> Path:
    """The five interface centroids from every model, overlaid.

    One glance either confirms or destroys the alignment assumption: if the brackets
    share a frame, this is five tight clusters. If it is a smear, they do not.
    """
    stacked = np.stack([landmarks_by_model[i] for i in sorted(landmarks_by_model)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (a, b, na, nb) in zip(axes, [(0, 1, "x", "y"), (0, 2, "x", "z"),
                                         (1, 2, "y", "z")]):
        for interface in range(stacked.shape[1]):
            label = "load lug" if interface == stacked.shape[1] - 1 else f"bolt {interface + 1}"
            ax.scatter(stacked[:, interface, a], stacked[:, interface, b],
                       s=6, alpha=0.5, label=label)
        ax.set_xlabel(f"{na} (mm)")
        ax.set_ylabel(f"{nb} (mm)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("interface landmark centroids across all models", y=1.01)
    return _save(fig, path)


def plot_stress_context(meta: pd.DataFrame, path: str | Path) -> Path:
    """Peak stress per load case, against the material's yield strength.

    The finding this exists to make visible: most models exceed yield at their peak
    node, because a linear-elastic solve reports unbounded stress at sharp corners.
    That is what forces a log-transformed target -- under a plain z-score those few
    singular nodes would supply most of the gradient.
    """
    cases = [(c, c.split("_")[1]) for c in meta.columns
             if c.startswith("max_") and c.endswith("_stress")]
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 4))

    for column, name in cases:
        ax_hist.hist(meta[column], bins=50, alpha=0.5, label=name)
    ax_hist.axvline(YIELD_MPA, color="tab:red", linestyle="--",
                    label=f"yield {YIELD_MPA:.0f} MPa")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("peak von Mises (MPa)")
    ax_hist.set_title("peak stress per load case")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    ax_box.boxplot([meta[c] for c, _ in cases], labels=[n for _, n in cases])
    ax_box.axhline(YIELD_MPA, color="tab:red", linestyle="--")
    ax_box.set_yscale("log")
    ax_box.set_ylabel("peak von Mises (MPa)")
    ax_box.set_title("spread by load case")
    ax_box.grid(alpha=0.3)

    above = int((meta[cases[0][0]] > YIELD_MPA).sum()) if cases else 0
    fig.suptitle(f"{above} of {len(meta)} models exceed yield at their peak node "
                 "-- singular corners, not strength", y=1.02)
    return _save(fig, path)


def plot_category_balance(meta: pd.DataFrame, split, path: str | Path) -> Path:
    """Category shares in train and test, side by side.

    Makes the stratification visible: the official random split ranges from 8.7% to
    30.2% against a 20% target, and a grouped stratified split should be flat.
    """
    train = meta.loc[sorted(split.train), "category"].value_counts(normalize=True)
    test = meta.loc[sorted(split.test), "category"].value_counts(normalize=True)
    categories = sorted(set(train.index) | set(test.index))
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [train.get(c, 0) for c in categories], 0.4, label="train")
    ax.bar(x + 0.2, [test.get(c, 0) for c in categories], 0.4, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("share of split")
    ax.set_title(f"category balance -- {split.name}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)
