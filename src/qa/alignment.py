"""Verify that the 381 brackets really do share a coordinate frame.

SimJEB states the models are pre-aligned. This measures it rather than trusting it,
using the five standardised interfaces -- four bolt holes and the load lug -- as
landmarks. The bracket *body* is meant to differ between designs, so a bounding box or
a PCA of the full point cloud varies for legitimate reasons and cannot separate
"different shape" from "misplaced". The interfaces are the only fair reference.

Translation and rotation are then treated differently, for a physical reason:

* A **translation** changes nothing -- geometry, load direction and clamps are all
  unchanged, only the coordinates shift. It is normalised away during graph building
  and costs nothing.
* A **rotation** is not neutral. The load is a fixed global vector, identical in all
  381 decks, and it does not rotate with the part. A turned bracket under that same
  global load carries its force along a different internal path: a different problem,
  not a different view of the same one. Those models are excluded, never re-oriented.

Re-orienting would also be unsafe in a way that leaves no trace. Rotating the geometry
back does not rotate the load that produced the stress field, so the result would be an
aligned mesh whose answers came from a different loading -- a corrupt sample that no
longer shows up as an outlier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidFit:
    """The rigid transform taking one model's landmarks onto the reference."""

    model_id: int
    rotation_deg: float      # magnitude of the rotation, in degrees
    translation_mm: float    # magnitude of the translation
    rmsd_mm: float           # residual after the best rigid fit
    determinant: float       # +1 for a rotation; a reflection would be -1

    @property
    def is_reflected(self) -> bool:
        return self.determinant < 0


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Best-fit rigid transform mapping ``source`` onto ``target``.

    Returns ``(rotation, translation, rmsd)``. Both inputs are ``(n_points, 3)``.

    The determinant correction below is not cosmetic. Without it the SVD can return a
    matrix with determinant -1 -- a reflection, not a rotation -- which would silently
    "align" a mirrored bracket by turning a left-hand design into a right-hand one.
    Forcing a proper rotation means a mirrored model shows up as a large residual
    instead of being quietly absorbed.
    """
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    p = source - source_centre
    q = target - target_centre

    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    translation = target_centre - rotation @ source_centre
    residual = (rotation @ p.T).T - q
    rmsd = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
    return rotation, translation, rmsd


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Rotation magnitude in degrees, from the matrix trace."""
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def fit_to_reference(model_id: int, landmarks: np.ndarray,
                     reference: np.ndarray) -> RigidFit:
    """Measure how far one model's interfaces sit from the reference model's."""
    rotation, translation, rmsd = kabsch(landmarks, reference)
    return RigidFit(
        model_id=model_id,
        rotation_deg=rotation_angle_deg(rotation),
        translation_mm=float(np.linalg.norm(translation)),
        rmsd_mm=rmsd,
        determinant=float(np.linalg.det(rotation)),
    )


def assess_alignment(landmarks_by_model: dict[int, np.ndarray],
                     reference_id: int | None = None,
                     rotation_tol_deg: float = 2.0,
                     rmsd_tol_mm: float = 2.0) -> tuple[list[RigidFit], list[int]]:
    """Fit every model to a reference and flag the ones that need rotating.

    Returns ``(fits, rotation_outliers)``.

    The reference defaults to the model whose landmarks are closest to the elementwise
    median across the dataset -- picking an arbitrary model risks choosing an outlier
    as the standard and declaring everything else misaligned.

    Tolerances are deliberately loose. A genuinely aligned dataset produces angles at
    the numerical-noise level, so anything above a couple of degrees is a real
    difference rather than a borderline call.
    """
    ids = sorted(landmarks_by_model)
    stacked = np.stack([landmarks_by_model[i] for i in ids])

    if reference_id is None:
        median = np.median(stacked, axis=0)
        distances = np.linalg.norm(stacked - median, axis=(1, 2))
        reference_id = ids[int(np.argmin(distances))]
    reference = landmarks_by_model[reference_id]

    fits = [fit_to_reference(i, landmarks_by_model[i], reference) for i in ids]
    outliers = [
        f.model_id for f in fits
        if f.rotation_deg > rotation_tol_deg or f.rmsd_mm > rmsd_tol_mm
    ]
    return fits, outliers
