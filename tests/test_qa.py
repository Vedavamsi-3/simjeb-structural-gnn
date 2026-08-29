"""Tests for the QA stage: frame alignment and outlier detection."""

import numpy as np
import pandas as pd
import pytest

from src.qa.alignment import (
    assess_alignment,
    fit_to_reference,
    kabsch,
    rotation_angle_deg,
)
from src.qa.outliers import (
    MAD_THRESHOLD,
    detect,
    hard_checks,
    modified_z_scores,
    summarise,
)


def rotation_z(degrees: float) -> np.ndarray:
    r = np.radians(degrees)
    return np.array([[np.cos(r), -np.sin(r), 0.0],
                     [np.sin(r), np.cos(r), 0.0],
                     [0.0, 0.0, 1.0]])


LANDMARKS = np.array([[10.0, 10.0, 0.0], [-10.0, 10.0, 0.0],
                      [-10.0, -10.0, 0.0], [10.0, -10.0, 0.0],
                      [0.0, 0.0, 40.0]])


class TestKabsch:
    def test_identical_sets_need_no_transform(self):
        _, translation, rmsd = kabsch(LANDMARKS, LANDMARKS)
        assert rmsd == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(translation, 0.0, atol=1e-9)

    def test_recovers_a_pure_translation(self):
        shift = np.array([100.0, -50.0, 25.0])
        rotation, translation, rmsd = kabsch(LANDMARKS, LANDMARKS + shift)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(translation, shift, atol=1e-9)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_recovers_a_pure_rotation(self):
        turned = LANDMARKS @ rotation_z(30.0).T
        rotation, _, rmsd = kabsch(LANDMARKS, turned)
        assert rotation_angle_deg(rotation) == pytest.approx(30.0, abs=1e-6)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_rejects_a_reflection(self):
        """A mirrored bracket is a different part, not a re-oriented one. Without the
        determinant correction the SVD would absorb the mirror into the 'rotation' and
        report a perfect fit, hiding the problem entirely."""
        mirrored = LANDMARKS * np.array([1.0, 1.0, -1.0])
        rotation, _, rmsd = kabsch(LANDMARKS, mirrored)
        assert np.linalg.det(rotation) > 0, "returned a reflection"
        assert rmsd > 1.0, "a mirror was silently absorbed into the rotation"

    def test_residual_survives_a_shape_change(self):
        """A model whose interfaces are the wrong size cannot be fixed by any rigid
        transform, so the residual is what catches it."""
        stretched = LANDMARKS * np.array([1.4, 1.0, 1.0])
        _, _, rmsd = kabsch(stretched, LANDMARKS)
        assert rmsd > 1.0


class TestAssessAlignment:
    def test_an_aligned_dataset_flags_nothing(self):
        models = {i: LANDMARKS + np.random.RandomState(i).randn(5, 3) * 1e-6
                  for i in range(20)}
        fits, outliers = assess_alignment(models)
        assert outliers == []
        assert max(f.rotation_deg for f in fits) < 0.1

    def test_translation_alone_is_not_flagged(self):
        """Translation is normalised away during graph building, so it must not
        count as an outlier -- only rotation and residual do."""
        models = {i: LANDMARKS + np.array([50.0 * i, 0.0, 0.0]) for i in range(10)}
        _, outliers = assess_alignment(models)
        assert outliers == []

    def test_a_rotated_model_is_flagged(self):
        models = {i: LANDMARKS.copy() for i in range(10)}
        models[7] = LANDMARKS @ rotation_z(25.0).T
        _, outliers = assess_alignment(models)
        assert outliers == [7]

    def test_reference_is_the_median_not_an_arbitrary_model(self):
        """Picking model 0 as the reference would declare everything else misaligned
        whenever model 0 happens to be the odd one out."""
        models = {i: LANDMARKS.copy() for i in range(10)}
        models[0] = LANDMARKS @ rotation_z(40.0).T
        _, outliers = assess_alignment(models)
        assert outliers == [0]

    def test_translation_magnitude_is_recorded(self):
        fit = fit_to_reference(1, LANDMARKS + np.array([3.0, 4.0, 0.0]), LANDMARKS)
        assert fit.translation_mm == pytest.approx(5.0)


class TestModifiedZScore:
    def test_clean_data_has_no_outliers(self):
        values = np.random.RandomState(0).normal(100, 10, 500)
        assert np.abs(modified_z_scores(values)).max() < 6

    def test_finds_an_obvious_outlier(self):
        values = np.concatenate([np.random.RandomState(0).normal(100, 5, 200), [1000.0]])
        assert np.abs(modified_z_scores(values))[-1] > MAD_THRESHOLD

    def test_resists_masking(self):
        """The reason MAD is used rather than standard deviation.

        A cluster of extreme values inflates the mean and the standard deviation until
        the threshold swallows them and nothing gets flagged. The median and MAD are
        unmoved by up to half the data being corrupt, so the same cluster stands out.
        """
        bulk = np.random.RandomState(0).normal(10.0, 1.0, 100)
        values = np.concatenate([bulk, np.full(8, 5000.0)])

        robust = np.abs(modified_z_scores(values)) > MAD_THRESHOLD
        assert robust[100:].all(), "the extreme group masked itself"
        assert not robust[:100].any(), "flagged ordinary values"

        # The same data under a plain z-score: the outliers drag the statistics far
        # enough that they no longer stand out from their own threshold.
        plain = np.abs((values - values.mean()) / values.std())
        assert plain[100:].max() < robust_max(values), \
            "plain z-score should be less sensitive here"

    def test_falls_back_when_mad_is_zero(self):
        """MAD is exactly zero whenever more than half the values are identical, which
        does not mean there are no outliers -- only that this estimator cannot see
        them. The mean-absolute-deviation fallback still catches them."""
        values = np.concatenate([np.full(100, 10.0), np.full(8, 5000.0)])
        flagged = np.abs(modified_z_scores(values)) > MAD_THRESHOLD
        assert flagged[100:].all(), "a zero MAD silently disabled detection"

    def test_constant_column_returns_zeros(self):
        # Genuinely constant: both MAD and mean-AD are zero, and there is nothing to
        # flag. Must return zeros rather than dividing by zero.
        assert np.all(modified_z_scores(np.full(50, 7.0)) == 0)


def robust_max(values: np.ndarray) -> float:
    return float(np.abs(modified_z_scores(values)).max())


def toy_metadata(n: int = 60) -> pd.DataFrame:
    rs = np.random.RandomState(0)
    frame = pd.DataFrame({
        "id": np.arange(n),
        "num_vertices": rs.randint(40_000, 60_000, n),
        "num_faces": rs.randint(80_000, 120_000, n),
        "num_tets": rs.randint(500_000, 700_000, n),
        "volume": rs.uniform(2e5, 4e5, n),
        "surface_area": rs.uniform(6e4, 9e4, n),
        "average_edge_length": rs.uniform(1.0, 1.6, n),
        "genus": rs.randint(5, 40, n).astype(float),
        "mass": rs.uniform(1.0, 2.0, n),
        "max_ver_stress": rs.uniform(300, 1500, n),
        "max_ver_magdisp": rs.uniform(0.1, 0.6, n),
        "category": rs.choice(["flat", "block", "beam"], n),
    })
    return frame.set_index("id", drop=False)


class TestHardChecks:
    def test_clean_data_passes(self):
        assert hard_checks(toy_metadata()) == {}

    def test_negative_genus_is_fatal(self):
        """Genus cannot be negative for a closed surface -- it means the mesh is
        disconnected, which breaks message passing outright."""
        meta = toy_metadata()
        meta.loc[3, "genus"] = -80.0
        failures = hard_checks(meta)
        assert 3 in failures and "genus" in failures[3]

    def test_non_finite_results_are_fatal(self):
        meta = toy_metadata()
        meta.loc[5, "max_ver_stress"] = np.nan
        assert 5 in hard_checks(meta)

    def test_zero_displacement_means_the_load_never_applied(self):
        meta = toy_metadata()
        meta.loc[9, "max_ver_magdisp"] = 0.0
        assert 9 in hard_checks(meta)


class TestDetect:
    def test_reports_both_methods_separately(self):
        report = detect(toy_metadata())
        assert set(report.by_method) == {"modified_z", "isolation_forest"}

    def test_union_covers_both(self):
        report = detect(toy_metadata())
        both = set(report.by_method["modified_z"]) | set(report.by_method["isolation_forest"])
        assert set(report.statistical) == both

    def test_isolation_forest_catches_a_bad_combination(self):
        """The case a per-metric z-score cannot see: every value individually
        unremarkable, but implausible together -- a huge part with a coarse mesh."""
        meta = toy_metadata()
        meta.loc[11, "volume"] = 4e5          # high but within range
        meta.loc[11, "num_vertices"] = 40_000  # low but within range
        report = detect(meta, contamination=0.1)
        assert 11 in report.statistical

    def test_statistical_flags_are_not_excluded(self):
        """The policy that matters: unusual is not broken. Dropping every flagged
        model removes the hard cases and inflates the test score."""
        meta = toy_metadata()
        meta.loc[4, "volume"] = 1e7
        report = detect(meta)
        assert 4 in report.statistical
        assert 4 not in report.must_exclude

    def test_only_hard_failures_are_excluded(self):
        meta = toy_metadata()
        meta.loc[6, "genus"] = -12.0
        report = detect(meta)
        assert report.must_exclude == {6}

    def test_summary_mentions_the_yield_finding(self):
        text = summarise(detect(toy_metadata()), toy_metadata())
        assert "yield" in text and "log-transformed" in text
