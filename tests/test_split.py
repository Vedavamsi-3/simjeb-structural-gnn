"""Tests for the split protocol, against the real SimJEB metadata table.

The failure this guards is not a crash. A split with leakage produces a *better*
looking number, so nothing about the run signals that anything is wrong -- which is
exactly why the properties have to be asserted rather than assumed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src.data.split import (
    Split,
    grouped_split,
    leakage_groups,
    load_metadata,
    official_split,
    select_grouped_split,
    standardized_mean_difference,
    verify_split,
)

DEFAULT_META = Path(
    r"D:\Vamsi_courses\Projects\3D_deep_learning_project_2\all_bracket_metadata.tab"
)


@pytest.fixture(scope="module")
def meta():
    path = Path(os.environ.get("SIMJEB_METADATA", DEFAULT_META))
    if not path.is_file():
        pytest.skip(f"metadata table not found at {path}; set SIMJEB_METADATA")
    return load_metadata(path)


@pytest.fixture(scope="module")
def groups(meta):
    return leakage_groups(meta)


class TestMetadata:
    def test_all_381_models(self, meta):
        assert len(meta) == 381

    def test_quotes_stripped(self, meta):
        assert set(meta["category"].unique()) == {
            "flat", "block", "beam", "butterfly", "arch", "other"
        }

    def test_category_counts(self, meta):
        counts = meta["category"].value_counts().to_dict()
        assert counts == {"flat": 147, "block": 99, "beam": 46,
                          "butterfly": 43, "arch": 37, "other": 9}

    def test_three_official_splits_each_holding_out_77(self, meta):
        for i in range(3):
            is_test = meta[f"test_split_{i}"].astype(str).str.lower() == "true"
            assert int(is_test.sum()) == 77


class TestLeakageGroups:
    def test_submission_grouping_matches_the_readme(self, groups):
        """20 pairs and 4 triples, covering 52 of the 381 models -- the same counts
        found independently by parsing submission URLs out of README_fea.txt."""
        sizes: dict[int, int] = {}
        for gid in groups.values():
            sizes[gid] = sizes.get(gid, 0) + 1
        histogram: dict[int, int] = {}
        for size in sizes.values():
            histogram[size] = histogram.get(size, 0) + 1

        assert histogram[2] == 20
        assert histogram[3] == 4
        assert sum(s for s in sizes.values() if s > 1) == 52

    def test_author_grouping_is_coarser(self, meta):
        by_submission = leakage_groups(meta)
        by_author = leakage_groups(meta, use_author=True)
        assert len(set(by_author.values())) < len(set(by_submission.values()))

    def test_geometric_links_merge_groups(self, meta):
        base = leakage_groups(meta)
        a, b = 0, 4
        assert base[a] != base[b]
        merged = leakage_groups(meta, geometric_links=[(a, b)])
        assert merged[a] == merged[b], "a confirmed geometric duplicate must merge"

    def test_every_model_has_exactly_one_group(self, meta, groups):
        assert set(groups) == set(meta["id"])


class TestOfficialSplit:
    def test_sizes(self, meta, groups):
        split = official_split(meta, groups, index=0)
        assert len(split.test) == 77
        assert len(split.train) + len(split.val) == 381 - 77
        assert len(split.val) > 0

    def test_no_model_appears_twice(self, meta, groups):
        split = official_split(meta, groups, index=0)
        assert len(split.all_ids) == len(set(split.all_ids)) == 381

    def test_the_official_test_set_is_used_verbatim(self, meta, groups):
        split = official_split(meta, groups, index=0)
        expected = set(meta.loc[
            meta["test_split_0"].astype(str).str.lower() == "true", "id"
        ])
        assert set(split.test) == expected

    def test_official_splits_do_leak_by_design_family(self, meta, groups):
        """The measured finding that motivates reporting two numbers.

        This is not an accusation -- random splitting is a normal default -- but the
        official number answers a slightly different question, and this is what
        quantifies the difference.
        """
        report = verify_split(official_split(meta, groups, index=0), meta, groups)
        assert report["groups_straddling"] > 0

    def test_validation_carve_respects_groups(self, meta, groups):
        """The official splits define train and test only. If the carve ignored
        groups, early stopping would leak even though the test set did not."""
        split = official_split(meta, groups, index=0)
        train_groups = {groups[i] for i in split.train}
        val_groups = {groups[i] for i in split.val}
        assert not (train_groups & val_groups)


class TestGroupedSplit:
    def test_no_group_straddles_any_split(self, meta, groups):
        split = grouped_split(meta, groups, seed=0)
        report = verify_split(split, meta, groups)
        assert report["groups_straddling"] == 0

    def test_test_fraction_matches_the_official_size(self, meta, groups):
        """Compared on equally sized test sets, or the gap between the two numbers
        would partly reflect test-set size rather than leakage."""
        split = grouped_split(meta, groups, test_fraction=0.2, seed=0)
        assert 0.15 < len(split.test) / 381 < 0.25

    def test_every_category_appears_in_every_split(self, meta, groups):
        """The failure stratification exists to prevent: `other` has only 9 models,
        so a random draw can leave it absent from a split entirely."""
        split = grouped_split(meta, groups, seed=0)
        for name in ("train", "val", "test"):
            present = set(meta.loc[sorted(getattr(split, name)), "category"])
            assert present == set(meta["category"].unique()), f"{name} is missing a category"

    def test_categories_are_better_balanced_than_the_official_split(self, meta, groups):
        """Official test_split_0 ranges 8.7% (beam) to 30.2% (butterfly) against a 20%
        target. Stratifying should tighten that spread."""
        def spread(split):
            report = verify_split(split, meta, groups)
            train, test = report["category_share_train"], report["category_share_test"]
            return max(abs(test[c] - train[c]) for c in test)

        assert spread(grouped_split(meta, groups, seed=0)) < spread(
            official_split(meta, groups, index=0)
        )

    def test_deterministic_for_a_given_seed(self, meta, groups):
        a = grouped_split(meta, groups, seed=7)
        b = grouped_split(meta, groups, seed=7)
        assert a.train == b.train and a.test == b.test

    def test_different_seeds_give_different_splits(self, meta, groups):
        assert grouped_split(meta, groups, seed=0).test != \
               grouped_split(meta, groups, seed=1).test

    def test_excluded_models_appear_nowhere(self, meta, groups):
        """QA exclusions -- broken meshes, rotational outliers -- must not reappear."""
        split = grouped_split(meta, groups, exclude={0, 4, 6}, seed=0)
        assert not ({0, 4, 6} & set(split.all_ids))
        assert len(split.all_ids) == 378


class TestSeedSelection:
    """Stratifying on the category label does not balance the continuous descriptors.

    Across ten seeds the size balance ranges from 0.039 to 0.299 -- so which seed you
    happen to use decides whether the test set is representative. Picking the best is
    cheap and legitimate, but only because the criterion is fixed before training.
    """

    def test_beats_an_arbitrary_seed(self, meta, groups):
        chosen = select_grouped_split(meta, groups, seeds=range(10))
        arbitrary = verify_split(grouped_split(meta, groups, seed=0), meta, groups)
        assert chosen.verification["smd_max"] <= arbitrary["smd_max"]

    def test_result_is_actually_balanced(self, meta, groups):
        chosen = select_grouped_split(meta, groups, seeds=range(10))
        assert chosen.verification["smd_max"] < 0.1, "conventional balance threshold"

    def test_leakage_is_still_zero(self, meta, groups):
        """Balance must never be bought by relaxing the grouping constraint."""
        chosen = select_grouped_split(meta, groups, seeds=range(10))
        assert chosen.verification["groups_straddling"] == 0

    def test_selection_is_recorded_for_audit(self, meta, groups):
        """The decision has to be inspectable, or it is indistinguishable from having
        picked the split that gave the nicest result."""
        sel = select_grouped_split(meta, groups, seeds=range(5)).provenance["seed_selection"]
        assert sel["decided_before_training"] is True
        assert "criterion" in sel
        assert len(sel["smd_by_seed"]) == 5
        assert sel["chosen_smd_max"] == min(sel["smd_by_seed"].values())

    def test_deterministic(self, meta, groups):
        a = select_grouped_split(meta, groups, seeds=range(10))
        b = select_grouped_split(meta, groups, seeds=range(10))
        assert a.test == b.test
        assert a.provenance["seed_selection"]["chosen_seed"] == \
               b.provenance["seed_selection"]["chosen_seed"]


class TestVerification:
    def test_smd_is_zero_for_identical_samples(self):
        x = np.random.RandomState(0).randn(100)
        assert standardized_mean_difference(x, x) == 0.0

    def test_smd_grows_with_separation(self):
        rs = np.random.RandomState(0)
        a = rs.randn(200)
        assert standardized_mean_difference(a, a + 0.5) < \
               standardized_mean_difference(a, a + 2.0)

    def test_smd_handles_a_constant_variable(self):
        # Would divide by zero without the guard.
        c = np.full(50, 3.0)
        assert standardized_mean_difference(c, c) == 0.0

    def test_grouped_split_reports_balanced(self, meta, groups):
        report = verify_split(grouped_split(meta, groups, seed=0), meta, groups)
        assert report["smd_max"] is not None
        assert report["n_train"] + report["n_val"] + report["n_test"] == 381

    def test_report_counts_add_up(self, meta, groups):
        split = grouped_split(meta, groups, seed=0)
        report = verify_split(split, meta, groups)
        assert report["n_train"] == len(split.train)
        assert report["n_test"] == len(split.test)


class TestPersistence:
    def test_round_trip(self, meta, groups, tmp_path):
        """A split that changes between runs makes every comparison meaningless --
        including the A/B/C/C-prime comparison the project is built around."""
        split = grouped_split(meta, groups, seed=3)
        split.verification = verify_split(split, meta, groups)
        path = split.save(tmp_path / "grouped_split_v1.json")

        restored = Split.load(path)
        assert restored.train == split.train
        assert restored.val == split.val
        assert restored.test == split.test
        assert restored.seed == split.seed
        assert restored.provenance["stratified_on"] == "category"

    def test_split_of_lookup(self, meta, groups):
        split = grouped_split(meta, groups, seed=0)
        assert split.split_of(split.train[0]) == "train"
        assert split.split_of(split.test[0]) == "test"
        with pytest.raises(KeyError):
            split.split_of(999_999)
